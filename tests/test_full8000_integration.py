"""Full-8000 integration tests; all identities, labels and tensors fabricated."""
import copy
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from evaluation.reporting import (CandidateSpec, EvaluationManifest, Sample, WP1_NAMESPACE,
                                  candidate_digest, fingerprint, canonical_json)
from evaluation.final_manifest import load_full8000_manifest, full8000_inference_order
from evaluation.wp2_handoff import BaseOwnerHandoff
from evaluation.model_adapters import ModelOutputAdapter, WP2_CONVNEXT
from scripts.import_wp5_final import import_full8000_wp2
from test_wp2_handoff import fixture
from evaluation.packet_io import decode_json, write_new_directory

ROOT = Path(os.environ.get('WP2_TEST_ARTIFACTS', Path.cwd()/'SYNTHETIC_ONLY_TEST_ARTIFACTS'))
ROOT.mkdir(parents=True, exist_ok=True)
PROTOCOL = ('cos30082-official-shared333', 'full8000-20261112-v1')
FITTING_NAMESPACE = 'full8000-original-ordered-filenames-json-sha256-v1'


def dump(path, data):
    path.write_text(canonical_json(data)+'\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def full_fixture():
    manifest, run, owner, outputs = fixture()
    fitting_sha = fingerprint([s.sample_id for s in manifest.samples if s.fold == 'fitting'])
    spec = replace(manifest.spec, protocol_id=PROTOCOL[0], protocol_version=PROTOCOL[1],
                   split_hash_namespace=FITTING_NAMESPACE, split_sha256=fitting_sha)
    manifest = replace(manifest, spec=spec)
    owner['ordered_fitting_names_sha256'] = fitting_sha
    return manifest, run, owner, outputs


class PatchReviewRegressionTests(unittest.TestCase):
    def test_exponent_overflow_is_rejected_as_nonfinite_JSON(self):
        for raw in [b'{"value":1e999}', b'{"value":-1e999}']:
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, 'Nonfinite'):
                decode_json(raw)
        self.assertEqual(decode_json(b'{"value":1e3}'), {'value':1000.0})

    def test_interrupted_completion_write_never_exposes_a_partial_marker(self):
        original_open = Path.open
        class InterruptedWrite:
            def __init__(self, stream): self.stream = stream
            def __enter__(self): self.stream.__enter__(); return self
            def __exit__(self, *args): return self.stream.__exit__(*args)
            def write(self, raw):
                self.stream.write(raw[:4]); self.stream.flush()
                raise OSError('synthetic interrupted completion write')
        def opening(path, mode='r', *args, **kwargs):
            stream = original_open(path, mode, *args, **kwargs)
            return InterruptedWrite(stream) if mode == 'xb' and path.name != 'payload.bin' else stream
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            output = Path(temp)/'packet'
            with patch.object(Path, 'open', opening), self.assertRaises(OSError):
                write_new_directory(output, {'payload.bin':b'payload', 'COMPLETED.json':b'{"complete":true}'},
                                    completion_name='COMPLETED.json')
            self.assertFalse((output/'COMPLETED.json').exists())
            self.assertEqual((output/'payload.bin').read_bytes(), b'payload')

    def test_overlapping_output_and_registry_are_rejected_before_input_IO(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = Path(temp)
            for output, registry in [(root/'same',root/'same'), (root/'out',root/'out/registry'),
                                     (root/'registry/out',root/'registry')]:
                with self.subTest(output=output), patch('evaluation.full8000_packets.read_json') as reader:
                    reader.side_effect = AssertionError('input must not be read for invalid destinations')
                    with self.assertRaisesRegex(ValueError, 'disjoint'):
                        import_full8000_wp2(root/'unused.json','0'*64,output,registry_path=registry)
                    reader.assert_not_called()
                self.assertFalse(output.exists()); self.assertFalse(registry.exists())


class ManifestVersionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.run_provenance, cls.owner, cls.outputs = full_fixture()

    def test_versioned_full8000_roundtrip_and_semantic_pin(self):
        result = load_full8000_manifest(self.manifest.to_dict(), expected_semantic_sha256=self.manifest.sha256,
                                        allow_synthetic=True)
        self.assertEqual(result, self.manifest)
        self.assertEqual(result.sha256, fingerprint(asdict(result)))

    def test_synthetic_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError, 'synthetic'):
            load_full8000_manifest(self.manifest.to_dict(), expected_semantic_sha256=self.manifest.sha256)

    def test_wrong_protocol_version_rejected(self):
        with self.assertRaisesRegex(ValueError, 'version'):
            replace(self.manifest, spec=replace(self.manifest.spec, protocol_version='full8000-OTHER'))

    def test_historical_fitting_count_cannot_enter_full8000(self):
        payload = self.manifest.to_dict()
        payload['samples'] = [s for s in payload['samples'] if s['fold'] != 'fitting']+[
            s for s in payload['samples'] if s['fold'] == 'fitting'][:4868]
        with self.assertRaisesRegex(ValueError, '8,000'):
            EvaluationManifest.from_dict(payload)

    def test_fitting_reorder_rejected_by_existing_fingerprint(self):
        payload = json.loads(canonical_json(self.manifest.to_dict())); payload['samples'][0], payload['samples'][1] = payload['samples'][1], payload['samples'][0]
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            EvaluationManifest.from_dict(payload)

    def test_test_row_reorder_rejected_by_semantic_pin(self):
        payload = json.loads(canonical_json(self.manifest.to_dict())); payload['samples'][8000], payload['samples'][8001] = payload['samples'][8001], payload['samples'][8000]
        with self.assertRaisesRegex(ValueError, 'semantic'):
            load_full8000_manifest(payload, expected_semantic_sha256=self.manifest.sha256, allow_synthetic=True)

    def test_missing_duplicate_extra_and_cross_fold_records(self):
        for mutation in ['missing', 'duplicate', 'extra', 'fold']:
            payload = json.loads(canonical_json(self.manifest.to_dict()))
            if mutation == 'missing': payload['samples'].pop()
            elif mutation == 'duplicate': payload['samples'][-1] = payload['samples'][-2]
            elif mutation == 'extra': payload['samples'].append(dict(sample_id='SYNTHETIC_ONLY_EXTRA', fold='unseen', truth_id=323))
            else: payload['samples'][8000]['fold'] = 'unseen'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                load_full8000_manifest(payload, expected_semantic_sha256=self.manifest.sha256, allow_synthetic=True)

    def test_wrong_candidate_hash_rejected(self):
        payload = self.manifest.to_dict(); payload['spec']['candidate_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            EvaluationManifest.from_dict(payload)

    def test_real_mode_rejects_fabricated_candidate_identity(self):
        payload = self.manifest.to_dict(); payload['spec']['synthetic'] = False
        with self.assertRaises(ValueError):
            load_full8000_manifest(payload, expected_semantic_sha256=self.manifest.sha256)

    def test_historical_real_mode_4868_guard_preserved_with_synthetic_identities(self):
        candidates = self.manifest.spec.candidates[:323]
        spec = CandidateSpec('HISTORICAL_SYNTHETIC_IDENTITIES', 'historical-v1', 'development', candidates,
                             candidate_digest(candidates, WP1_NAMESPACE), WP1_NAMESPACE, '3'*64, '4'*64,
                             'HISTORICAL_NAMESPACE', 'fabricated metadata; no real data', (), synthetic=False)
        def records(fitting_count):
            return tuple(Sample(f'SYNTHETIC_{fold}_{i}',fold,i%258 if fold!='unseen' else 258+i%65)
                         for fold,count in [('fitting',fitting_count),('seen',1238),('unseen',1894)] for i in range(count))
        old = EvaluationManifest(spec,records(4868),tuple(range(258)),tuple(range(258,323)),1238,1894,'synthetic identities in real-mode guard')
        self.assertEqual(sum(s.fold=='fitting' for s in old.samples),4868)
        with self.assertRaisesRegex(ValueError,'4,868'):
            replace(old,samples=records(8000))

    def test_label_free_order_contains_no_per_image_truths(self):
        order = full8000_inference_order(self.manifest)
        self.assertEqual(len(order['folds']['seen']), 2000)
        self.assertEqual(len(order['folds']['unseen']), 1448)
        forbidden = {'truth_id', 'truths', 'ground_truth_ids', 'samples', 'seen_truth_ids', 'unseen_truth_ids'}
        def check(value):
            if isinstance(value, dict):
                self.assertFalse(forbidden & value.keys())
                for item in value.values(): check(item)
            elif isinstance(value, list):
                for item in value: check(item)
        check(order)


class OwnerImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.run_provenance, cls.owner, cls.outputs = full_fixture()
        cls.order = full8000_inference_order(cls.manifest)
        cls.ids = cls.manifest.spec.ids
        cls.pairs = [(c.crop_id, c.disease_id) for c in cls.manifest.spec.candidates]
        cls.temp = tempfile.TemporaryDirectory(prefix='SYNTHETIC_ONLY_RELEASE_', dir=ROOT)
        cls.root = Path(cls.temp.name)
        cls.packet = cls.root/'OWNER_PACKET'
        collector = cls.collector()
        for fold, out in cls.outputs.items():
            names = cls.order['folds'][fold]
            for start in range(0, len(names), 257):
                collector.add_batch(fold, {k: v[start:start+257] for k,v in out.items()}, names[start:start+257], cls.ids, cls.pairs)
        cls.receipt_pin = collector.export(cls.packet)
        cls.collector_instance = collector
        dump(cls.root/'manifest.json', cls.manifest.to_dict())
        cls.bundle = dict(schema='wp5-full8000-wp2-import-v1', protocol_id=PROTOCOL[0], protocol_version=PROTOCOL[1],
                          identity_manifest=dict(path='manifest.json', file_sha256=sha(cls.root/'manifest.json')),
                          manifest_sha256=cls.manifest.sha256, inference_order_sha256=fingerprint(cls.order),
                          owner_packet=dict(format='directory', path='OWNER_PACKET', receipt_sha256=cls.receipt_pin),
                          provenance=asdict(cls.run_provenance), reviewed_owner_metadata=cls.owner,
                          final_policy_reference='SYNTHETIC_ONLY_POLICY',
                          evaluation_authorization_reference='SYNTHETIC_ONLY_AUTHORIZATION',
                          owner_approval_reference='SYNTHETIC_ONLY_REVIEW')

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    @classmethod
    def collector(cls, **kwargs):
        return BaseOwnerHandoff(kwargs.get('order', cls.order), kwargs.get('run', cls.run_provenance),
                                kwargs.get('owner', cls.owner), expected_order_sha256=fingerprint(kwargs.get('order', cls.order)),
                                allow_synthetic=True)

    def case(self):
        temp = tempfile.TemporaryDirectory(prefix='case_', dir=self.root); self.addCleanup(temp.cleanup)
        root = Path(temp.name); shutil.copytree(self.packet, root/'OWNER_PACKET')
        shutil.copyfile(self.root/'manifest.json', root/'manifest.json')
        return root, copy.deepcopy(self.bundle)

    def import_case(self, root, bundle, output='WP5_OUTPUT', registry='REGISTRY'):
        path = root/'handoff.json'; dump(path, bundle)
        return import_full8000_wp2(path, sha(path), root/output, registry_path=root/registry, allow_synthetic=True)

    def refresh(self, root, bundle):
        packet = root/'OWNER_PACKET'
        receipt = json.loads((packet/'COMPLETED.json').read_text())
        receipt['artifacts'] = {p.name:sha(p) for p in packet.iterdir() if p.name != 'COMPLETED.json'}
        dump(packet/'COMPLETED.json', receipt); bundle['owner_packet']['receipt_sha256'] = sha(packet/'COMPLETED.json')

    def mutate_npz(self, root, field, fn):
        path = root/'OWNER_PACKET/seen_scores.npz'
        with np.load(path, allow_pickle=False) as z: arrays = {k:z[k] for k in z.files}
        arrays[field] = fn(arrays[field]); np.savez_compressed(path, **arrays)

    def test_complete_owner_to_WP5_flow_summary_and_confusion(self):
        root, bundle = self.case(); receipt = self.import_case(root, bundle)
        summary = json.loads((root/'WP5_OUTPUT/summary.json').read_text())
        self.assertEqual(summary['overall']['count'], 3448)
        self.assertFalse(receipt['checkpoint_bytes_verified'])
        self.assertEqual(receipt['status'], 'SAVED_SCORE_VALIDATED_NOT_MODEL_REPRODUCTION')
        self.assertIn('evaluation/full8000_packets.py', receipt['verification_code_sha256'])
        for fold in ['seen', 'unseen']:
            with np.load(root/'OWNER_PACKET'/f'{fold}_scores.npz', allow_pickle=False) as z:
                truth = np.array([s.truth_id for s in self.manifest.samples if s.fold == fold])
                ranks = self.ids and np.asarray(self.ids)[np.argsort(-z['scores'], axis=1, kind='stable')[:,:5]]
                self.assertEqual(summary[fold]['correct_top1'], int((ranks[:,0] == truth).sum()))
                self.assertEqual(summary[fold]['correct_top5'], int((ranks == truth[:,None]).any(axis=1).sum()))
                internal=json.loads((root/'WP5_OUTPUT'/f'{fold}_validated_packet.json').read_text())
                np.testing.assert_array_equal(np.asarray(internal['scores']),z['scores'])
                adapter=ModelOutputAdapter(self.manifest,self.run_provenance,WP2_CONVNEXT,'SYNTHETIC_ONLY_AUTHORIZATION')
                self.assertEqual(len(adapter.adapt(internal)),len(truth))
        report = json.loads((root/'WP5_OUTPUT/report.json').read_text())
        self.assertEqual(report['seen']['confusion_matrix']['count'], 2000)
        self.assertEqual(report['unseen']['confusion_matrix']['count'], 1448)

    def test_export_contains_no_attached_truths_or_metrics(self):
        self.assertEqual({p.name for p in self.packet.iterdir()}, {'seen_packet.json','unseen_packet.json',
                         'seen_scores.npz','unseen_scores.npz','provenance.json','COMPLETED.json'})
        for fold in ['seen','unseen']:
            with np.load(self.packet/f'{fold}_scores.npz', allow_pickle=False) as z:
                self.assertEqual(set(z.files), {'scores','filenames','candidate_ids','top5_ids'})
            packet = json.loads((self.packet/f'{fold}_packet.json').read_text())
            self.assertNotIn('ground_truth_ids', packet)

    def test_export_bytes_and_WP5_summary_are_deterministic(self):
        root, _ = self.case(); collector = self.collector()
        for fold,out in self.outputs.items(): collector.add_batch(fold,out,self.order['folds'][fold],self.ids,self.pairs)
        pin = collector.export(root/'SECOND_EXPORT')
        self.assertEqual(pin, self.receipt_pin)
        for name in ['seen_scores.npz','unseen_scores.npz','COMPLETED.json']:
            self.assertEqual(sha(root/'SECOND_EXPORT'/name), sha(self.packet/name))
        a, b = self.case(), self.case()
        ra, rb = self.import_case(*a), self.import_case(*b)
        self.assertEqual(ra['summary_sha256'], rb['summary_sha256'])

    def test_duplicate_import_rejected_across_output_directories(self):
        root,bundle = self.case(); self.import_case(root,bundle)
        with self.assertRaisesRegex(FileExistsError, 'already imported|claim'):
            self.import_case(root,bundle,output='SECOND_WP5_OUTPUT')
        self.assertFalse((root/'SECOND_WP5_OUTPUT').exists())

    def test_wrong_protocol_and_historical_checkpoint(self):
        for key,value in [('protocol_version','historical-v1'), ('checkpoint','7a0460b7d03bc08d19b68ca06cdd266746953322149a3edad177cf29992d7c04')]:
            root,bundle = self.case()
            if key == 'checkpoint': bundle['provenance']['checkpoint_sha256'] = value
            else: bundle[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.import_case(root,bundle)
            self.assertFalse((root/'WP5_OUTPUT').exists())

    def test_conflicting_model_identity(self):
        root,bundle = self.case(); p=root/'OWNER_PACKET/seen_packet.json'; packet=json.loads(p.read_text()); packet['model_kind']='wp1_independent_pair'; dump(p,packet); self.refresh(root,bundle)
        with self.assertRaisesRegex(ValueError, 'model|provenance'): self.import_case(root,bundle)

    def test_wrong_candidate_order_and_hash(self):
        for field in ['candidate_ids','candidate_sha256']:
            root,bundle = self.case()
            if field == 'candidate_ids': self.mutate_npz(root,field,lambda x:x[::-1])
            else:
                p=root/'OWNER_PACKET/seen_packet.json'; packet=json.loads(p.read_text()); packet[field]='0'*64; dump(p,packet)
            self.refresh(root,bundle)
            with self.subTest(field=field), self.assertRaises(ValueError): self.import_case(root,bundle)

    def test_score_precision_nonfinite_shape_and_saved_ranks(self):
        mutations=[('scores',lambda x:x.astype(np.float32)), ('scores',lambda x:np.full_like(x,np.nan)),
                   ('scores',lambda x:x[:,:-1]), ('top5_ids',lambda x:x[:,::-1])]
        for field,fn in mutations:
            root,bundle=self.case(); self.mutate_npz(root,field,fn); self.refresh(root,bundle)
            with self.subTest(field=field), self.assertRaises(ValueError): self.import_case(root,bundle)

    def test_missing_duplicate_and_fold_swapped_names(self):
        for mutation in ['missing','duplicate','fold']:
            root,bundle=self.case(); p=root/'OWNER_PACKET/seen_packet.json'; packet=json.loads(p.read_text())
            if mutation=='missing': packet['image_names'].pop()
            elif mutation=='duplicate': packet['image_names'][0]=packet['image_names'][1]
            else: packet['image_names'][0]=self.order['folds']['unseen'][0]
            dump(p,packet); self.refresh(root,bundle)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): self.import_case(root,bundle)

    def test_labels_and_extra_owner_files_rejected(self):
        root,bundle=self.case(); p=root/'OWNER_PACKET/seen_packet.json'; packet=json.loads(p.read_text()); packet['ground_truth_ids']=[0]*2000; dump(p,packet); self.refresh(root,bundle)
        with self.assertRaisesRegex(ValueError, 'label|fields'): self.import_case(root,bundle)
        root,bundle=self.case(); (root/'OWNER_PACKET/unexpected.txt').write_text('extra'); self.refresh(root,bundle)
        with self.assertRaisesRegex(ValueError, 'files|layout'): self.import_case(root,bundle)

    def test_owner_identity_and_finiteness_fail_before_export(self):
        for mutation in ['duplicate','order','nan','columns']:
            collector=self.collector(); names=self.order['folds']['seen'][:2].copy()
            out={k:v[:2].clone() for k,v in self.outputs['seen'].items()}; ids=list(self.ids)
            if mutation=='duplicate': names[1]=names[0]
            elif mutation=='order': names.reverse()
            elif mutation=='nan': out['crop_logits'][0,0]=float('nan')
            else: ids.reverse()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): collector.add_batch('seen',out,names,ids,self.pairs)

    def test_partial_export_and_stale_companion_hash_fail(self):
        root,bundle=self.case(); (root/'OWNER_PACKET/COMPLETED.json').unlink()
        with self.assertRaises((ValueError,FileNotFoundError)): self.import_case(root,bundle)
        root,bundle=self.case(); (root/'OWNER_PACKET/seen_scores.npz').write_bytes(b'partial')
        with self.assertRaisesRegex(ValueError,'hash'): self.import_case(root,bundle)

    def test_interrupted_owner_write_has_no_completion_and_cannot_overwrite(self):
        root,_=self.case(); collector=self.collector()
        for fold,out in self.outputs.items(): collector.add_batch(fold,out,self.order['folds'][fold],self.ids,self.pairs)
        with patch('evaluation.packet_io.os.fsync',side_effect=OSError('synthetic disk failure')):
            with self.assertRaises(OSError): collector.export(root/'INTERRUPTED')
        self.assertFalse((root/'INTERRUPTED/COMPLETED.json').exists())
        with self.assertRaises(FileExistsError): collector.export(root/'INTERRUPTED')

    def test_interrupted_intake_retains_fail_closed_registry_claim(self):
        root,bundle=self.case()
        with patch('evaluation.full8000_packets.write_new_directory',side_effect=OSError('synthetic write failure')):
            with self.assertRaises(OSError): self.import_case(root,bundle)
        with self.assertRaisesRegex(FileExistsError,'claim'):
            self.import_case(root,bundle,output='RETRY')
        self.assertFalse((root/'RETRY/receipt.json').exists())

    def test_oversized_NPY_header_and_object_array_rejected_before_allocation(self):
        import io
        for kind in ['huge','object']:
            root,bundle=self.case(); path=root/'OWNER_PACKET/seen_scores.npz'
            with np.load(path,allow_pickle=False) as z: arrays={k:z[k] for k in z.files}
            if kind=='object':
                arrays['scores']=np.array([object()],dtype=object); np.savez_compressed(path,**arrays)
            else:
                header=io.BytesIO(); np.lib.format.write_array_header_1_0(header,dict(descr='<f8',fortran_order=False,shape=(2**40,333)))
                original=io.BytesIO(path.read_bytes())
                with zipfile.ZipFile(original) as source:
                    members={name:source.read(name) for name in source.namelist()}
                members['scores.npy']=header.getvalue()
                with zipfile.ZipFile(path,'w') as target:
                    for name,raw in members.items(): target.writestr(name,raw)
            self.refresh(root,bundle)
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError,'Unsafe array'):
                self.import_case(root,bundle)

    def test_missing_approval_and_wrong_external_pin_rejected(self):
        root,bundle=self.case(); bundle['owner_approval_reference']='PENDING_OWNER'
        with self.assertRaisesRegex(ValueError,'reference'): self.import_case(root,bundle)
        root,bundle=self.case(); path=root/'handoff.json'; dump(path,bundle)
        with self.assertRaisesRegex(ValueError,'hash'):
            import_full8000_wp2(path,'0'*64,root/'WP5_OUTPUT',registry_path=root/'REGISTRY',allow_synthetic=True)

    def test_optional_checkpoint_bytes_are_never_inferred_from_declarations(self):
        root,bundle=self.case(); path=root/'handoff.json'; dump(path,bundle)
        weights=root/'SYNTHETIC_NOT_WEIGHTS'; weights.write_bytes(b'not a real checkpoint')
        with self.assertRaisesRegex(ValueError,'Checkpoint bytes differ'):
            import_full8000_wp2(path,sha(path),root/'WP5_OUTPUT',registry_path=root/'REGISTRY',
                                 allow_synthetic=True,checkpoint_file=weights)

    def test_malformed_JSON_and_duplicate_keys_rejected(self):
        root,bundle=self.case(); (root/'OWNER_PACKET/provenance.json').write_text('{"duplicate":1,"duplicate":2}')
        self.refresh(root,bundle)
        with self.assertRaisesRegex(ValueError, 'Duplicate'): self.import_case(root,bundle)

    def test_incomplete_owner_export_does_not_create_directory(self):
        path=self.root/'INCOMPLETE'
        with self.assertRaisesRegex(ValueError, 'Incomplete'): self.collector().export(path)
        self.assertFalse(path.exists())

    def test_owner_order_with_attached_truths_is_rejected(self):
        order=copy.deepcopy(self.order); order['ground_truth_ids']=[0]*3448
        with self.assertRaisesRegex(ValueError,'label-free'):
            self.collector(order=order)

    def test_manifest_raw_hash_is_separate_from_semantic_identity(self):
        root,bundle=self.case(); path=root/'manifest.json'; path.write_text(json.dumps(self.manifest.to_dict(),indent=3))
        self.assertEqual(EvaluationManifest.from_dict(json.loads(path.read_text())).sha256,self.manifest.sha256)
        with self.assertRaisesRegex(ValueError,'hash'): self.import_case(root,bundle)

    def test_malformed_and_symlink_archives_fail_closed(self):
        import stat
        for kind in ['malformed','symlink']:
            root,bundle=self.case(); path=root/'unsafe.zip'
            if kind=='malformed': path.write_bytes(b'not a zip')
            else:
                info=zipfile.ZipInfo('seen_packet.json'); info.create_system=3
                info.external_attr=(stat.S_IFLNK|0o777)<<16
                with zipfile.ZipFile(path,'w') as archive: archive.writestr(info,'../outside')
            bundle['owner_packet']=dict(format='zip',path='unsafe.zip',archive_sha256=sha(path),receipt_sha256=self.receipt_pin)
            with self.subTest(kind=kind), self.assertRaises(ValueError): self.import_case(root,bundle)

    def test_CLI_selects_new_protocol_and_requires_registry(self):
        import io
        from scripts.import_wp5_final import main
        args=['--handoff','SYNTHETIC.json','--handoff-sha256','0'*64,'--output','SYNTHETIC_OUTPUT',
              '--protocol',PROTOCOL[1],'--allow-synthetic']
        with patch('scripts.import_wp5_final.import_full8000_wp2') as importer:
            with patch('sys.stderr',new_callable=io.StringIO), self.assertRaises(SystemExit): main(args)
            importer.assert_not_called()
            importer.return_value={'synthetic':True}
            with patch('sys.stdout',new_callable=io.StringIO) as stream:
                self.assertEqual(main(args+['--registry','SYNTHETIC_REGISTRY']),0)
                self.assertIn('four-model acceptance pending',stream.getvalue())
            self.assertTrue(importer.call_args.kwargs['allow_synthetic'])

    def test_archive_paths_and_case_collisions_fail_before_extraction(self):
        cases=[['../escape'], ['/absolute'], ['C:/drive'], ['dir\\escape'], ['a/../../escape'],
               ['CON'], ['trailing.'], ['same','SAME']]
        for index,names in enumerate(cases):
            root,bundle=self.case(); path=root/'unsafe.zip'
            with zipfile.ZipFile(path,'w') as z:
                for name in names: z.writestr(name,b'not model data')
            bundle['owner_packet']=dict(format='zip',path='unsafe.zip',archive_sha256=sha(path),receipt_sha256=self.receipt_pin)
            with self.subTest(index=index), self.assertRaises(ValueError): self.import_case(root,bundle)
            self.assertFalse((root/'escape').exists()); self.assertFalse((root/'WP5_OUTPUT').exists())

    def test_valid_archive_packet_imports_without_executing_code(self):
        root,bundle=self.case(); path=root/'owner.zip'
        with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted((root/'OWNER_PACKET').iterdir()): z.write(p,p.name)
        bundle['owner_packet']=dict(format='zip',path='owner.zip',archive_sha256=sha(path),receipt_sha256=self.receipt_pin)
        receipt=self.import_case(root,bundle); self.assertTrue(receipt['label_free_owner_packet'])


if __name__=='__main__': unittest.main(verbosity=2)
