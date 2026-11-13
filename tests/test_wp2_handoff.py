"""Synthetic-only bridge tests: no ConvNeXt construction or image access.

Set WP2_AUDIT_ROOT to additionally exercise the preserved independent WP5 gate.
Those gate tests are separate from the normal project tests and need no private
assets. Artifacts use synthetic names and stay under WP2_TEST_ARTIFACTS/cwd.
"""
import copy
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from evaluation.reporting import (
    Candidate, CandidateSpec, EvaluationManifest, RunProvenance, Sample,
    FINAL_NAMESPACE, canonical_json, fingerprint, candidate_digest,
)
from evaluation.wp2_handoff import BaseHandoff, INFERENCE_PREPROCESSING
from evaluation import wp2_base_compat as base

torch.set_num_threads(1)
ARTIFACTS = Path(os.environ.get('WP2_TEST_ARTIFACTS', Path.cwd() / 'SYNTHETIC_ONLY_TEST_ARTIFACTS'))
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture():
    candidates = tuple(Candidate(i, i % 55, i % 175, 'SYNTHETIC_CROP_'+str(i % 55),
                                 'SYNTHETIC_DISEASE_'+str(i % 175), 'SYNTHETIC_PAIR_'+str(i))
                       for i in range(333))
    spec = CandidateSpec('SYNTHETIC_ONLY_PROTOCOL', 'SYNTHETIC_ONLY_VERSION', 'final', candidates,
                         candidate_digest(candidates, FINAL_NAMESPACE), FINAL_NAMESPACE, '3'*64, '4'*64,
                         'SYNTHETIC_ONLY_SPLIT', 'SYNTHETIC_ONLY_SOURCE', (), synthetic=True,
                         approval_reference='SYNTHETIC_ONLY_NOT_REAL_APPROVAL', final_prompt_json='{"synthetic":true}')
    samples = tuple(Sample(f'SYNTHETIC_ONLY_{fold}_{i:05d}.synthetic', fold,
                           i % 323 if fold != 'unseen' else 323+i % 10)
                    for fold, count in [('fitting', 8000), ('seen', 2000), ('unseen', 1448)]
                    for i in range(count))
    manifest = EvaluationManifest(spec, samples, tuple(range(323)), tuple(range(323, 333)),
                                  2000, 1448, 'SYNTHETIC_ONLY_IDENTITY_SOURCE')
    runtime = dict(python='SYNTHETIC_TEST_DECLARATION', torch='SYNTHETIC_TEST_DECLARATION',
                   torchvision='SYNTHETIC_TEST_DECLARATION', device='CPU_SYNTHETIC_NO_MODEL',
                   image_precision='NO_IMAGES', score_precision='float64')
    owner = dict(architecture=base.ARCHITECTURE, epoch=1, inference_size=224,
                 training_scope='full_8000', training_count=8000, training_source_sha256=spec.source_sha256,
                 ordered_fitting_names_sha256=spec.split_sha256,
                 initialization_identity='SYNTHETIC_ONLY_NO_CHECKPOINT', selection_rule='SYNTHETIC_ONLY_FIXED',
                 training_preprocessing={'synthetic': 'NO_TRAINING'},
                 inference_preprocessing=copy.deepcopy(INFERENCE_PREPROCESSING), runtime=runtime,
                 scorer_source_identity='SYNTHETIC_ONLY_TEST_SOURCE', training_recipe_identity='SYNTHETIC_ONLY_NO_TRAINING',
                 selection_release_receipt_identity='SYNTHETIC_ONLY_NO_REAL_RELEASE')
    run = RunProvenance('SYNTHETIC_ONLY_WP2', 'SYNTHETIC_ONLY_RUN', 'SYNTHETIC_ONLY_DECLARED_PIN_NO_BYTES',
                        base.EXPECTED_FINAL_CHECKPOINT_SHA256, canonical_json(INFERENCE_PREPROCESSING),
                        'SYNTHETIC_ONLY_CODE', canonical_json(runtime), 'sum_log_softmax_v1', 'float64',
                        'SYNTHETIC_ONLY_COMMON_BUDGET')
    generator = torch.Generator().manual_seed(9182)
    outputs = {fold: dict(crop_logits=torch.randn(n, 55, generator=generator),
                          disease_logits=torch.randn(n, 175, generator=generator))
               for fold, n in [('seen', 2000), ('unseen', 1448)]}
    for out in outputs.values():  # exact ties, including rank-five boundary
        out['crop_logits'][0].zero_()
        out['disease_logits'][0].zero_()
    return manifest, run, owner, outputs


class BridgeFixture:
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.run_provenance, cls.owner, cls.outputs = fixture()
        cls.ids = cls.manifest.spec.ids
        cls.pairs = [(c.crop_id, c.disease_id) for c in cls.manifest.spec.candidates]
        cls.rows = {f: [asdict(s) for s in cls.manifest.samples if s.fold == f] for f in cls.outputs}

    def bridge(self, **kwargs):
        return BaseHandoff(kwargs.get('manifest', self.manifest), kwargs.get('run', self.run_provenance),
                           kwargs.get('owner', self.owner))

    def add(self, bridge, fold='seen', n=4, rows=None, ids=None, pairs=None, outputs=None):
        bridge.add_batch(fold, outputs if outputs is not None else
                         {k: v[:n] for k, v in self.outputs[fold].items()},
                         rows if rows is not None else self.rows[fold][:n],
                         ids if ids is not None else self.ids, pairs if pairs is not None else self.pairs)


class BridgeTests(BridgeFixture, unittest.TestCase):
    def test_base_only_release_and_no_fallback(self):
        for field, value in [('architecture', 'multitask_convnext_tiny'), ('epoch', True), ('inference_size', 320)]:
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Base release'):
                self.bridge(owner={**self.owner, field: value})

    def test_checkpoint_declaration_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'checkpoint declaration'):
            self.bridge(run=replace(self.run_provenance, checkpoint_sha256='0'*64))

    def test_training_scope_is_owner_declared_not_inferred_from_test_manifest(self):
        for field, value in [('training_count', 4868), ('training_source_sha256', '0'*64),
                             ('ordered_fitting_names_sha256', '1'*64)]:
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Owner training declaration'):
                self.bridge(owner={**self.owner, field: value})

    def test_duplicate_batch_is_rejected(self):
        bridge = self.bridge(); self.add(bridge)
        with self.assertRaisesRegex(ValueError, 'alignment'):
            self.add(bridge)

    def test_missing_private_asset_and_unknown_schema_stay_blocked(self):
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as temp:
            path = Path(temp) / 'SYNTHETIC_NOT_A_WEIGHT_FILE'
            with self.assertRaisesRegex(FileNotFoundError, 'unavailable'):
                base.load_private_final_checkpoint(path)
            path.write_bytes(b'SYNTHETIC_NOT_WEIGHTS')
            with self.assertRaisesRegex(base.UnverifiedPrivateCheckpointError, 'no deserialization'):
                base.load_private_final_checkpoint(path)

    def test_transform_declaration_and_runtime_alignment(self):
        bad = copy.deepcopy(self.owner)
        bad['inference_preprocessing']['resize'] = [320, 320]
        with self.assertRaisesRegex(ValueError, '224-square'):
            self.bridge(owner=bad)
        with self.assertRaisesRegex(ValueError, 'runtime declarations'):
            self.bridge(run=replace(self.run_provenance, environment='different'))

    def test_no_unresolved_owner_release(self):
        with self.assertRaisesRegex(ValueError, 'unresolved'):
            self.bridge(owner={**self.owner, 'selection_release_receipt_identity': 'PENDING_OWNER'})

    def test_wrong_score_precision_declaration(self):
        with self.assertRaisesRegex(ValueError, 'FP64'):
            self.bridge(run=replace(self.run_provenance, score_precision='float32'))

    def test_reject_historical_denominators(self):
        historical = replace(self.manifest, samples=tuple(s for s in self.manifest.samples
                             if s.fold != 'fitting') + tuple(s for s in self.manifest.samples
                             if s.fold == 'fitting')[:4868])
        with self.assertRaisesRegex(ValueError, 'full-8000'):
            self.bridge(manifest=historical)

    def test_rejected_batch_does_not_advance_cursor(self):
        bridge = self.bridge()
        with self.assertRaisesRegex(ValueError, 'alignment'):
            self.add(bridge, rows=list(reversed(self.rows['seen'][:4])))
        self.add(bridge)  # identical correct rows still accepted from cursor zero

    def test_duplicate_filename_wrong_truth_and_fold(self):
        for key, value in [('sample_id', self.rows['seen'][1]['sample_id']), ('truth_id', 300), ('fold', 'unseen')]:
            rows = copy.deepcopy(self.rows['seen'][:4]); rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.add(self.bridge(), rows=rows)

    def test_candidate_order_and_pair_mapping_rejected(self):
        with self.assertRaises(ValueError):
            self.add(self.bridge(), ids=list(reversed(self.ids)))
        with self.assertRaisesRegex(ValueError, 'semantics'):
            self.add(self.bridge(), pairs=[self.pairs[1], self.pairs[0], *self.pairs[2:]])
        with self.assertRaisesRegex(ValueError, 'unique ascending'):
            self.add(self.bridge(), ids=[0, *self.ids[:-1]])

    def test_shapes_and_nonfinite_logits(self):
        for value in [torch.zeros(4, 54), torch.full((4, 55), float('nan')),
                      torch.full((4, 55), float('inf'))]:
            with self.subTest(shape=value.shape), self.assertRaises(ValueError):
                self.add(self.bridge(), outputs=dict(crop_logits=value, disease_logits=torch.zeros(4, 175)))

    def test_finite_extreme_logits_and_stable_global_ties(self):
        c = torch.linspace(-1e30, 1e30, 55).repeat(2, 1)
        d = torch.linspace(1e30, -1e30, 175).repeat(2, 1)
        scores = base.score_components(c, d, self.ids, self.pairs)
        self.assertTrue(torch.isfinite(scores).all()); self.assertEqual(scores.dtype, torch.float64)
        self.assertEqual(base.rank_global_ids(torch.zeros(1, 333, dtype=torch.float64), self.ids).tolist(), [[0, 1, 2, 3, 4]])

    def test_incomplete_coverage_no_directory_created(self):
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as temp:
            path = Path(temp) / 'never_created'
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                self.bridge().export(path)
            self.assertFalse(path.exists())


class ExportTests(BridgeFixture, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.temp = tempfile.TemporaryDirectory(prefix='SYNTHETIC_ONLY_E2E_', dir=ARTIFACTS)
        cls.sandbox = Path(cls.temp.name)
        cls.packet = cls.sandbox / 'SYNTHETIC_ONLY_PACKET'
        bridge = BaseHandoff(cls.manifest, cls.run_provenance, cls.owner)
        for fold, out in cls.outputs.items():
            for start in range(0, len(cls.rows[fold]), 257):
                bridge.add_batch(fold, {k: v[start:start+257] for k, v in out.items()},
                                 cls.rows[fold][start:start+257], cls.ids, cls.pairs)
        # Batch snapshots must survive caller buffer reuse.
        cls.saved_outputs = {f: {k: v.clone() for k, v in o.items()} for f, o in cls.outputs.items()}
        for out in cls.outputs.values():
            for tensor in out.values():
                tensor.zero_()
        cls.pin = bridge.export(cls.packet)
        cls.exported_bridge = bridge

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_complete_export_exact_scorer_and_python_rank_parity(self):
        for fold in ('seen', 'unseen'):
            out = self.saved_outputs[fold]
            pairs = torch.tensor(self.pairs)
            reference = (F.log_softmax(out['crop_logits'].double(), dim=1)[:, pairs[:, 0]] +
                         F.log_softmax(out['disease_logits'].double(), dim=1)[:, pairs[:, 1]]).numpy()
            with np.load(self.packet / (fold+'_scores.npz'), allow_pickle=False) as z:
                self.assertEqual(z['scores'].dtype, np.float64)
                np.testing.assert_array_equal(z['scores'], reference)
                ranks = np.array([sorted(self.ids, key=lambda j: (-float(row[j]), j))[:5] for row in reference])
                np.testing.assert_array_equal(z['top5_ids'], ranks)
                self.assertEqual(z['filenames'].tolist(), [r['sample_id'] for r in self.rows[fold]])
                self.assertEqual(z['top5_ids'][0].tolist(), [0, 1, 2, 3, 4])

    def test_receipt_companions_and_scope(self):
        receipt = json.loads((self.packet / 'COMPLETED.json').read_text())
        self.assertEqual(sha(self.packet / 'COMPLETED.json'), self.pin)
        self.assertEqual(set(receipt['artifacts']), {p.name for p in self.packet.iterdir()}-{'COMPLETED.json'})
        for name, pin in receipt['artifacts'].items():
            self.assertEqual(sha(self.packet/name), pin)
        self.assertFalse(receipt['independent_acceptance'])
        provenance = json.loads((self.packet/'provenance.json').read_text())
        self.assertTrue(provenance['synthetic']); self.assertFalse(provenance['actual_checkpoint_bytes_hashed'])

    def test_sealed_export_cannot_be_modified(self):
        with self.assertRaisesRegex(ValueError, 'already sealed'):
            self.exported_bridge.export(self.packet)
        with self.assertRaisesRegex(ValueError, 'already sealed'):
            self.add(self.exported_bridge)

    def test_existing_directory_refused_before_any_write(self):
        bridge = self.bridge()
        # Complete a fresh collector using the saved synthetic tensors.
        for fold in self.saved_outputs:
            bridge.add_batch(fold, self.saved_outputs[fold], self.rows[fold], self.ids, self.pairs)
        before = {p.name: sha(p) for p in self.packet.iterdir()}
        with self.assertRaisesRegex(FileExistsError, 'overwrite'):
            bridge.export(self.packet)
        self.assertEqual(before, {p.name: sha(p) for p in self.packet.iterdir()})

    @unittest.skipUnless(os.environ.get('WP2_AUDIT_ROOT'), 'Independent audit scripts are external to project')
    def test_export_through_preserved_WP5_intake_and_corruption_rejection(self):
        audit = Path(os.environ['WP2_AUDIT_ROOT'])
        scripts = self.sandbox/'audit_scripts'; scripts.mkdir()
        for name in ['accept_future_score_packet.py', 'accept_future_score_packet_reviewed.py']:
            shutil.copyfile(audit/'audit_scripts'/name, scripts/name)
            self.assertEqual(sha(scripts/name), sha(audit/'audit_scripts'/name))
        manifest_path = self.sandbox/'extracted_manifest_clip/identity_manifest.json'
        dump(manifest_path, self.manifest.to_dict())
        dump(self.sandbox/'Manifest_Frozen_CLIP_Verification_Receipt.json',
             dict(synthetic=True, manifest=dict(raw_sha256=sha(manifest_path), semantic_sha256=self.manifest.sha256)))
        spec = json.loads((audit/'audit_scripts/intake_templates/WP2_reviewed.json').read_text())
        provenance = json.loads((self.packet/'provenance.json').read_text())
        spec.update(packet_directory=self.packet.name, externally_pinned_receipt_sha256=self.pin,
                    reviewed_score_semantics='sum_log_softmax_v1',
                    reviewed_selection_and_preprocessing_reference='SYNTHETIC_ONLY_NOT_REAL_APPROVAL',
                    provenance_paths={k: k for k in spec['provenance_paths']})
        spec['reviewed_provenance_expectations'] = {k: dict(json_path=k, expected=provenance[k])
                                                   for k in spec['reviewed_provenance_expectations']}
        case_results = []
        for case, field in [('valid_export', None), ('wrong_candidate_order', 'candidate_ids'),
                            ('wrong_filename_order', 'filenames'), ('nonfinite_scores', 'scores'),
                            ('wrong_score_shape', 'shape'), ('wrong_native_precision', 'dtype'),
                            ('wrong_truth', 'truths'), ('wrong_saved_top5', 'top5_ids')]:
            case_dir = self.sandbox/('SYNTHETIC_ONLY_'+case)
            shutil.copytree(self.packet, case_dir)
            if field:
                with np.load(case_dir/'seen_scores.npz', allow_pickle=False) as z:
                    arrays = {k: z[k] for k in z.files}
                if field in ('candidate_ids', 'filenames'):
                    arrays[field] = arrays[field][::-1]
                elif field == 'scores': arrays['scores'][0, 0] = np.nan
                elif field == 'shape': arrays['scores'] = arrays['scores'][:, :-1]
                elif field == 'dtype': arrays['scores'] = arrays['scores'].astype(np.float32)
                elif field == 'truths': arrays['truths'][0] = 300
                else: arrays['top5_ids'][0] = arrays['top5_ids'][0][::-1]
                np.savez_compressed(case_dir/'seen_scores.npz', **arrays)
                receipt = json.loads((case_dir/'COMPLETED.json').read_text())
                receipt['artifacts']['seen_scores.npz'] = sha(case_dir/'seen_scores.npz')
                dump(case_dir/'COMPLETED.json', receipt)
            case_spec = copy.deepcopy(spec)
            case_spec.update(packet_directory=case_dir.name,
                             externally_pinned_receipt_sha256=sha(case_dir/'COMPLETED.json'),
                             output_receipt='results/SYNTHETIC_ONLY_'+case+'.json')
            spec_path = self.sandbox/('SYNTHETIC_ONLY_'+case+'_spec.json'); dump(spec_path, case_spec)
            proc = subprocess.run([sys.executable, '-B', str(scripts/'accept_future_score_packet_reviewed.py'),
                                   '--spec', str(spec_path)], capture_output=True, text=True, cwd=self.sandbox)
            case_results.append(dict(case=case, returncode=proc.returncode,
                                     stderr=proc.stderr.strip(), expected_acceptance=(field is None)))
            with self.subTest(case=case):
                self.assertEqual(proc.returncode == 0, field is None, proc.stderr)
                self.assertEqual((self.sandbox/case_spec['output_receipt']).exists(), field is None)
        dump(ARTIFACTS/'SYNTHETIC_ONLY_intake_cases.json', case_results)
        receipt = json.loads((self.sandbox/'results/SYNTHETIC_ONLY_valid_export.json').read_text())
        self.assertTrue(all(c['passed'] for c in receipt['checks']))
        self.assertFalse(receipt['actual_checkpoint_bytes_hashed'])
        # Do not preserve fake performance numbers as a model report.


if __name__ == '__main__':
    unittest.main(verbosity=2)
