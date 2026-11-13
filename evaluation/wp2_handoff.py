"""Named ConvNeXt-Base outputs -> WP5 records and sealed data-only handoff.

No inference, image IO, checkpoint deserialization, or implicit model selection.
The caller supplies an already approved EvaluationManifest and run declarations.
Export integrity is not independent acceptance or proof of checkpoint identity.
"""
from dataclasses import asdict
import csv
import hashlib
import json
import io
from pathlib import Path

import numpy as np

from evaluation.model_adapters import (
    IDENTITY_MANIFEST, ModelOutputAdapter, WP2_CONVNEXT,
)
from evaluation.reporting import (
    EvaluationManifest, RunProvenance, build_report, canonical_json, fingerprint,
    Candidate, candidate_digest, FINAL_NAMESPACE, FINAL_CANDIDATE_SHA256, FULL8000_FINAL_PROTOCOL,
)
from evaluation.wp2_base_compat import (
    ARCHITECTURE, EXPECTED_FINAL_CHECKPOINT_SHA256, adapt_named_outputs,
    validate_candidates, validate_output, score_components, rank_global_ids,
)
from evaluation.packet_io import json_bytes, sha256_bytes, write_new_directory

INFERENCE_PREPROCESSING = {
    'input_mode': 'RGB', 'resize': [224, 224], 'interpolation': 'bilinear',
    'antialias': True, 'tensor_dtype': 'float32', 'scale_uint8': True,
    'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
}
COMMON_MANIFEST_SHA256 = '41e4b1d499e1c45097fd8b74d917f06e0efc324772d9459e811259746f0f3a6c'
OWNER_FIELDS = (
    'architecture', 'epoch', 'inference_size', 'initialization_identity',
    'training_scope', 'training_count', 'training_source_sha256', 'ordered_fitting_names_sha256',
    'selection_rule', 'training_preprocessing', 'inference_preprocessing',
    'runtime', 'scorer_source_identity', 'training_recipe_identity',
    'selection_release_receipt_identity',
)


def _complete(value):
    if value is None or (isinstance(value, str) and
                        (not value.strip() or value.strip().upper().startswith(('PENDING', 'TODO')))):
        raise ValueError('Owner declarations must be explicit; unresolved release evidence')
    if isinstance(value, (dict, list)):
        if not value:
            raise ValueError('Empty owner declaration')
        for item in (value.values() if isinstance(value, dict) else value):
            _complete(item)


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')


def _csv(path, rows):
    with path.open('x', encoding='utf-8', newline='') as stream:
        csv.writer(stream, lineterminator='\n').writerows(rows)


def validate_owner_release(provenance, owner_metadata, training_source_sha256, fitting_sha256):
    """Shared declared-release checks; no checkpoint-byte or training proof."""
    if provenance.checkpoint_sha256 != EXPECTED_FINAL_CHECKPOINT_SHA256:
        raise ValueError('WP2 final checkpoint declaration differs; no fallback')
    if provenance.score_precision != 'float64' or provenance.score_semantics != 'sum_log_softmax_v1':
        raise ValueError('Require native FP64 sum_log_softmax_v1 scoring')
    # Freeze JSON values so later caller mutations cannot alter the release.
    owner = json.loads(canonical_json(owner_metadata))
    if not isinstance(owner, dict) or not set(OWNER_FIELDS) <= owner.keys():
        raise ValueError('Missing owner release declarations')
    _complete(owner)
    for key, expected in (('architecture', ARCHITECTURE), ('epoch', 1), ('inference_size', 224)):
        if type(owner[key]) is not type(expected) or owner[key] != expected:
            raise ValueError('Incompatible Base release declaration: '+key)
    for key, expected in (('training_scope', 'full_8000'), ('training_count', 8000),
                          ('training_source_sha256', training_source_sha256),
                          ('ordered_fitting_names_sha256', fitting_sha256)):
        if type(owner[key]) is not type(expected) or owner[key] != expected:
            raise ValueError('Owner training declaration differs from common contract: '+key)
    if owner['inference_preprocessing'] != INFERENCE_PREPROCESSING:
        raise ValueError('Require evidenced RGB 224-square Base preprocessing')
    if not isinstance(owner['training_preprocessing'], dict) or not isinstance(owner['runtime'], dict):
        raise ValueError('Structured training preprocessing and runtime declarations required')
    runtime = owner['runtime']
    if not {'python', 'torch', 'torchvision', 'device', 'image_precision', 'score_precision'} <= runtime.keys():
        raise ValueError('Incomplete runtime declaration')
    if runtime['score_precision'] != 'float64':
        raise ValueError('Runtime must declare native float64 scoring')
    if provenance.preprocessing != canonical_json(owner['inference_preprocessing']):
        raise ValueError('Run preprocessing and owner transform declarations differ')
    if provenance.environment != canonical_json(runtime):
        raise ValueError('Run and owner runtime declarations differ')
    return owner


class BaseHandoff:
    """WP5-INTERNAL truth-bearing collector; not an owner-channel exporter.

    Use BaseOwnerHandoff for label-free owner packets. Historical caller API is
    preserved; rejected batches never advance a fold cursor.
    """

    def __init__(self, manifest, provenance, owner_metadata, *, final_evaluation_reference=''):
        if type(manifest) is not EvaluationManifest or type(provenance) is not RunProvenance:
            raise ValueError('Approved EvaluationManifest and RunProvenance required')
        spec = manifest.spec
        if (spec.task_mode != 'final' or spec.ids != tuple(range(333)) or
                (manifest.expected_seen, manifest.expected_unseen) != (2000, 1448) or
                sum(s.fold == 'fitting' for s in manifest.samples) != 8000 or
                (len(manifest.seen_truth_ids), len(manifest.unseen_truth_ids)) != (323, 10)):
            raise ValueError('Require full-8000 / seen2000 / unseen1448 / shared333 contract')
        if not spec.synthetic and (manifest.sha256 != COMMON_MANIFEST_SHA256 or
                (spec.protocol_id, spec.protocol_version) !=
                ('cos30082-official-shared333', 'full8000-20261112-v1')):
            raise ValueError('Real handoff differs from independently pinned common manifest')
        owner = validate_owner_release(provenance, owner_metadata, spec.source_sha256, spec.split_sha256)
        self._manifest, self._provenance, self._owner = manifest, provenance, owner
        self._adapter = ModelOutputAdapter(manifest, provenance, WP2_CONVNEXT, final_evaluation_reference)
        self._rows = {fold: [asdict(s) for s in manifest.samples if s.fold == fold]
                      for fold in ('seen', 'unseen')}
        self._ids = spec.ids
        self._pairs = tuple((c.crop_id, c.disease_id) for c in spec.candidates)
        self._batches = {'seen': [], 'unseen': []}
        self._counts = {'seen': 0, 'unseen': 0}
        self._written = False

    def add_batch(self, fold, outputs, records, candidate_ids, candidate_pairs):
        """Accept explicit original identities, never attach inferred filenames.

        Candidate arguments describe the caller's actual output mapping, and
        must equal the manifest. Only independent-head logits are accepted.
        """
        if self._written:
            raise ValueError('Handoff already sealed')
        if fold not in self._batches:
            raise ValueError('Only separate seen/unseen batches are accepted')
        records = list(records)
        start = self._counts[fold]
        expected = self._rows[fold][start:start+len(records)]
        named = adapt_named_outputs(outputs, records, expected, candidate_ids, candidate_pairs,
                                    expected_candidate_ids=self._ids, expected_pairs=self._pairs)
        packet = self._packet(fold, named)
        validated = self._adapter.adapt(packet)
        self._batches[fold].append((named, validated))
        self._counts[fold] += len(validated)

    def _packet(self, fold, named):
        run, manifest = self._provenance, self._manifest
        return dict(schema='wp5-model-output-v1', model_kind=WP2_CONVNEXT,
                    **{k: getattr(run, k) for k in ('model_id', 'run_id', 'checkpoint_id',
                       'checkpoint_sha256', 'preprocessing', 'code_version', 'environment',
                       'score_precision', 'score_semantics')},
                    manifest_sha256=manifest.sha256, source_sha256=manifest.spec.source_sha256,
                    candidate_hash_namespace=manifest.spec.candidate_hash_namespace,
                    candidate_sha256=manifest.spec.candidate_sha256,
                    split_hash_namespace=IDENTITY_MANIFEST, split_sha256=manifest.sha256,
                    selection=dict(architecture=ARCHITECTURE, epoch=1, input_resolution=224),
                    fold=fold, image_names=[r['sample_id'] for r in named['records']],
                    ground_truth_ids=[r['truth_id'] for r in named['records']],
                    candidate_ids=self._ids, scores=named['scores'])

    def export(self, directory):
        """Write a new directory; completion receipt is always written last.

        Producer hashes require an independently communicated receipt pin and
        separate WP5 intake review. They are not an authenticity guarantee.
        """
        if self._written:
            raise ValueError('Handoff already sealed')
        if self._counts != {'seen': 2000, 'unseen': 1448}:
            raise ValueError('Incomplete fold coverage; no export written')
        directory = Path(directory)
        if directory.exists():
            raise FileExistsError('Refuse to overwrite an existing handoff directory')
        records = tuple(r for fold in ('seen', 'unseen')
                        for _, batch in self._batches[fold] for r in batch)
        report = build_report(self._manifest, records)  # existing complete-record validator
        metrics = {}
        for fold, summary in [('seen', report['seen']), ('unseen', report['unseen']),
                              ('overall', report['combined_micro'])]:
            metrics[fold] = dict(count=summary['count'], hits_top1=summary['correct_top1'],
                                 hits_top5=summary['correct_top5'], top1=summary['top1'], top5=summary['top5'])
        for k in (1, 5):
            metrics['harmonic_mean_top'+str(k)] = report['harmonic_mean_top'+str(k)]
        manifest, spec, run = self._manifest, self._manifest.spec, self._provenance
        provenance = dict(self._owner)
        provenance.update(asdict(run))
        provenance.update(synthetic=spec.synthetic,
                          candidate_sha256=spec.candidate_sha256, protocol_id=spec.protocol_id,
                          protocol_version=spec.protocol_version, tie_rule=spec.tie_rule,
                          prompt_sha256=fingerprint(json.loads(spec.final_prompt_json)),
                          manifest_sha256=manifest.sha256, actual_checkpoint_bytes_hashed=False,
                          private_checkpoint_loading_verified=False,
                          artifact_scope=('SYNTHETIC_ONLY_INTERFACE_TEST_NOT_MODEL_RESULTS' if spec.synthetic
                                          else 'OWNER_DECLARATIONS_REQUIRING_INDEPENDENT_ACCEPTANCE'))
        directory.mkdir(parents=True, exist_ok=False)
        for fold in ('seen', 'unseen'):
            named = [n for n, _ in self._batches[fold]]
            scores = np.concatenate([n['scores'].numpy() for n in named])
            ranks = np.concatenate([n['top5_ids'].numpy() for n in named])
            rows = self._rows[fold]
            np.savez_compressed(directory / (fold+'_scores.npz'), scores=scores,
                                filenames=np.array([r['sample_id'] for r in rows]),
                                truths=np.array([r['truth_id'] for r in rows], dtype=np.int64),
                                candidate_ids=np.array(self._ids, dtype=np.int64), top5_ids=ranks)
            _csv(directory / (fold+'_predictions.csv'),
                 [['image_name', 'truth_global_id', *['rank'+str(k) for k in range(1, 6)]]]+
                 [[r['sample_id'], r['truth_id'], *rank] for r, rank in zip(rows, ranks)])
            confusion = report[fold]['confusion_matrix']
            _csv(directory / (fold+'_confusion.csv'),
                 [['truth_global_id', *confusion['prediction_ids']]]+
                 [[t, *row] for t, row in zip(confusion['truth_ids'], confusion['counts'])])
        for name, data in [('metrics.json', metrics), ('provenance.json', provenance)]:
            (directory / name).write_bytes(_json_bytes(data))
        receipt = dict(schema='wp2-base-handoff-v1', synthetic=spec.synthetic,
                       manifest_sha256=manifest.sha256, independent_acceptance=False,
                       actual_checkpoint_bytes_hashed=False,
                       artifacts={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(directory.iterdir())})
        receipt_bytes = _json_bytes(receipt)
        (directory / 'COMPLETED.json').write_bytes(receipt_bytes)
        self._written = True
        return hashlib.sha256(receipt_bytes).hexdigest()


class BaseOwnerHandoff:
    """Label-free owner collector: receives no manifest samples or test truths.

    Caller supplies already collected dual-head outputs. No model construction,
    checkpoint deserialization, downloads, image IO or inference occurs here.
    """

    def __init__(self, order, provenance, owner_metadata, *, expected_order_sha256, allow_synthetic=False):
        order = json.loads(canonical_json(order))
        fields = {'schema', 'manifest_sha256', 'source_sha256', 'candidate_hash_namespace', 'candidate_sha256',
                  'candidate_ids', 'split_hash_namespace', 'split_sha256', 'clip_prompt_sha256', 'synthetic',
                  'folds', 'protocol_id', 'protocol_version', 'candidate_rows', 'training'}
        if (not isinstance(order, dict) or set(order) != fields or
                order['schema'] != 'wp5-full8000-inference-order-v1' or
                (order['protocol_id'], order['protocol_version']) != FULL8000_FINAL_PROTOCOL):
            raise ValueError('Explicit label-free full-8000 inference order required')
        if type(allow_synthetic) is not bool or type(order['synthetic']) is not bool:
            raise ValueError('Explicit synthetic flag required')
        if order['synthetic'] and not allow_synthetic:
            raise ValueError('synthetic fixtures require explicit opt-in')
        if fingerprint(order) != expected_order_sha256:
            raise ValueError('Authenticated inference-order fingerprint mismatch')
        if not order['synthetic'] and order['manifest_sha256'] != COMMON_MANIFEST_SHA256:
            raise ValueError('Real inference order must use the pinned common manifest')
        rows = tuple(Candidate(**row) for row in order['candidate_rows'])
        self._ids, self._pairs = validate_candidates(order['candidate_ids'], [(r.crop_id, r.disease_id) for r in rows])
        if (self._ids != tuple(range(333)) or tuple(c.composition_id for c in rows) != self._ids or
                order['candidate_hash_namespace'] != FINAL_NAMESPACE or
                order['candidate_sha256'] != candidate_digest(rows, FINAL_NAMESPACE)):
            raise ValueError('Candidate semantic fingerprint or original ordering differs')
        if not order['synthetic'] and order['candidate_sha256'] != FINAL_CANDIDATE_SHA256:
            raise ValueError('Real candidate identity differs from shared333')
        if (order['split_hash_namespace'] != IDENTITY_MANIFEST or order['split_sha256'] != order['manifest_sha256']
                or set(order['folds']) != {'seen', 'unseen'}):
            raise ValueError('Inference-order identity namespace mismatch')
        all_names = []
        for fold, count in [('seen', 2000), ('unseen', 1448)]:
            names = order['folds'][fold]
            if (not isinstance(names, list) or len(names) != count or
                    any(not isinstance(n, str) or not n.strip() for n in names)):
                raise ValueError('Incorrect inference-order filename denominator')
            all_names.extend(names)
        if len(set(all_names)) != 3448:
            raise ValueError('Duplicate inference-order filenames across folds')
        if type(provenance) is not RunProvenance or not isinstance(owner_metadata, dict) or set(owner_metadata) != set(OWNER_FIELDS):
            raise ValueError('Exact run and owner release fields required; no label attachments')
        training = order['training']
        if set(training) != {'training_scope','training_count','training_source_sha256','ordered_fitting_names_sha256'}:
            raise ValueError('Explicit full-data training identity required')
        owner = validate_owner_release(provenance, owner_metadata, order['source_sha256'], training['ordered_fitting_names_sha256'])
        if canonical_json({k:owner[k] for k in training}) != canonical_json(training):
            raise ValueError('Owner training identity differs from order')
        forbidden = {'truths', 'truth_id', 'ground_truth_ids', 'labels', 'targets', 'samples', 'seen_truth_ids', 'unseen_truth_ids'}
        def no_labels(value):
            if isinstance(value, dict):
                if forbidden & value.keys():
                    raise ValueError('Owner release metadata must not attach evaluation labels')
                for item in value.values(): no_labels(item)
            elif isinstance(value, list):
                for item in value: no_labels(item)
        no_labels(owner)
        self._order, self._run, self._owner = order, provenance, owner
        self._batches = {'seen': [], 'unseen': []}
        self._counts = {'seen': 0, 'unseen': 0}
        self._sealed = False

    def add_batch(self, fold, outputs, image_names, candidate_ids, candidate_pairs):
        if self._sealed:
            raise ValueError('Owner handoff already sealed')
        if fold not in self._batches:
            raise ValueError('Separate seen/unseen fold required')
        names = list(image_names)
        start = self._counts[fold]
        if not names or names != self._order['folds'][fold][start:start+len(names)]:
            raise ValueError('Owner filename/fold order mismatch or duplicate batch')
        if validate_candidates(candidate_ids, candidate_pairs) != (self._ids, self._pairs):
            raise ValueError('Owner candidate mapping/order mismatch')
        validate_output(outputs, expected_batch=len(names))
        scores = score_components(outputs['crop_logits'], outputs['disease_logits'], self._ids, self._pairs).detach().to('cpu', copy=True)
        ranks = rank_global_ids(scores, self._ids)
        self._batches[fold].append((scores.numpy(), ranks.numpy()))
        self._counts[fold] += len(names)

    def export(self, directory):
        if self._sealed:
            raise ValueError('Owner handoff already sealed')
        if self._counts != {'seen': 2000, 'unseen': 1448}:
            raise ValueError('Incomplete owner fold coverage; no export written')
        files = {}
        for fold in ('seen', 'unseen'):
            arrays = self._batches[fold]
            stream = io.BytesIO()
            np.savez_compressed(stream, scores=np.concatenate([x[0] for x in arrays]),
                                filenames=np.asarray(self._order['folds'][fold]),
                                candidate_ids=np.asarray(self._ids, dtype=np.int64),
                                top5_ids=np.concatenate([x[1] for x in arrays]))
            files[fold+'_scores.npz'] = stream.getvalue()
            packet = dict(schema='wp5-model-output-npz-v1', model_kind=WP2_CONVNEXT,
                          **{k:getattr(self._run,k) for k in ('model_id','run_id','checkpoint_id','checkpoint_sha256',
                              'preprocessing','code_version','environment','score_precision','score_semantics')},
                          **{k:self._order[k] for k in ('manifest_sha256','source_sha256','candidate_hash_namespace',
                              'candidate_sha256','candidate_ids','split_hash_namespace','split_sha256')},
                          selection=dict(architecture=ARCHITECTURE, epoch=1, input_resolution=224),
                          fold=fold, image_names=self._order['folds'][fold], scores_file=fold+'_scores.npz')
            files[fold+'_packet.json'] = json_bytes(packet)
        files['provenance.json'] = json_bytes(dict(run=asdict(self._run), owner=self._owner))
        receipt = dict(schema='wp2-label-free-handoff-v1', synthetic=self._order['synthetic'],
                       inference_order_sha256=fingerprint(self._order), independent_acceptance=False,
                       checkpoint_bytes_verified=False,
                       artifacts={name:sha256_bytes(raw) for name,raw in sorted(files.items())})
        files['COMPLETED.json'] = json_bytes(receipt)
        write_new_directory(directory, files, completion_name='COMPLETED.json')
        self._sealed = True
        return sha256_bytes(files['COMPLETED.json'])
