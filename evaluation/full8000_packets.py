"""Authenticated WP2-only full-8000 intake using existing WP5 validators.

Partial four-model intake is deliberate: no four-model acceptance or frozen
CLIP reuse decision is minted here. Archive and NPZ contents are data only.
"""
from dataclasses import asdict
import io
import math
import platform
from pathlib import Path
import zipfile

import numpy as np

from evaluation.final_manifest import load_full8000_manifest, full8000_inference_order
from evaluation.model_adapters import ModelOutputAdapter, WP2_CONVNEXT
from evaluation.packet_io import (child, decode_json, inspect_zip, json_bytes, packet_directory,
                                  read_json, sha256_bytes, write_new_directory, MAX_PACKET_BYTES, MAX_FILE_BYTES)
from evaluation.reporting import (FULL8000_FINAL_PROTOCOL, RunProvenance, build_report,
                                  canonical_json, fingerprint)
from evaluation.wp2_handoff import BaseOwnerHandoff

PACKET_FILES = frozenset({'seen_packet.json', 'unseen_packet.json', 'seen_scores.npz',
                          'unseen_scores.npz', 'provenance.json', 'COMPLETED.json'})
REFERENCES = ('final_policy_reference', 'evaluation_authorization_reference', 'owner_approval_reference')


def _verified_owner_files(directory, receipt_pin):
    actual = list(directory.iterdir())
    if ({p.name for p in actual} != PACKET_FILES or
            any(p.is_symlink() or not p.is_file() for p in actual)):
        raise ValueError('Owner packet files/layout differ from label-free contract')
    receipt = read_json(directory/'COMPLETED.json', receipt_pin)
    if (not isinstance(receipt, dict) or receipt.get('schema') != 'wp2-label-free-handoff-v1' or
            not isinstance(receipt.get('artifacts'), dict) or
            set(receipt['artifacts']) != PACKET_FILES-{'COMPLETED.json'}):
        raise ValueError('Incomplete owner completion ledger')
    snapshots, total = {}, 0
    for name, pin in receipt['artifacts'].items():
        path = directory/name
        total += path.stat().st_size
        if total > MAX_PACKET_BYTES:
            raise ValueError('Oversized owner packet')
        raw = path.read_bytes()
        if sha256_bytes(raw) != pin:
            raise ValueError('Owner companion raw hash mismatch')
        snapshots[name] = raw
    return receipt, snapshots


def _scores(raw, fold, packet, manifest):
    # Bound nested NPZ members before NumPy allocation. No pickle or extraction.
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = inspect_zip(archive)
            if {m.filename for m in members} != {'scores.npy', 'filenames.npy', 'candidate_ids.npy', 'top5_ids.npy'}:
                raise ValueError('Unexpected label-bearing or extra NPZ fields')
            for member in members:
                # A small ZIP can contain an NPY header declaring an enormous
                # allocation. Validate header/data lengths before np.load.
                with archive.open(member) as stream:
                    version = np.lib.format.read_magic(stream)
                    reader = {(1,0):np.lib.format.read_array_header_1_0,
                              (2,0):np.lib.format.read_array_header_2_0}.get(version)
                    if reader is None:
                        raise ValueError('Unsupported score array header version')
                    shape, _, dtype = reader(stream, max_header_size=10000)
                    size = math.prod(shape)*dtype.itemsize
                    if dtype.hasobject or size > MAX_FILE_BYTES or size != member.file_size-stream.tell():
                        raise ValueError('Unsafe array dtype, allocation or data length')
        with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
            scores, names, ids, ranks = (arrays[k] for k in ('scores', 'filenames', 'candidate_ids', 'top5_ids'))
    except (OSError, zipfile.BadZipFile, KeyError) as error:
        raise ValueError('Invalid native score archive') from error
    count = getattr(manifest, 'expected_'+fold)
    expected_names = [s.sample_id for s in manifest.samples if s.fold == fold]
    if (scores.dtype != np.dtype('float64') or scores.shape != (count, 333) or not np.isfinite(scores).all()):
        raise ValueError('Require native finite FP64 score matrix with exact fold denominator and 333 columns')
    if (names.dtype.kind != 'U' or names.shape != (count,) or names.tolist() != expected_names or
            packet.get('image_names') != expected_names):
        raise ValueError('Exact filename/fold ordering mismatch')
    if ids.dtype.kind not in 'iu' or ids.shape != (333,) or tuple(ids) != manifest.spec.ids:
        raise ValueError('Original 333 candidate column ordering mismatch')
    if ranks.dtype.kind not in 'iu' or ranks.shape != (count, 5):
        raise ValueError('Saved ordered Top-5 shape/type mismatch')
    return scores, ranks


def import_full8000_wp2(handoff_path, handoff_sha256, output_path, *, registry_path,
                        allow_synthetic=False, checkpoint_file=None):
    """Validate one separately authorized packet, attach truths only within WP5.

    Registry claims reserve one WP2 role per manifest, across output paths.
    Failed writes leave an incomplete claim/output for explicit review; no
    automatic overwrite, recovery, model reselection or retry is performed.
    """
    # Reuse the existing PR #24 owner-field boundary; no second truth join.
    from scripts.import_wp5_final import PACKET_FIELDS, _attach_truths

    output, registry = Path(output_path), Path(registry_path)
    if (output.resolve().is_relative_to(registry.resolve()) or
            registry.resolve().is_relative_to(output.resolve())):
        raise ValueError('WP5 output and intake registry must be disjoint directories')
    if output.exists() or output.is_symlink():
        raise FileExistsError('Preserving existing WP5 output')
    if registry.is_symlink() or (registry.exists() and not registry.is_dir()):
        raise ValueError('Registry must be an ordinary directory')
    handoff = Path(handoff_path)
    bundle = read_json(handoff, handoff_sha256)
    fields = {'schema', 'protocol_id', 'protocol_version', 'identity_manifest', 'manifest_sha256',
              'inference_order_sha256', 'owner_packet', 'provenance', 'reviewed_owner_metadata', *REFERENCES}
    if (not isinstance(bundle, dict) or set(bundle) != fields or
            bundle.get('schema') != 'wp5-full8000-wp2-import-v1' or
            (bundle.get('protocol_id'), bundle.get('protocol_version')) != FULL8000_FINAL_PROTOCOL):
        raise ValueError('Explicit versioned full-8000 WP2 handoff required')
    for key in REFERENCES:
        value = bundle[key]
        if not isinstance(value, str) or not value.strip() or value.strip().upper().startswith(('PENDING', 'TODO')):
            raise ValueError('Explicit reviewed reference required: '+key)
    entry = bundle['identity_manifest']
    if not isinstance(entry, dict) or set(entry) != {'path', 'file_sha256'}:
        raise ValueError('Manifest requires path and raw file hash')
    manifest_path = child(handoff.parent, entry['path'])
    manifest = load_full8000_manifest(read_json(manifest_path, entry['file_sha256']),
                                     expected_semantic_sha256=bundle['manifest_sha256'], allow_synthetic=allow_synthetic)
    order = full8000_inference_order(manifest)
    if fingerprint(order) != bundle['inference_order_sha256']:
        raise ValueError('Owner inference-order semantic fingerprint mismatch')
    run = RunProvenance(**bundle['provenance'])
    # Selected full-data identity and release declarations checked BEFORE packet
    # interpretation, including synthetic fixtures. No historical pin exception.
    BaseOwnerHandoff(order, run, bundle['reviewed_owner_metadata'],
                     expected_order_sha256=bundle['inference_order_sha256'], allow_synthetic=allow_synthetic)
    claim = registry/(WP2_CONVNEXT+'_'+manifest.sha256)
    if claim.exists() or claim.is_symlink():
        raise FileExistsError('WP2 role already imported or has an incomplete registry claim')
    checkpoint_verified = False
    if checkpoint_file is not None:
        path = Path(checkpoint_file)
        if path.is_symlink() or not path.is_file():
            raise ValueError('Checkpoint-byte verification requires an explicit regular file')
        import hashlib
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''): digest.update(chunk)
        if digest.hexdigest() != run.checkpoint_sha256:
            raise ValueError('Checkpoint bytes differ from selected full-data pin')
        checkpoint_verified = True
    adapter = ModelOutputAdapter(manifest, run, WP2_CONVNEXT, bundle['evaluation_authorization_reference'])
    truths = {s.sample_id:s.truth_id for s in manifest.samples if s.fold in ('seen', 'unseen')}
    records, validated_packets = [], {}
    with packet_directory(handoff.parent, bundle['owner_packet']) as directory:
        if output.resolve().is_relative_to(directory.resolve()) or registry.resolve().is_relative_to(directory.resolve()):
            raise ValueError('Intake must not write inside immutable owner packet')
        receipt, snapshots = _verified_owner_files(directory, bundle['owner_packet']['receipt_sha256'])
        if (receipt.get('synthetic') is not manifest.spec.synthetic or
                receipt.get('inference_order_sha256') != bundle['inference_order_sha256']):
            raise ValueError('Owner receipt contract identity mismatch')
        provenance = decode_json(snapshots['provenance.json'])
        if canonical_json(provenance) != canonical_json(dict(run=asdict(run), owner=bundle['reviewed_owner_metadata'])):
            raise ValueError('Owner provenance differs from independently reviewed handoff')
        for fold in ('seen', 'unseen'):
            packet = decode_json(snapshots[fold+'_packet.json'])
            expected_fields = (PACKET_FIELDS-{'scores','prompt_sha256'}) | {'scores_file'}
            if (not isinstance(packet, dict) or set(packet) != expected_fields or
                    packet.get('schema') != 'wp5-model-output-npz-v1' or packet.get('scores_file') != fold+'_scores.npz'
                    or packet.get('fold') != fold):
                raise ValueError('Exact label-free owner packet fields/fold required')
            scores, saved_ranks = _scores(snapshots[fold+'_scores.npz'], fold, packet, manifest)
            packet = {k:v for k,v in packet.items() if k != 'scores_file'}
            packet.update(schema='wp5-model-output-v1', scores=scores)
            packet, _ = _attach_truths(packet, truths)
            batch = adapter.adapt(packet)
            if not np.array_equal(saved_ranks, np.asarray([r.ranking()[:5] for r in batch])):
                raise ValueError('Saved ordered Top-5 differs from stable canonical rankings')
            records.extend(batch)
            # WP5-INTERNAL canonical payloads for later existing four-role
            # ingestion. They never enter the owner packet directory.
            validated_packets[fold] = {**packet, 'scores':scores.tolist()}
    report = build_report(manifest, records)
    summary = {fold:{k:report[fold][k] for k in ('count','correct_top1','correct_top5','top1','top5')}
               for fold in ('seen','unseen')}
    summary['overall'] = {k:report['combined_micro'][k] for k in ('count','correct_top1','correct_top5','top1','top5')}
    summary.update({'H@1':report['harmonic_mean_top1'], 'H@5':report['harmonic_mean_top5'],
                    'synthetic':manifest.spec.synthetic, 'accuracy_scale':'fraction_0_to_1'})
    result = dict(schema='wp5-full8000-wp2-receipt-v1', status='SAVED_SCORE_VALIDATED_NOT_MODEL_REPRODUCTION',
                  synthetic=manifest.spec.synthetic, model_kind=WP2_CONVNEXT,
                  manifest_sha256=manifest.sha256, candidate_sha256=manifest.spec.candidate_sha256,
                  manifest_raw_sha256=entry['file_sha256'], inference_order_sha256=fingerprint(order),
                  handoff_sha256=handoff_sha256, owner_receipt_sha256=bundle['owner_packet']['receipt_sha256'],
                  checkpoint_declared_sha256=run.checkpoint_sha256, checkpoint_bytes_verified=checkpoint_verified,
                  checkpoint_identity_scope='declared identity; byte verification separately recorded; no inference reproduction',
                  label_free_owner_packet=True, label_handling='truths attached only by WP5; not label-blind evaluation',
                  summary_sha256=fingerprint(summary), four_model_acceptance=False,
                  frozen_clip_reuse_approval=False, references={k:bundle[k] for k in REFERENCES})
    source_root = Path(__file__).resolve().parents[1]
    source_paths = ('evaluation/full8000_packets.py', 'evaluation/packet_io.py', 'evaluation/final_manifest.py',
                    'evaluation/reporting.py', 'evaluation/model_adapters.py', 'evaluation/metrics.py',
                    'evaluation/wp2_base_compat.py', 'evaluation/wp2_handoff.py', 'scripts/import_wp5_final.py')
    result['verification_code_sha256'] = {name:sha256_bytes((source_root/name).read_bytes()) for name in source_paths}
    result['verification_environment'] = {'python':platform.python_version(), 'numpy':np.__version__}
    files = {'report.json':json_bytes(report), 'summary.json':json_bytes(summary),
             **{fold+'_validated_packet.json':json_bytes(packet) for fold,packet in validated_packets.items()}}
    result['outputs'] = {name:sha256_bytes(raw) for name,raw in sorted(files.items())}
    files['receipt.json'] = json_bytes(result)
    registry.mkdir(parents=True, exist_ok=True)
    try:
        claim.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError('WP2 role already imported or has an incomplete registry claim') from error
    # Exclusive role claim persists if output is interrupted; no successful
    # receipt is written until every WP5 output has been flushed.
    write_new_directory(output, files, completion_name='receipt.json')
    write_new_directory(claim/'completed', {'receipt.json':json_bytes(result)}, completion_name='receipt.json')
    return result
