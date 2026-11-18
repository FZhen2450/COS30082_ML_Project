"""Replay a frozen WP1 owner export from saved FP32 scores, without inference.

This verifies the owner notebook format, not PR #24's label-free packet format.
It does not establish which model produced the scores or replace peer replay.
"""
import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ROLE = 'wp1_independent_pair'
PINS = {
    'wp1_crop': 'f0c82ca78153f492078a3a39a11eac1a847cc85eb496cf3b3d4d74db752c7d14',
    'wp1_disease': 'f497e19da62b68dfbcfbae58b9a5318a26b1989ba59b7a486223cad0d56e31a5',
}
FIELDS = {'scores', 'filenames', 'truths', 'candidate_ids', 'top5_ids',
          'independent_crop_ids', 'independent_disease_ids'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_packet(packet, samples, candidate_ids):
    """Reject malformed rows before recomputing deterministic global-ID ranks."""
    require(set(packet) == FIELDS, 'Unexpected or missing score-packet fields')
    n, width = len(samples), len(candidate_ids)
    scores = packet['scores']
    require(scores.dtype == np.dtype('float32') and scores.shape == (n, width)
            and np.isfinite(scores).all(), 'Require full finite native float32 scores')
    require(packet['candidate_ids'].dtype == np.dtype('int64')
            and np.array_equal(packet['candidate_ids'], candidate_ids), 'Candidate columns changed')
    require(packet['filenames'].dtype.kind == 'U'
            and packet['filenames'].tolist() == [s.sample_id for s in samples], 'Filename order changed')
    require(packet['truths'].dtype == np.dtype('int64')
            and packet['truths'].tolist() == [s.truth_id for s in samples], 'Truth order changed')
    for field, limit in [('independent_crop_ids', 55), ('independent_disease_ids', 175)]:
        values = packet[field]
        require(values.dtype == np.dtype('int64') and values.shape == (n,)
                and np.all((values >= 0) & (values < limit)), 'Invalid independent component IDs')
    ranks = np.asarray(candidate_ids)[np.argsort(-scores, axis=1, kind='stable')[:, :min(5, width)]]
    require(packet['top5_ids'].dtype == np.dtype('int64')
            and np.array_equal(packet['top5_ids'], ranks), 'Saved Top-5 disagrees with canonical scores')
    truth = packet['truths']
    hits1 = int(np.count_nonzero(ranks[:, 0] == truth))
    hits5 = int(np.count_nonzero(np.any(ranks == truth[:, None], axis=1)))
    return {'count': n, 'hits_top1': hits1, 'hits_top5': hits5,
            'top1': hits1 / n, 'top5': hits5 / n}, ranks


def verify(run):
    from evaluation.reporting import (EvaluationManifest, RunProvenance,
                                     build_report, export_report, records_from_scores)
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    plan, completed = read(run / 'PLAN.json'), read(run / 'COMPLETED.json')
    require(plan['requested_models'] == [ROLE], 'Require the separate WP1 owner run')
    require(completed['training_run'] is False and completed['checkpoint_selection_changed'] is False,
            'Frozen inference declarations changed')
    require(completed['images_processed_per_model'] == 3448, 'Incomplete official run')
    for key, pin in PINS.items():
        require(plan['assets'][key]['sha256'] == pin, 'Unexpected checkpoint declaration')
    manifest = EvaluationManifest.from_dict(read(run / 'identity_manifest.json'))
    require(manifest.sha256 == '36cdf1ecdbec0522f3ce83250dc21eacc499f8f61db25955eed7a3d578e77606',
            'Frozen cross-model official task changed')
    require(manifest.sha256 == completed['manifest_sha256'], 'Completion manifest mismatch')
    require(tuple(manifest.spec.ids) == tuple(range(333)), 'Require original 333 candidate IDs')
    model = run / ROLE
    provenance = RunProvenance(**read(model / 'provenance.json'))
    records, metrics = [], {}
    for fold, expected in [('seen', 2000), ('unseen', 1448)]:
        samples = tuple(s for s in manifest.samples if s.fold == fold)
        require(len(samples) == expected, 'Official denominator changed')
        with np.load(model / (fold + '_scores.npz'), allow_pickle=False) as archive:
            packet = {k: archive[k] for k in archive.files}
        metrics[fold], ranks = validate_packet(packet, samples, manifest.spec.ids)
        with (model / (fold + '_predictions.csv')).open(newline='', encoding='utf-8') as stream:
            rows = list(csv.reader(stream))
        require(rows[0] == ['image_name', 'truth_global_id', 'rank1', 'rank2', 'rank3', 'rank4', 'rank5'],
                'Prediction CSV header changed')
        expected_rows = [[s.sample_id, str(s.truth_id), *map(str, row)] for s, row in zip(samples, ranks)]
        require(rows[1:] == expected_rows, 'Prediction CSV differs from saved scores')
        batch = records_from_scores(manifest.spec, provenance, samples, packet['scores'], manifest.spec.ids)
        records.extend(replace(r, independent_crop_id=int(c), independent_disease_id=int(d))
                       for r, c, d in zip(batch, packet['independent_crop_ids'], packet['independent_disease_ids']))
    for k in ('1', '5'):
        s, u = metrics['seen']['top' + k], metrics['unseen']['top' + k]
        metrics['H@' + k] = 2 * s * u / (s + u) if s + u else 0.0
    require(metrics == read(model / 'metrics.json'), 'Saved metrics disagree with NumPy replay')
    report = build_report(manifest, records)
    require(json.loads(json.dumps(report, allow_nan=False)) == read(model / 'wp5_report/report.json'),
            'Full WP5 report replay differs')
    csv_tables = 0
    with tempfile.TemporaryDirectory(prefix='wp1-official-replay-') as temporary:
        fresh = export_report(report, Path(temporary) / 'report')
        for table in fresh.glob('*.csv'):
            with table.open(newline='', encoding='utf-8') as a, (model / 'wp5_report' / table.name).open(
                    newline='', encoding='utf-8') as b:
                require(list(csv.reader(a)) == list(csv.reader(b)), 'Saved report table differs: ' + table.name)
            csv_tables += 1
    for fold in ('seen', 'unseen'):
        matrix = report[fold]['confusion_matrix']
        require(sum(map(sum, matrix['counts'])) == metrics[fold]['count'], 'Confusion denominator differs')
    return {'schema': 'wp1-official-saved-score-replay-v1', 'passed': True,
            'metrics': metrics, 'manifest_sha256': manifest.sha256,
            'all_3448_saved_top5_rows_match': True, 'complete_wp5_report_matches': True,
            'reproduced_csv_tables': csv_tables,
            'inference_performed': False, 'not_a_second_human_model_replay': True,
            'files_sha256': {str(p.relative_to(run)).replace('\\', '/'): sha(p)
                             for p in sorted(run.rglob('*')) if p.is_file()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--receipt', required=True, type=Path)
    args = parser.parse_args()
    require(not args.receipt.exists(), 'Preserve existing verification receipts')
    result = verify(args.run.resolve(strict=True))
    with args.receipt.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'files_sha256'}, indent=2))


if __name__ == '__main__':
    main()
