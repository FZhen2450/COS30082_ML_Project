"""Evaluate one frozen full-training WP1 pair on the complete shared-333 tests."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch

from data_utils import load_metadata
from evaluation.metrics import harmonic_mean
from final_protocol import file_sha256, load_final_protocol, write_new_json
from wp1_final import FinalIndependentPredictor


def summarize_folds(seen, unseen):
    for fold, count in ((seen, 2000), (unseen, 1448)):
        if fold['count'] != count or not 0 <= fold['hits_top1'] <= fold['hits_top5'] <= count:
            raise ValueError('Require complete official folds and valid hit counts')
    overall = {'count': 3448, **{key: seen[key]+unseen[key] for key in ('hits_top1', 'hits_top5')}}
    for fold in (seen, unseen, overall):
        fold.update(top1=fold['hits_top1']/fold['count'], top5=fold['hits_top5']/fold['count'])
    return {'overall': overall, 'seen': seen, 'unseen': unseen,
            'harmonic_mean_top1': harmonic_mean(seen['top1'], unseen['top1']),
            'harmonic_mean_top5': harmonic_mean(seen['top5'], unseen['top5'])}


def validate_official_frames(protocol, train, seen, unseen):
    by_id = {row['composition_id']:row for row in protocol['candidates']}
    training_names = set(train.image_name)
    if len(seen) != 2000 or len(unseen) != 1448:
        raise ValueError('Official denominator mismatch')
    sets = [set(frame.image_name) for frame in (seen, unseen)]
    if any(len(names) != len(frame) for names, frame in zip(sets, (seen, unseen))):
        raise ValueError('Duplicate official identities')
    if sets[0] & sets[1] or any(names & training_names for names in sets):
        raise ValueError('Official identities overlap each other or training')
    train_ids = set(train.composition_id)
    if not set(seen.composition_id) <= train_ids or set(unseen.composition_id) & train_ids:
        raise ValueError('Official seen/unseen composition memberships disagree with full training')
    for frame in (seen, unseen):
        for row in frame.to_dict('records'):
            candidate = by_id.get(row['composition_id'])
            if not candidate or any(row[key] != value for key, value in candidate.items()):
                raise ValueError('Official labels disagree with frozen class semantics')


def evaluate(args):
    protocol, training = load_final_protocol(args.protocol, args.source)
    release = json.loads(args.release.read_text(encoding='utf-8'))
    if (release.get('schema') != 'wp1-full8000-release-v1'
            or release.get('training_manifest_sha256') != protocol['training_manifest_sha256']
            or release.get('candidate_sha256') != protocol['candidate_sha256']):
        raise ValueError('Release does not match the frozen training/evaluation contract')
    assets = {task: ROOT/release['checkpoints'][task]['path'] for task in ('crop', 'disease')}
    expected = {task: release['checkpoints'][task]['sha256'] for task in assets}
    if any(file_sha256(path) != expected[task] for task, path in assets.items()):
        raise ValueError('A frozen checkpoint changed')
    args.output.mkdir(parents=True, exist_ok=False)
    write_new_json(args.output/'PLAN.json', {'schema': 'wp1-full8000-official-plan-v1',
                   'training_count': 8000, 'candidate_count': 333,
                   'candidate_sha256': protocol['candidate_sha256'], 'checkpoint_sha256': expected,
                   'training_manifest_sha256': protocol['training_manifest_sha256'],
                   'training_source_sha256': protocol['source_sha256'],
                   'official_csv_sha256': ({'seen':file_sha256(args.seen),'unseen':file_sha256(args.unseen)}
                                          if args.execute_evaluation else None),
                   'checkpoint_rule': 'fixed_crop_epoch_2_disease_epoch_3_before_official_evaluation',
                   'test_previously_observed': True, 'batch_size': args.batch_size,
                   'input_policy': 'PIL untransposed RGB; native ResNet50 ImageNet V2 preprocessing',
                   'score_semantics': 'native_float32_sum_component_log_probabilities',
                   'source_code_sha256': {name:file_sha256(ROOT/name) for name in
                       ('wp1_final.py', 'final_protocol.py', 'scripts/evaluate_wp1_full8000.py')}})
    if not args.execute_evaluation:
        print('Plan written. Official metadata/images were not opened.', flush=True)
        return
    seen, unseen = load_metadata(args.seen), load_metadata(args.unseen)
    validate_official_frames(protocol, training, seen, unseen)
    torch.set_num_threads(4)
    predictor = FinalIndependentPredictor(assets['crop'], assets['disease'], args.protocol, args.source, args.device)
    images = args.images.resolve()
    metrics, image_hashes = {}, {}
    for fold, frame in (('seen', seen), ('unseen', unseen)):
        scores, ranks = [], []
        names = frame.image_name.tolist()
        for start in range(0, len(frame), args.batch_size):
            batch = []
            for name in names[start:start+args.batch_size]:
                path = (images/name).resolve()
                if path.parent != images:
                    raise ValueError('Official image resolves outside the declared directory')
                image_hashes[name] = file_sha256(path)
                with Image.open(path) as image:
                    batch.append(image.convert('RGB'))
            result = predictor.predict(batch)
            scores.append(result['scores'].numpy())
            ranks.append(result['topk_composition_ids'].numpy())
            if (start//args.batch_size) % 100 == 0:
                print(f'{fold}: {min(start+args.batch_size,len(frame))}/{len(frame)}', flush=True)
        scores, ranks = np.concatenate(scores), np.concatenate(ranks)
        truths = frame.composition_id.to_numpy(dtype=np.int64)
        metrics[fold] = {'count': len(frame), 'hits_top1': int((ranks[:, 0] == truths).sum()),
                         'hits_top5': int((ranks == truths[:, None]).any(1).sum())}
        np.savez_compressed(args.output/(fold+'_scores.npz'), scores=scores, top5_ids=ranks,
                            filenames=np.asarray(names), truths=truths, candidate_ids=np.arange(333,dtype=np.int64))
        with (args.output/(fold+'_predictions.csv')).open('x', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['image_name','truth_global_id','rank1','rank2','rank3','rank4','rank5'])
            writer.writerows([name, int(truth), *map(int, rank)] for name,truth,rank in zip(names,truths,ranks))
        truth_ids = sorted(set(map(int, truths)))
        matrix = np.zeros((len(truth_ids),333),dtype=np.int64)
        index = {value:row for row,value in enumerate(truth_ids)}
        for truth, prediction in zip(truths, ranks[:,0]):
            matrix[index[int(truth)],int(prediction)] += 1
        with (args.output/(fold+'_confusion.csv')).open('x',newline='',encoding='utf-8') as stream:
            writer=csv.writer(stream)
            writer.writerow(['truth_global_id',*range(333)])
            writer.writerows([value,*map(int,row)] for value,row in zip(truth_ids,matrix))
    if any(file_sha256(path) != expected[task] for task,path in assets.items()):
        raise ValueError('Checkpoint bytes changed during inference')
    if any(file_sha256(images/name) != digest for name,digest in image_hashes.items()):
        raise ValueError('Official image bytes changed during inference')
    plan=json.loads((args.output/'PLAN.json').read_text(encoding='utf-8'))
    if plan['official_csv_sha256']!={'seen':file_sha256(args.seen),'unseen':file_sha256(args.unseen)}:
        raise ValueError('Official CSV bytes changed during inference')
    result = {'schema':'wp1-full8000-official-results-v1', 'training_count':8000, 'candidate_count':333,
              'candidate_sha256':protocol['candidate_sha256'], 'checkpoint_sha256':expected,
              'accuracy_scale':'fraction_0_to_1', **summarize_folds(metrics['seen'],metrics['unseen'])}
    write_new_json(args.output/'metrics.json',result)
    write_new_json(args.output/'COMPLETED.json',{'checkpoint_bytes_unchanged':True,
                   'image_bytes_unchanged':True,'images_processed':3448,'checkpoint_selection_changed':False,
                   'artifacts':{p.name:file_sha256(p) for p in args.output.iterdir() if p.is_file()}})
    print(json.dumps(result,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'artifacts/data/final_full8000_333_v1.json')
    parser.add_argument('--source',type=Path,default=ROOT/'DPD_Small_train.csv')
    parser.add_argument('--seen',type=Path,default=ROOT/'DPD_Small_test_seen.csv')
    parser.add_argument('--unseen',type=Path,default=ROOT/'DPD_Small_test_unseen.csv')
    parser.add_argument('--images',type=Path,default=ROOT/'DPD_Small_dataset')
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--execute-evaluation',action='store_true')
    args=parser.parse_args()
    if args.batch_size <= 0:
        parser.error('Batch size must be positive')
    evaluate(args)


if __name__=='__main__':
    main()
