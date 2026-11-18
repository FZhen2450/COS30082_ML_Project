"""Replay full-training WP1 saved scores without loading models or images."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
from data_utils import load_metadata
from final_protocol import file_sha256, load_final_protocol, write_new_json
from scripts.evaluate_wp1_full8000 import summarize_folds, validate_official_frames


def validate_saved_scores(packet, names, truths):
    required={'scores','top5_ids','filenames','truths','candidate_ids'}
    if set(packet)!=required:
        raise ValueError('Unexpected or missing final score fields')
    scores=packet['scores']
    if scores.dtype!=np.float32 or scores.shape!=(len(names),333) or not np.isfinite(scores).all():
        raise ValueError('Require complete finite native float32 scores')
    ids=packet['candidate_ids']
    if ids.dtype!=np.int64 or not np.array_equal(ids,np.arange(333,dtype=np.int64)):
        raise ValueError('Final global candidate order or ID type changed')
    if packet['filenames'].dtype.kind!='U' or packet['filenames'].tolist()!=list(names):
        raise ValueError('Saved row identities differ')
    if packet['truths'].dtype!=np.int64 or not np.array_equal(packet['truths'],truths):
        raise ValueError('Saved truth identities differ')
    ranks=np.argsort(-scores,axis=1,kind='stable')[:,:5]
    if packet['top5_ids'].dtype!=np.int64 or not np.array_equal(packet['top5_ids'],ranks):
        raise ValueError('Saved ranks differ from native scores or have invalid ID types')
    return ranks


def verify(run, protocol_path, source, seen_source, unseen_source):
    protocol,training=load_final_protocol(protocol_path,source)
    seen,unseen=load_metadata(seen_source),load_metadata(unseen_source)
    validate_official_frames(protocol,training,seen,unseen)
    plan=json.loads((run/'PLAN.json').read_text(encoding='utf-8'))
    completed=json.loads((run/'COMPLETED.json').read_text(encoding='utf-8'))
    result=json.loads((run/'metrics.json').read_text(encoding='utf-8'))
    checks=[]
    def check(condition,name):
        if not condition:
            raise ValueError(name)
        checks.append(name)
    check(plan['candidate_sha256']==protocol['candidate_sha256'],'verified final candidate identity')
    check(plan['training_manifest_sha256']==protocol['training_manifest_sha256'],'full-training identity')
    check(plan['official_csv_sha256']=={'seen':file_sha256(seen_source),'unseen':file_sha256(unseen_source)},'official CSV byte identities')
    for name,digest in completed['artifacts'].items():
        check(Path(name).name==name and file_sha256(run/name)==digest,'saved artifact bytes: '+name)
    folds={}
    for fold,frame in (('seen',seen),('unseen',unseen)):
        with np.load(run/(fold+'_scores.npz'),allow_pickle=False) as saved:
            packet={key:saved[key] for key in saved.files}
        scores=packet['scores']
        check(scores.dtype==np.float32 and scores.shape==(len(frame),333) and np.isfinite(scores).all(),fold+' canonical native score shape/precision')
        check(packet['candidate_ids'].dtype==np.int64 and np.array_equal(packet['candidate_ids'],np.arange(333)),fold+' ordered global IDs')
        truths=frame.composition_id.to_numpy(dtype=np.int64)
        check(packet['filenames'].tolist()==frame.image_name.tolist(),fold+' exact filename order')
        check(np.array_equal(packet['truths'],truths),fold+' exact truth identities')
        ranks=validate_saved_scores(packet,frame.image_name.tolist(),truths)
        check(np.array_equal(packet['top5_ids'],ranks),fold+' all saved Top-5 rows')
        with (run/(fold+'_predictions.csv')).open(newline='',encoding='utf-8') as stream:
            rows=list(csv.reader(stream))
        expected=[['image_name','truth_global_id','rank1','rank2','rank3','rank4','rank5']]
        expected.extend([name,str(truth),*map(str,rank)] for name,truth,rank in zip(frame.image_name,truths,ranks))
        check(rows==expected,fold+' named prediction CSV')
        ids=sorted(set(map(int,truths)))
        index={value:i for i,value in enumerate(ids)}
        matrix=np.zeros((len(ids),333),dtype=np.int64)
        for truth,prediction in zip(truths,ranks[:,0]):
            matrix[index[int(truth)],int(prediction)]+=1
        with (run/(fold+'_confusion.csv')).open(newline='',encoding='utf-8') as stream:
            rows=list(csv.reader(stream))
        expected=[['truth_global_id',*map(str,range(333))]]
        expected.extend([str(value),*map(str,row)] for value,row in zip(ids,matrix))
        check(rows==expected,fold+' complete confusion matrix')
        folds[fold]={'count':len(frame),'hits_top1':int((ranks[:,0]==truths).sum()),
                     'hits_top5':int((ranks==truths[:,None]).any(1).sum())}
    metrics=summarize_folds(folds['seen'],folds['unseen'])
    check(all(result[key]==value for key,value in metrics.items()),'Overall/Seen/Unseen Top-1/Top-5 and H exact replay')
    check(result['checkpoint_sha256']==plan['checkpoint_sha256'],'metric checkpoint identities')
    check(result['candidate_count']==333 and result['training_count']==8000
          and result['candidate_sha256']==protocol['candidate_sha256'],'metric task and training scope')
    check(completed['images_processed']==3448 and completed['checkpoint_selection_changed'] is False,'complete frozen inference declaration')
    return {'schema':'wp1-full8000-score-replay-v1','passed':len(checks),'checks':checks,
            'reproduced_top5_rows':3448,'metrics':metrics,'new_model_inference':False,
            'second_human_inference_replay':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,default=ROOT/'artifacts/data/final_full8000_333_v1.json')
    parser.add_argument('--source',type=Path,default=ROOT/'DPD_Small_train.csv')
    parser.add_argument('--seen',type=Path,default=ROOT/'DPD_Small_test_seen.csv')
    parser.add_argument('--unseen',type=Path,default=ROOT/'DPD_Small_test_unseen.csv')
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args()
    receipt=verify(args.run,args.protocol,args.source,args.seen,args.unseen)
    write_new_json(args.receipt,receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
