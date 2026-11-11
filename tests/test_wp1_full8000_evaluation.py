"""Official-summary arithmetic and metadata checks; no real image inference."""
import copy
from pathlib import Path
import unittest
import pandas as pd
import numpy as np

from data_utils import load_metadata
from final_protocol import load_final_protocol
from scripts.evaluate_wp1_full8000 import summarize_folds, validate_official_frames
from scripts.verify_wp1_full8000 import validate_saved_scores

ROOT = Path(__file__).resolve().parents[1]


class FinalEvaluationTests(unittest.TestCase):
    def test_saved_score_replay_rejects_changed_precision_rows_ids_and_rank_types(self):
        packet={'scores':np.zeros((2,333),dtype=np.float32),'top5_ids':np.tile(np.arange(5,dtype=np.int64),(2,1)),
                'filenames':np.array(['one.jpg','two.jpg']),'truths':np.array([0,4],dtype=np.int64),
                'candidate_ids':np.arange(333,dtype=np.int64)}
        ranks=validate_saved_scores(packet,['one.jpg','two.jpg'],np.array([0,4],dtype=np.int64))
        self.assertEqual(ranks[0].tolist(),[0,1,2,3,4])
        for field in ('scores','top5_ids','candidate_ids','filenames','truths'):
            changed=copy.deepcopy(packet)
            if field=='filenames':
                changed[field]=changed[field][::-1]
            elif field=='scores':
                changed[field]=changed[field].astype(np.float64)
            else:
                changed[field]=changed[field].astype(np.float64)
            with self.subTest(field=field),self.assertRaises(ValueError):
                validate_saved_scores(changed,['one.jpg','two.jpg'],np.array([0,4],dtype=np.int64))

    def test_metadata_identity_validation_rejects_overlap_duplicates_and_label_changes(self):
        protocol,training=load_final_protocol(ROOT/'artifacts/data/final_full8000_333_v1.json',ROOT/'DPD_Small_train.csv')
        seen_id=int(training.iloc[0].composition_id)
        def fixture(candidate_id,count,prefix):
            candidate=protocol['candidates'][candidate_id]
            return pd.DataFrame([{**candidate,'image_name':f'{prefix}-{i}.jpg'} for i in range(count)])
        seen=fixture(seen_id,2000,'synthetic-seen')
        unseen=fixture(39,1448,'synthetic-unseen')
        validate_official_frames(protocol,training,seen,unseen)
        for mutation in ('duplicate','training_overlap','fold_overlap','label'):
            left,right=seen.copy(),unseen.copy()
            if mutation=='duplicate':
                left.loc[0,'image_name']=left.loc[1,'image_name']
            elif mutation=='training_overlap':
                left.loc[0,'image_name']=training.iloc[0].image_name
            elif mutation=='fold_overlap':
                right.loc[0,'image_name']=left.loc[0,'image_name']
            else:
                right.loc[0,'disease_name']='incorrect-label'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                validate_official_frames(protocol,training,left,right)

    def test_overall_uses_total_hits_and_h_is_separate(self):
        result = summarize_folds({'count':2000,'hits_top1':1000,'hits_top5':1500},
                                 {'count':1448,'hits_top1':100,'hits_top5':200})
        self.assertEqual(result['overall']['count'],3448)
        self.assertEqual(result['overall']['top1'],1100/3448)
        self.assertEqual(result['overall']['top5'],1700/3448)
        self.assertNotEqual(result['overall']['top1'],result['harmonic_mean_top1'])
        self.assertNotEqual(result['overall']['top1'],(result['seen']['top1']+result['unseen']['top1'])/2)

    def test_missing_fold_or_impossible_hits_cannot_improve_denominators(self):
        for unseen in ({'count':0,'hits_top1':0,'hits_top5':0},
                       {'count':1448,'hits_top1':10,'hits_top5':9}):
            with self.assertRaises(ValueError):
                summarize_folds({'count':2000,'hits_top1':1,'hits_top5':2},unseen)

    def test_zero_fold_scores_have_zero_h(self):
        result=summarize_folds({'count':2000,'hits_top1':0,'hits_top5':0},
                               {'count':1448,'hits_top1':0,'hits_top5':0})
        self.assertEqual(result['harmonic_mean_top1'],0)
        self.assertEqual(result['overall']['top1'],0)


if __name__=='__main__':
    unittest.main()
