"""Notebook-local clean gating; no model, real image, weights or GPU execution."""
import ast,copy,json,unittest
from pathlib import Path
import numpy as np
import torch
NB=Path(__file__).resolve().parents[1]/'notebooks/gui_native_clean_parity.ipynb'

def notebook_definitions():
    namespace={'np':np,'torch':torch,'Path':Path,'json':json}
    import numbers,hashlib,time
    namespace.update({'numbers':numbers,'hashlib':hashlib,'time':time})
    cells=json.loads(NB.read_text())['cells']
    for cell in cells:
        if cell['cell_type']=='code':
            tree=ast.parse(''.join(cell['source']))
            nodes=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))]
            if nodes:exec(compile(ast.Module(body=nodes,type_ignores=[]),str(NB),'exec'),namespace)
    return namespace

class ParityGateTests(unittest.TestCase):
    def setUp(self):
        self.ns=notebook_definitions()
        self.assertIn('run_clean_gate',self.ns,'Complete notebook-local clean gate is required')
        self.assertIn('run_guarded_corruptions',self.ns,'A guarded corruption boundary is required')
        self.ids=[3,7,9,11,15];self.folds={'seen':['s0','s1','s2'],'unseen':['u0','u1']};self.truths={'seen':[3,3,3],'unseen':[3,3]}
        self.counts={'seen':3,'unseen':2};self.predictors={};self.refs={}
        for model in ('frozen','m6'):
            self.predictors[model]=CountingPredictor(self.ids)
            self.refs[model]={}
            for fold,names in self.folds.items():
                self.refs[model][fold]={'candidate_ids':self.ids,'image_names':names.copy(),'ground_truth_ids':self.truths[fold].copy(),'ranked_ids':[[3,7,9,11,15] for _ in names],'available_k':5}
    def gate(self):
        return self.ns['run_clean_gate'](self.predictors,self.folds,self.truths,self.ids,self.refs,lambda name:name,batch_size=2,expected_counts=self.counts)
    def test_complete_two_model_two_fold_gate_preserves_final_partial_batches(self):
        result=self.gate();self.assertTrue(result['passed']);self.assertEqual(result['processed_rows'],10)
        for model in self.predictors.values():self.assertEqual(model.batch_sizes,[2,1,2])
    def test_last_row_last_model_mismatch_blocks_every_corrupted_call(self):
        self.refs['m6']['unseen']['ranked_ids'][-1]=[7,3,9,11,15]
        result=self.gate();self.assertFalse(result['passed']);self.assertEqual(result['mismatch_rows'],1)
        calls=[]
        with self.assertRaises(ValueError):self.ns['run_guarded_corruptions'](result,True,lambda:calls.append('corrupted'),expected_task_sha256=self.expected_task())
        self.assertEqual(calls,[]);self.assertEqual(result['mismatches'][0]['filename'],'u1')
    def test_clean_success_still_requires_explicit_corruption_enablement(self):
        calls=[];out=self.ns['run_guarded_corruptions'](self.gate(),False,lambda:calls.append('corrupted'),expected_task_sha256=self.expected_task())
        self.assertIsNone(out);self.assertEqual(calls,[])
    def test_complete_pass_and_explicit_flag_reach_the_real_runner_boundary(self):
        calls=[];result=self.ns['run_guarded_corruptions'](self.gate(),True,lambda:calls.append('ran') or 17,expected_task_sha256=self.expected_task())
        self.assertEqual(result,17);self.assertEqual(calls,['ran'])
    def test_missing_model_fold_rejected_before_any_prediction(self):
        del self.refs['m6']['unseen']
        with self.assertRaises(ValueError):self.gate()
        self.assertTrue(all(not p.batch_sizes for p in self.predictors.values()))
    def test_duplicate_and_overlapping_filenames_rejected_before_prediction(self):
        for names in (['s0','s0','s2'],['u0','s1','s2']):
            self.folds['seen']=names
            with self.assertRaises(ValueError):self.gate()
        self.assertTrue(all(not p.batch_sizes for p in self.predictors.values()))
    def test_reference_reordered_names_rejected_before_prediction(self):
        self.refs['m6']['unseen']['image_names'].reverse()
        with self.assertRaises(ValueError):self.gate()
        self.assertTrue(all(not p.batch_sizes for p in self.predictors.values()))
    def test_reference_wrong_truth_rejected_before_prediction(self):
        self.refs['m6']['unseen']['ground_truth_ids'][-1]=7
        with self.assertRaises(ValueError):self.gate()
        self.assertTrue(all(not p.batch_sizes for p in self.predictors.values()))
    def test_reference_incomplete_topfive_rejected(self):
        self.refs['m6']['unseen']['ranked_ids'][-1]=[3]
        with self.assertRaises(ValueError):self.gate()
    def test_invalid_global_ids_are_rejected_before_integer_casting(self):
        for ids in ([3.0,7,9,11,15],[True,7,9,11,15],['3',7,9,11,15],np.array([3.,7.,9.,11.,15.]),[3,7,9,11,2**65]):
            with self.subTest(ids=ids):
                self.ids=ids
                with self.assertRaises(ValueError):self.gate()
    def test_reordered_duplicate_and_unknown_candidate_ids_rejected(self):
        for ids in ([7,3,9,11,15],[3,3,9,11,15],[3,7,9,11,17]):
            self.ids=ids
            with self.assertRaises(ValueError):self.gate()
    def test_fractional_predictor_ids_and_rank_ids_rejected(self):
        for key in ('candidate_ids','topk_composition_ids'):
            predictor=self.predictors['m6'];predictor.bad=key
            with self.assertRaises(ValueError):self.gate()
            predictor.bad=None
    def test_nonfinite_float32_and_wrong_shape_scores_rejected(self):
        for bad in ('nan','float32','shape'):
            self.predictors['m6'].bad=bad
            with self.assertRaises(ValueError):self.gate()
    def test_predicted_rank_must_agree_with_own_stable_scores(self):
        self.predictors['m6'].bad='rank'
        with self.assertRaises(ValueError):self.gate()
    def test_exact_ties_use_ascending_global_ids(self):
        for predictor in self.predictors.values():predictor.tie=True
        self.assertTrue(self.gate()['passed'])
    def test_wrong_fold_counts_and_empty_loaders_rejected(self):
        self.counts={'seen':4,'unseen':2}
        with self.assertRaises(ValueError):self.gate()
        self.folds['seen']=[];self.counts={'seen':0,'unseen':2}
        with self.assertRaises(ValueError):self.gate()
    def test_late_predictor_exception_never_authorizes_corruptions(self):
        self.predictors['m6'].bad='raise'
        with self.assertRaises(RuntimeError):self.gate()
    def test_default_notebook_execution_does_not_load_models_or_run_inference(self):
        notebook=json.loads(NB.read_text());flags={}
        for cell in notebook['cells']:
            if cell['cell_type']=='code':
                self.assertIsNone(cell.get('execution_count'));self.assertFalse(cell.get('outputs'))
                for n in ast.parse(''.join(cell['source'])).body:
                    if isinstance(n,ast.Assign):
                        for target in n.targets:
                            if isinstance(target,ast.Name) and target.id in ('LOAD_MODELS','RUN_CLEAN_PARITY','RUN_CORRUPTIONS'):
                                flags[target.id]=ast.literal_eval(n.value)
        self.assertEqual(flags,{'LOAD_MODELS':False,'RUN_CLEAN_PARITY':False,'RUN_CORRUPTIONS':False})
    def test_mutation_after_gate_invalidates_corruption_receipt(self):
        result=self.gate();result['processed_rows']-=1
        with self.assertRaises(ValueError):self.ns['run_guarded_corruptions'](result,True,lambda:None,expected_task_sha256=self.expected_task())

    def test_single_model_receipt_cannot_unlock_two_model_robustness(self):
        del self.predictors['m6'];del self.refs['m6']
        result=self.gate()
        with self.assertRaises(ValueError):
            self.ns['run_guarded_corruptions'](result,True,lambda:None,expected_task_sha256=self.expected_task())
    def expected_task(self):
        import hashlib
        payload={'schema':'actual-gui-clean-task-v1','models':['frozen','m6'],'candidate_ids':[3,7,9,11,15],
                 'folds':{'seen':['s0','s1','s2'],'unseen':['u0','u1']},'truths':{'seen':[3,3,3],'unseen':[3,3]},'counts':{'seen':3,'unseen':2}}
        return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    def test_late_exception_persists_failure_without_success_receipt(self):
        import tempfile
        self.assertIn('persist_clean_run',self.ns,'Late failures need immutable evidence')
        with tempfile.TemporaryDirectory() as directory:
            self.predictors['m6'].bad='raise'
            with self.assertRaises(RuntimeError):self.ns['persist_clean_run'](self.gate,Path(directory))
            saved=json.loads((Path(directory)/'clean_failure.json').read_text())
            self.assertFalse(saved['passed']);self.assertEqual(saved['error_type'],'RuntimeError')
            self.assertFalse((Path(directory)/'clean_receipt.json').exists())

    def test_publication_uses_its_own_checkout_source_root(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'wp4').mkdir();(root/'notebooks').mkdir()
            (root/'wp4/clip_predictor.py').write_text('# source fixture')
            (root/'notebooks/gui_native_clean_parity.ipynb').write_text('{}')
            self.ns['ROOT']=root/'separate-private-assets'
            self.assertEqual(self.ns['find_source_root'](root/'notebooks'),root.resolve())

class CountingPredictor:
    def __init__(self,ids):self.ids=ids;self.batch_sizes=[];self.bad=None;self.tie=False
    def predict(self,images,topk=5):
        self.batch_sizes.append(len(images))
        if self.bad=='raise':raise RuntimeError('Predictor failed')
        scores=torch.zeros((len(images),5),dtype=torch.float64) if self.tie else torch.tensor([[5.,4.,3.,2.,1.]]*len(images),dtype=torch.float64)
        ids=torch.tensor(self.ids);ranks=ids.repeat(len(images),1)
        if self.bad=='nan':scores[0,0]=float('nan')
        if self.bad=='float32':scores=scores.float()
        if self.bad=='shape':scores=scores[:,:4]
        if self.bad=='candidate_ids':ids=ids.double()+.1
        if self.bad=='topk_composition_ids':ranks=ranks.double()+.1
        if self.bad=='rank':ranks[:,[0,1]]=ranks[:,[1,0]]
        return {'candidate_ids':ids,'scores':scores,'topk_composition_ids':ranks,'score_semantics':'direct_float64_clip_cosine_similarity'}

if __name__=='__main__':unittest.main(verbosity=2)
