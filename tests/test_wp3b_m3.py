"""M3 visible-notebook contracts: synthetic CPU only, no CLIP/image inference."""
import ast
import copy
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
import test_wp3b_adapter as reference

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/wp3b_disease_generalization.ipynb'

def notebook():
    return json.loads(NOTEBOOK.read_text())

def namespace():
    nodes, values = [], {}
    for cell in notebook()['cells']:
        if cell['cell_type'] != 'code': continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)): nodes.append(node)
            elif isinstance(node, ast.Assign):
                try: value = ast.literal_eval(node.value)
                except (ValueError, TypeError): continue
                for target in node.targets:
                    if isinstance(target, ast.Name): values[target.id] = value
    ns = {**vars(reference), **values, 'ROOT': ROOT, '__name__': __name__}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<visible M3>', 'exec'), ns)
    return ns

class M3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = namespace()
        cls.split = cls.ns['build_development_split'](ROOT / 'DPD_Small_train.csv')
        cls.table = cls.ns['build_prompt_table'](cls.split)
        cls.contract = cls.ns['make_m3_contract'](cls.split, cls.table, cls.ns['CONFIG'])
        cls.direct = cls.ns['unit_features'](torch.randn(323, 768, generator=torch.Generator().manual_seed(42)))
        proto = cls.ns['component_prototypes'](cls.direct, cls.table, cls.contract['fitting_composition_ids'])
        cls.state = cls.ns['seal_text_state'](cls.direct, proto, cls.contract)
    def setUp(self):
        torch.manual_seed(5)
        self.adapter = self.ns['ResidualVisualAdapter']()
        self.x = torch.randn(2, 768)
        self.rows = self.split.fit.iloc[:2]
        self.c = torch.tensor(self.rows.crop_id.to_numpy())
        self.d = torch.tensor(self.rows.disease_id.to_numpy())
        self.p = torch.tensor(self.rows.composition_id.to_numpy())

    def test_conditional_loss_matches_manual_same_crop_softmax(self):
        y = self.adapter(self.x)
        actual = self.ns['same_crop_loss'](y, self.c, self.d, self.p, self.state)
        logits = torch.nn.functional.normalize(y, dim=1) @ self.state['conditional_text'].T / .07
        values = []
        for i in range(len(y)):
            mask = self.state['conditional_pairs'][:, 0] == self.c[i]
            target = self.state['conditional_ids'][mask].tolist().index(int(self.p[i]))
            values.append(torch.nn.functional.cross_entropy(logits[i, mask][None], torch.tensor([target])))
        torch.testing.assert_close(actual, torch.stack(values).mean(), rtol=0, atol=1e-6)

    def test_only_training_pool_in_conditional_bank(self):
        self.assertEqual(self.state['conditional_ids'].tolist(), self.contract['fitting_composition_ids'])
        self.assertEqual(len(self.state['conditional_ids']), 258)
        self.assertFalse(set(self.state['conditional_ids'].tolist()) & set(self.split.val_unseen.composition_id))

    def test_heldout_text_cannot_change_conditional_loss(self):
        changed = self.direct.clone()
        mask = ~self.table.global_composition_id.isin(self.contract['fitting_composition_ids']).to_numpy()
        changed[mask] = torch.randn(int(mask.sum()), 768, dtype=torch.float64)
        proto = self.ns['component_prototypes'](changed, self.table, self.contract['fitting_composition_ids'])
        state = self.ns['seal_text_state'](changed, proto, self.contract)
        torch.testing.assert_close(state['conditional_text'], self.state['conditional_text'], rtol=0, atol=0)
        torch.testing.assert_close(self.ns['same_crop_loss'](self.x,self.c,self.d,self.p,state),
                                   self.ns['same_crop_loss'](self.x,self.c,self.d,self.p,self.state),rtol=0,atol=0)

    def test_other_crop_vectors_do_not_enter_denominator(self):
        state = copy.deepcopy(self.state)
        other = ~torch.isin(state['conditional_pairs'][:, 0], self.c)
        state['conditional_text'][other] = torch.randn(int(other.sum()), 768)
        a=self.ns['same_crop_loss'](self.x,self.c,self.d,self.p,state)
        b=self.ns['same_crop_loss'](self.x,self.c,self.d,self.p,self.state)
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_singleton_crop_has_finite_zero_conditional_loss(self):
        state={'conditional_ids':torch.tensor([100]),'conditional_pairs':torch.tensor([[12,7]]),
               'conditional_text':torch.nn.functional.normalize(torch.randn(1,768),dim=1)}
        loss=self.ns['same_crop_loss'](self.x[:1],torch.tensor([12]),torch.tensor([7]),torch.tensor([100]),state)
        self.assertEqual(loss.item(),0.)

    def test_mismatched_pair_and_heldout_targets_rejected(self):
        for c,d,p in [(self.c+100,self.d,self.p),(self.c,self.d+1000,self.p),
                      (self.c,self.d,torch.full_like(self.p,int(self.split.val_unseen.composition_id.iloc[0]))),
                      (self.c.float(),self.d,self.p)]:
            with self.assertRaises(ValueError): self.ns['same_crop_loss'](self.x,c,d,p,self.state)

    def test_invalid_nonfinite_embeddings_rejected(self):
        for x in [torch.zeros(2,768),torch.full((2,768),float('nan')),torch.ones(2,12)]:
            with self.assertRaises(ValueError):self.ns['same_crop_loss'](x,self.c,self.d,self.p,self.state)

    def test_control_is_exact_original_M2_objective(self):
        y=self.adapter(self.x)
        a=self.ns['m3_training_objective'](y,self.x,self.c,self.d,self.p,self.state,0.)
        b=self.ns['adapter_objective'](y,self.x,self.c,self.d,self.state)
        torch.testing.assert_close(a['loss'],b['loss'],rtol=0,atol=0)

    def test_fixed_weighted_intervention_math(self):
        y=self.adapter(self.x)
        a=self.ns['m3_training_objective'](y,self.x,self.c,self.d,self.p,self.state,1.)
        b=self.ns['adapter_objective'](y,self.x,self.c,self.d,self.state)
        torch.testing.assert_close(a['loss'],b['loss']+a['conditional_loss'],rtol=0,atol=0)
        with self.assertRaises(ValueError):self.ns['m3_training_objective'](y,self.x,self.c,self.d,self.p,self.state,.5)

    def test_gradients_adapter_only_and_finite(self):
        x=self.x.requires_grad_(True)
        a=self.ns['m3_training_objective'](self.adapter(x),x,self.c,self.d,self.p,self.state,1.)
        a['loss'].backward()
        self.assertIsNone(x.grad)
        self.assertGreater(self.adapter.up.weight.grad.abs().sum().item(),0)
        self.assertTrue(all(t.grad is not None and torch.isfinite(t.grad).all() for t in self.adapter.parameters()))
        self.assertTrue(all(not self.state[k].requires_grad for k in ['direct','crop','disease','conditional_text']))

    def test_conditional_term_pushes_true_above_healthy_negative(self):
        state={'conditional_ids':torch.tensor([10,70]),'conditional_pairs':torch.tensor([[2,1],[2,9]]),
               'conditional_text':torch.eye(768)[:2]}
        y=torch.zeros(1,768);y[0,1]=1.;y.requires_grad_(True)
        loss=self.ns['same_crop_loss'](y,torch.tensor([2]),torch.tensor([1]),torch.tensor([10]),state)
        loss.backward();self.assertLess(y.grad[0,0].item(),0.)
        self.assertTrue(torch.isfinite(y.grad).all())

    def test_frozen_backbone_enforcement(self):
        model=torch.nn.Linear(3,768).eval().requires_grad_(False)
        self.ns['require_frozen_clip'](model)
        model.train()
        with self.assertRaises(ValueError):self.ns['require_frozen_clip'](model)
        model.eval().requires_grad_(True)
        with self.assertRaises(ValueError):self.ns['require_frozen_clip'](model)

    def test_identity_and_global_ranking_compatibility(self):
        a=self.ns['cosine_candidate_scores'](self.x.half(),self.direct)
        b=self.ns['cosine_candidate_scores'](self.adapter(self.x.half()),self.direct)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        ids=(10,70,332);scores=torch.ones(1,3,dtype=torch.float64)
        self.assertEqual(self.ns['rank_global_ids'](scores,ids)[0].tolist(),list(ids))
        scores[0,1]=torch.nextafter(scores[0,1],torch.tensor(float('inf'),dtype=torch.float64))
        self.assertEqual(self.ns['rank_global_ids'](scores,ids)[0,0],70)

    def test_checkpoint_roundtrip_no_overwrite_and_arm_mismatch(self):
        with reference.workspace_temporary_directory() as folder:
            path=Path(folder)/'m3.pt'
            self.ns['save_adapter_checkpoint'](path,self.adapter,self.state,self.contract,{'epoch':0})
            a,state,record=self.ns['load_adapter_checkpoint'](path,self.contract)
            torch.testing.assert_close(a(self.x),self.adapter(self.x),rtol=0,atol=0)
            self.assertEqual(record['epoch'],0)
            with self.assertRaises(FileExistsError):self.ns['save_adapter_checkpoint'](path,self.adapter,self.state,self.contract,{'epoch':0})
            bad=copy.deepcopy(self.contract);bad['hyperparameters']['conditional_weight']=0.
            with self.assertRaises(ValueError):self.ns['load_adapter_checkpoint'](path,bad)

    def test_checkpoint_bank_tampering_rejected(self):
        bad=copy.deepcopy(self.state);bad['conditional_text'][0,0]+=.1
        with self.assertRaises(ValueError):self.ns['validate_text_state'](bad,self.contract)
        bad=copy.deepcopy(self.state);bad['conditional_ids'][0]+=1000
        with self.assertRaises(ValueError):self.ns['validate_text_state'](bad,self.contract)

    def test_contract_budget_and_frozen_protocol(self):
        c=self.contract
        self.assertEqual(c['candidate_ids'],list(self.split.candidate_ids))
        self.assertEqual(c['hyperparameters']['epochs'],5)
        for key,value in [('epochs',6),('batch_size',16),('seed',1),('conditional_weight',.5),('learning_rate',.001)]:
            with self.assertRaises(ValueError):self.ns['make_m3_contract'](self.split,self.table,{**self.ns['CONFIG'],key:value})

    def test_validation_uses_base_loss_only(self):
        source='\n'.join(''.join(c['source']) for c in notebook()['cells'] if c['cell_type']=='code')
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='validate_adapter')
        calls=[ast.unparse(n.func) for n in ast.walk(node) if isinstance(n,ast.Call)]
        self.assertIn('adapter_objective',calls);self.assertNotIn('m3_training_objective',calls)

    def test_model_and_scoring_definitions_unchanged(self):
        def defs(book):return {n.name:ast.dump(n,include_attributes=False) for c in book['cells'] if c['cell_type']=='code' for n in ast.parse(''.join(c['source'])).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
        old=defs(reference.notebook());new=defs(notebook())
        for name in ['ResidualVisualAdapter','adapter_objective','build_development_split','normalize_label','build_prompt_table','cosine_candidate_scores','rank_global_ids','better_selection','selection_record']:
            self.assertEqual(old[name],new[name],name)


    def test_bounded_epoch_updates_only_adapter(self):
        loader,frame=reference.AdapterTests.synthetic_loader(self)
        model=reference.AdapterTests.fake_clip(self)
        before={k:v.clone() for k,v in model.state_dict().items()}
        optimizer=self.ns['make_adapter_optimizer'](self.adapter,self.ns['CONFIG'])
        initial=self.adapter.up.weight.detach().clone()
        result=self.ns['train_adapter_epoch'](model,self.adapter,loader,optimizer,self.state,frame,'cpu',False)
        self.assertEqual(result['sample_count'],3)
        self.assertTrue(torch.isfinite(torch.tensor(result['conditional_loss'])))
        self.assertFalse(torch.equal(initial,self.adapter.up.weight))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(all(torch.equal(before[k],v) for k,v in model.state_dict().items()))

    def test_validation_preserves_weights_and_buffers(self):
        loader,frame=reference.AdapterTests.synthetic_loader(self,'val_seen')
        model=reference.AdapterTests.fake_clip(self)
        before={k:v.clone() for k,v in model.state_dict().items()}
        result=self.ns['validate_adapter'](model,self.adapter,loader,self.state,frame,self.split.candidate_ids,'cpu',False)
        self.assertEqual(result['sample_count'],3)
        self.assertTrue(all(torch.equal(before[k],v) for k,v in model.state_dict().items()))

    def test_disguised_validation_rejected_before_forward(self):
        loader,_=reference.AdapterTests.synthetic_loader(self,'val_seen');loader.dataset.role='fit'
        model=reference.AdapterTests.fake_clip(self)
        optimizer=self.ns['make_adapter_optimizer'](self.adapter,self.ns['CONFIG'])
        with patch.object(model,'encode_image',side_effect=AssertionError('no encoding')):
            with self.assertRaises(ValueError):self.ns['train_adapter_epoch'](model,self.adapter,loader,optimizer,self.state,self.split.fit,'cpu',False)

    def complete_predictions(self):
        pd=self.ns['pd'];rows=[];ids=list(self.split.candidate_ids)
        for role in ['val_seen','val_unseen']:
            for r in getattr(self.split,role).itertuples():
                rank=[int(r.composition_id)]+[i for i in ids if i!=r.composition_id][:4]
                rows.append({'fold':role,'image_name':r.image_name,'truth_global_id':int(r.composition_id),
                             **{'rank_'+str(k+1):v for k,v in enumerate(rank)}})
        return pd.DataFrame(rows)

    def test_diagnostics_perfect_and_missing_records_rejected(self):
        pred=self.complete_predictions();d=self.ns['diagnose_predictions'](pred,self.split)
        self.assertEqual(d['val_unseen']['correct1'],1894)
        self.assertEqual(d['val_unseen']['diseased_correct'],1496)
        self.assertEqual(d['val_unseen']['false_healthy'],0)
        self.assertEqual(len([r for r in d['per_pair'] if r['fold']=='val_unseen']),65)
        for bad in [pred.iloc[:-1],self.ns['pd'].concat([pred,pred.iloc[:1]],ignore_index=True)]:
            with self.assertRaises(ValueError):self.ns['diagnose_predictions'](bad,self.split)

    def test_diagnostics_unknown_and_duplicate_rank_ids_rejected(self):
        for value in [99999,int(self.split.candidate_ids[0])]:
            pred=self.complete_predictions()
            if value==99999:pred.loc[0,'rank_1']=value
            else:pred.loc[0,'rank_2']=pred.loc[0,'rank_1']
            with self.assertRaises(ValueError):self.ns['diagnose_predictions'](pred,self.split)

    def test_paired_alignment_and_counts(self):
        pred=self.complete_predictions();other=pred.copy();other.loc[0,'rank_1']=other.loc[0,'rank_2']
        result=self.ns['paired_correctness'](pred,other)
        self.assertEqual(result['val_seen']['first_only'],1)
        self.assertEqual(result['val_unseen']['both_correct'],1894)
        with self.assertRaises(ValueError):self.ns['paired_correctness'](pred,other.iloc[:-1])

    def test_comparison_budget_and_disease_recovery_gate(self):
        def summary(weight,seen,unseen,diseased,false_healthy):
            return {'protocol':{'candidate_ids':list(self.split.candidate_ids)},
                'config':{**self.ns['CONFIG'],'conditional_weight':weight},
                'history':[{'epoch':i,'train_sample_count':None if i==0 else 4868} for i in range(6)],
                'selected':{'seen_top1':seen/1238,'unseen_top1':unseen/1894,
                            'h1':self.ns['harmonic_mean'](seen/1238,unseen/1894)},
                'diagnostics':{'val_seen':{'count':1238,'correct1':seen},
                               'val_unseen':{'count':1894,'correct1':unseen,'diseased_correct':diseased,'false_healthy':false_healthy}}}
        control=summary(0.,286,178,16,490);new=summary(1.,300,230,90,300)
        self.assertTrue(self.ns['compare_pilot_summaries'](control,new)['supports_exploratory_success'])
        weak=summary(1.,500,180,20,300)
        self.assertFalse(self.ns['compare_pilot_summaries'](control,weak)['supports_exploratory_success'])
        for key,value in [('epochs',4),('seed',7)]:
            bad=copy.deepcopy(new);bad['config'][key]=value
            with self.assertRaises(ValueError):self.ns['compare_pilot_summaries'](control,bad)
        bad=copy.deepcopy(new);bad['history'][1]['train_sample_count']=100
        with self.assertRaises(ValueError):self.ns['compare_pilot_summaries'](control,bad)

    def test_export_retains_epoch1_and_all_five_rankings(self):
        pred=self.complete_predictions();validation={}
        for role,part in pred.groupby('fold',sort=False):
            ranking=torch.tensor(part[['rank_'+str(k) for k in range(1,6)]].to_numpy())
            validation[role]={'filenames':part.image_name.tolist(),'truths':torch.tensor(part.truth_global_id.to_numpy()),
                              'ranking':ranking,'control_ranking':ranking.clone()}
        epoch_rows=[]
        for epoch in range(6):epoch_rows.extend(self.ns['prediction_rows'](validation,epoch))
        pilot={'history':[{'epoch':i} for i in range(6)],'best_record':{'epoch':5},'baseline_record':{'epoch':0},
               'best_checkpoint':'synthetic-fixture-only','strict_H1_gain_over_identity':False,
               'best_validation':validation,'epoch_predictions':epoch_rows}
        with reference.workspace_temporary_directory() as directory:
            paths=self.ns['export_m3_results'](directory,self.contract,pilot,self.split)
            self.assertEqual(len(paths),4)
            records=self.ns['pd'].read_csv(Path(directory)/'adapter_epoch_predictions.csv')
            self.assertEqual(len(records[records.epoch==1]),3132)
            summary=json.loads((Path(directory)/'adapter_summary.json').read_text())
            self.assertEqual(summary['protocol']['direct_prompt_sha256'],self.contract['direct_prompt_sha256'])
            with self.assertRaises(FileExistsError):self.ns['export_m3_results'](directory,self.contract,pilot,self.split)

    def test_historical_checkpoint_diagnostics_are_safe_cpu_reads(self):
        with patch('torch.load',return_value={'schema':'wrong'}) as load:
            with self.assertRaises(ValueError):self.ns['inspect_saved_text_geometry']('synthetic.pt',self.split)
        self.assertIs(load.call_args.kwargs['weights_only'],True)
        self.assertEqual(load.call_args.kwargs['map_location'],'cpu')

    def test_actual_history_row_preserves_processed_budget_and_new_loss(self):
        keys=['loss','crop_loss','disease_loss','identity_loss','elapsed_seconds','peak_allocated_bytes','peak_reserved_bytes']
        part={**{k:1. for k in keys},'sample_count':3,'metrics':{'top1':.5,'top5':1.}}
        train={**{k:1. for k in keys},'sample_count':4868,'conditional_loss':.75}
        row=self.ns['history_row'](1,train,{'val_seen':part,'val_unseen':part})
        self.assertEqual(row['train_sample_count'],4868)
        self.assertEqual(row['train_conditional_loss'],.75)
        baseline=self.ns['history_row'](0,None,{'val_seen':part,'val_unseen':part})
        self.assertIsNone(baseline['train_sample_count'])

class M3StructureTests(unittest.TestCase):
    def test_notebook_json_and_no_hidden_pipeline(self):
        b=notebook();self.assertEqual(b['nbformat'],4)
        self.assertEqual(len(b['cells']),len({c['id'] for c in b['cells']}))
        for c in b['cells']:
            self.assertFalse(c.get('outputs'));self.assertIsNone(c.get('execution_count'));self.assertFalse(c.get('attachments'))
            if c['cell_type']=='code':
                s=''.join(c['source']);compile(s,c['id'],'exec')
                for n in ast.walk(ast.parse(s)):
                    if isinstance(n,ast.ImportFrom):self.assertNotIn((n.module or '').split('.')[0],['dataset','data_utils','development_data','train_multitask','evaluation'])
                    if isinstance(n,ast.Call):self.assertNotIn(ast.unparse(n.func),['exec','eval','compile','importlib.import_module'])
        self.assertNotIn('DPD_Small_test_',NOTEBOOK.read_text())
        self.assertNotIn('%run',NOTEBOOK.read_text())

    def test_cold_default_run_all_is_safe(self):
        ns={'__name__':__name__,'display':lambda *a,**k:None}
        with patch('clip.load',side_effect=AssertionError('no model')),patch('PIL.Image.open',side_effect=AssertionError('no image')),patch('torch.save',side_effect=AssertionError('no save')),patch.object(Path,'write_text',side_effect=AssertionError('no write')),reference.redirect_stdout(io.StringIO()):
            for c in notebook()['cells']:
                if c['cell_type']=='code':exec(compile(''.join(c['source']),c['id'],'exec'),ns)
        for name in ['PREDECLARE_RUN','LOAD_CLIP_MODEL','RUN_TEXT_ENCODING','BUILD_ADAPTER','RUN_TRAINING','RUN_DEVELOPMENT_EVALUATION','SAVE_CHECKPOINTS','EXPORT_RESULTS','VERIFY_CHECKPOINT','RUN_RECORD_DIAGNOSTICS','RUN_COMPARISON_REPORT','RUN_HISTORICAL_EPOCH1_EVALUATION']:
            self.assertIs(ns[name],False)
        self.assertIsNone(ns['FROZEN_CLIP']);self.assertIsNone(ns['RUN_DIRECTORY'])

if __name__=='__main__':unittest.main()
