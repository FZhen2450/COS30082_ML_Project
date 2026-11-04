"""Bounded residual and immutable-cache contracts; synthetic CPU only."""
import ast, copy, json, io, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import torch
import test_wp3b_m3 as reference

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/'notebooks/wp3b_bounded_cached.ipynb'

def namespace():
    b=json.loads(NOTEBOOK.read_text());nodes=[];values={}
    for c in b['cells']:
        if c['cell_type']!='code':continue
        for n in ast.parse(''.join(c['source'])).body:
            if isinstance(n,(ast.FunctionDef,ast.ClassDef)):nodes.append(n)
            elif isinstance(n,ast.Assign):
                try:v=ast.literal_eval(n.value)
                except (ValueError,TypeError):continue
                for t in n.targets:
                    if isinstance(t,ast.Name):values[t.id]=v
    ns={**vars(reference.reference),**values,'ROOT':ROOT,'__name__':__name__}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<visible M4>','exec'),ns)
    return ns

class BoundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.ns=namespace()
    def setUp(self):
        torch.manual_seed(42);self.x=torch.randn(4,768)
        self.model=self.ns['ResidualVisualAdapter'](.2)
    def test_exact_identity_at_initialization(self):
        torch.testing.assert_close(self.model(self.x),self.x,rtol=0,atol=0)
        self.assertEqual(sum(p.numel() for p in self.model.parameters()),99136)
    def test_hard_norm_and_angular_bound_after_large_updates(self):
        with torch.no_grad():self.model.up.weight.normal_(std=50);self.model.up.bias.fill_(30)
        y=self.model(self.x);ratio=torch.linalg.vector_norm(y-self.x,dim=1)/torch.linalg.vector_norm(self.x,dim=1)
        self.assertTrue((ratio<=.200001).all())
        cosine=torch.nn.functional.cosine_similarity(y,self.x)
        self.assertTrue((cosine>=(1-.2**2)**.5-1e-6).all())
        self.assertTrue((torch.linalg.vector_norm(y,dim=1)>0).all())
    def test_invalid_caps_rejected(self):
        for v in [0,-1,1,True,float('nan'),.5]:
            with self.assertRaises(ValueError):self.ns['ResidualVisualAdapter'](v)
    def test_unbounded_control_matches_original_M3(self):
        old=reference.namespace()['ResidualVisualAdapter']();new=self.ns['ResidualVisualAdapter'](None)
        new.load_state_dict(old.state_dict());torch.testing.assert_close(new(self.x),old(self.x),rtol=0,atol=0)
        with torch.no_grad():old.up.weight.normal_(std=.1)
        new.load_state_dict(old.state_dict());torch.testing.assert_close(new(self.x),old(self.x),rtol=0,atol=0)
    def test_finite_gradient_at_zero_and_saturation(self):
        optimizer=torch.optim.AdamW(self.model.parameters(),lr=.001)
        for i in range(2):
            optimizer.zero_grad();y=self.model(self.x.requires_grad_(True));loss=y[:,0].mean();loss.backward()
            self.assertIsNone(self.x.grad)
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.parameters()))
            optimizer.step()
        self.assertGreater(self.model.down.weight.grad.abs().sum().item(),0.)
    def test_cached_backbone_is_frozen_and_checks_shape(self):
        model=self.ns['CachedFeatureBackbone']().eval();self.ns['require_frozen_clip'](model)
        torch.testing.assert_close(model.encode_image(self.x),self.x,rtol=0,atol=0)
        self.assertEqual(list(model.parameters()),[])
        for x in [torch.ones(2,4),torch.zeros(2,768),torch.full((2,768),float('nan'))]:
            with self.assertRaises(ValueError):model.encode_image(x)

class CacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns=namespace();cls.split=cls.ns['build_development_split'](ROOT/'DPD_Small_train.csv');cls.table=cls.ns['build_prompt_table'](cls.split)
        cls.config=copy.deepcopy(cls.ns['CONFIG']);cls.cache=cls.synthetic_cache()
    @classmethod
    def synthetic_cache(cls):
        g=torch.Generator().manual_seed(1);folds={}
        for role in ['fit','val_seen','val_unseen']:
            f=getattr(cls.split,role);x=torch.randn(len(f),768,generator=g)
            folds[role]={'features':x,'features_sha256':cls.ns['tensor_sha256'](x),'filenames':f.image_name.tolist(),
                'labels':torch.tensor(f[['crop_id','disease_id','composition_id']].to_numpy()),
                'labels_sha256':cls.ns['tensor_sha256'](torch.tensor(f[['crop_id','disease_id','composition_id']].to_numpy()))}
        direct=cls.ns['unit_features'](torch.randn(323,768,generator=g))
        spec=cls.ns['expected_cache_spec'](cls.split,cls.table,cls.config)
        cache={'schema':'wp3b-native-clip-cache-v1','spec':spec,'spec_sha256':cls.ns['json_hash'](spec),
               'folds':folds,'direct':direct,'direct_sha256':cls.ns['tensor_sha256'](direct),
               'image_hashes':{n:'0'*64 for n in cls.split.fit.image_name.tolist()+cls.split.val_seen.image_name.tolist()+cls.split.val_unseen.image_name.tolist()},
               'model_provenance':{'model':'ViT-L/14','revision':cls.ns['CLIP_REVISION'],
                    'artifact':{'sha256':cls.ns['MODEL_REGISTRY']['ViT-L/14']['sha256']},
                    'native_preprocess':cls.ns['native_preprocess_repr'](), 'trainable_parameters':0}}
        cache['image_hashes_sha256']=cls.ns['json_hash'](cache['image_hashes'])
        return cache
    def test_cache_identity_and_full_ordered_mapping(self):
        self.assertTrue(self.ns['validate_feature_cache'](self.cache,self.split,self.table,self.config))
        self.assertEqual([len(self.cache['folds'][k]['features']) for k in ['fit','val_seen','val_unseen']],[4868,1238,1894])
    def test_cache_reordered_filename_and_label_rejected(self):
        for key in ['filenames','labels']:
            bad=copy.deepcopy(self.cache)
            if key=='filenames':bad['folds']['fit'][key][:2]=reversed(bad['folds']['fit'][key][:2])
            else:bad['folds']['fit'][key][0,2]+=1
            with self.assertRaises(ValueError):self.ns['validate_feature_cache'](bad,self.split,self.table,self.config)
    def test_cache_tampered_features_and_hash_rejected(self):
        bad=copy.deepcopy(self.cache);bad['folds']['val_unseen']['features'][0,0]+=.01
        with self.assertRaises(ValueError):self.ns['validate_feature_cache'](bad,self.split,self.table,self.config)
    def test_cache_changed_model_prompt_preprocessing_rejected(self):
        for field in ['model_sha256','prompt_sha256','preprocessing']:
            bad=copy.deepcopy(self.cache);bad['spec'][field]='bad'
            bad['spec_sha256']=self.ns['json_hash'](bad['spec'])
            with self.assertRaises(ValueError):self.ns['validate_feature_cache'](bad,self.split,self.table,self.config)
    def test_cache_unexpected_fold_and_official_filename_rejected(self):
        bad=copy.deepcopy(self.cache);bad['folds']['official_test']=bad['folds']['val_seen']
        with self.assertRaises(ValueError):self.ns['validate_feature_cache'](bad,self.split,self.table,self.config)
        bad=copy.deepcopy(self.cache);bad['image_hashes']['official.jpg']='0'*64
        with self.assertRaises(ValueError):self.ns['validate_feature_cache'](bad,self.split,self.table,self.config)
    def test_dataset_preserves_ids_and_fitting_only_loader(self):
        ds=self.ns['CachedFeatureDataset'](self.split,'fit',self.cache)
        x,t=ds[0];self.assertEqual(tuple(x.shape),(768,));self.assertEqual(t['composition'],int(self.split.fit.composition_id.iloc[0]))
        self.assertEqual(ds.role,'fit')
        with self.assertRaises(ValueError):self.ns['CachedFeatureDataset'](self.split,'test',self.cache)
    def test_cache_safe_roundtrip_and_no_overwrite(self):
        with reference.reference.workspace_temporary_directory() as directory:
            path=Path(directory)/'features.pt';self.ns['save_feature_cache'](path,self.cache)
            restored=self.ns['load_feature_cache'](path,self.ns['sha256_file'](path),self.split,self.table,self.config)
            torch.testing.assert_close(restored['direct'],self.cache['direct'],rtol=0,atol=0)
            with self.assertRaises(FileExistsError):self.ns['save_feature_cache'](path,self.cache)
            with self.assertRaises(ValueError):self.ns['load_feature_cache'](path,'0'*64,self.split,self.table,self.config)
    def test_cache_training_matches_live_synthetic_encoding(self):
        ns=self.ns;x=torch.randn(3,768);a=ns['ResidualVisualAdapter'](.2);b=ns['ResidualVisualAdapter'](.2);b.load_state_dict(a.state_dict())
        direct=ns['unit_features'](torch.randn(323,768));cfg={**self.config,'feature_cache_path':str(Path(self.config['output_directory'])/'synthetic.pt'),'feature_cache_sha256':'0'*64}
        contract=ns['make_m4_contract'](self.split,self.table,cfg)
        proto=ns['component_prototypes'](direct,self.table,contract['fitting_composition_ids']);state=ns['seal_text_state'](direct,proto,contract)
        f=self.split.fit.iloc[:3];c=torch.tensor(f.crop_id.to_numpy());d=torch.tensor(f.disease_id.to_numpy());p=torch.tensor(f.composition_id.to_numpy())
        cached=ns['CachedFeatureBackbone']().eval().encode_image(x)
        first=ns['m3_training_objective'](a(x),x,c,d,p,state,1.)['loss']
        second=ns['m3_training_objective'](b(cached),cached,c,d,p,state,1.)['loss']
        torch.testing.assert_close(first,second,rtol=0,atol=0)
        first.backward();second.backward()
        for pa,pb in zip(a.parameters(),b.parameters()):torch.testing.assert_close(pa.grad,pb.grad,rtol=0,atol=0)
    def test_bound_and_cache_are_in_checkpoint_contract(self):
        cfg={**self.config,'feature_cache_path':str(Path(self.config['output_directory'])/'synthetic.pt'),'feature_cache_sha256':'a'*64}
        a=self.ns['make_m4_contract'](self.split,self.table,cfg)
        b=self.ns['make_m4_contract'](self.split,self.table,{**cfg,'residual_cap':None})
        self.assertNotEqual(self.ns['json_hash'](a),self.ns['json_hash'](b))
        self.assertEqual(a['adapter']['residual_cap'],.2)
        self.assertEqual(a['hyperparameters']['feature_cache_sha256'],'a'*64)
        for k,v in [('conditional_weight',0.),('residual_cap',.5),('epochs',6)]:
            with self.assertRaises(ValueError):self.ns['make_m4_contract'](self.split,self.table,{**cfg,k:v})
    def test_checkpoint_reconstructs_cap_and_rejects_mismatch(self):
        cfg={**self.config,'feature_cache_path':str(Path(self.config['output_directory'])/'synthetic.pt'),'feature_cache_sha256':'a'*64}
        c=self.ns['make_m4_contract'](self.split,self.table,cfg)
        proto=self.ns['component_prototypes'](self.cache['direct'],self.table,c['fitting_composition_ids']);state=self.ns['seal_text_state'](self.cache['direct'],proto,c)
        adapter=self.ns['ResidualVisualAdapter'](.2)
        with reference.reference.workspace_temporary_directory() as directory:
            path=Path(directory)/'adapter.pt';self.ns['save_adapter_checkpoint'](path,adapter,state,c,{'epoch':0})
            restored,_,_=self.ns['load_adapter_checkpoint'](path,c)
            self.assertEqual(restored.residual_cap,.2)
            bad=copy.deepcopy(c);bad['hyperparameters']['residual_cap']=None
            with self.assertRaises(ValueError):self.ns['load_adapter_checkpoint'](path,bad)

    def test_cache_producer_keeps_normal_raw_features_and_only_requested_names(self):
        from types import SimpleNamespace
        tiny=SimpleNamespace(fit=self.split.fit.iloc[:2],val_seen=self.split.val_seen.iloc[:1],val_unseen=self.split.val_unseen.iloc[:1])
        class TinyDataset(torch.utils.data.Dataset):
            def __init__(self,frame,*args):self.frame=frame.reset_index(drop=True)
            def __len__(self):return len(self.frame)
            def __getitem__(self,i):
                row=self.frame.iloc[i]
                return torch.ones(3,4,4)*(i+1),{'image_name':row.image_name}
        class Fake(torch.nn.Module):
            def encode_image(self,images):
                assert not torch.is_grad_enabled()
                return images.mean((1,2,3))[:,None]+torch.arange(768)[None,:]
        requested=[]
        def digest(path):requested.append(Path(path).name);return '0'*64
        with patch.dict(self.ns,{'CLIPDevelopmentDataset':TinyDataset,
                    'expected_cache_spec':lambda *a:self.cache['spec'],
                    'validate_feature_cache':lambda *a:True,'sha256_file':digest}):
            cache=self.ns['build_feature_cache'](Fake().eval(),object(),self.cache['model_provenance'],tiny,self.table,{**self.config,'device':'cpu'},self.cache['direct'],False)
        self.assertEqual(len(requested),4)
        self.assertEqual(set(requested),set(tiny.fit.image_name)|set(tiny.val_seen.image_name)|set(tiny.val_unseen.image_name))
        for part in cache['folds'].values():
            self.assertFalse(torch.is_inference(part['features']))
            self.assertFalse(part['features'].requires_grad)
            self.assertEqual(part['features'].dtype,torch.float32)

    def test_cached_fitting_epoch_has_two_steps_and_no_backbone_parameters(self):
        ns=self.ns;frame=self.split.fit.iloc[:3].copy();features=self.cache['folds']['fit']['features'][:3]
        class Tiny(torch.utils.data.Dataset):
            role='fit'
            def __len__(self):return 3
            def __getitem__(self,i):
                r=frame.iloc[i]
                return features[i],{'crop':int(r.crop_id),'disease':int(r.disease_id),
                    'composition':int(r.composition_id),'image_name':r.image_name}
        cfg={**self.config,'feature_cache_path':str(Path(self.config['output_directory'])/'synthetic.pt'),'feature_cache_sha256':'a'*64}
        contract=ns['make_m4_contract'](self.split,self.table,cfg)
        proto=ns['component_prototypes'](self.cache['direct'],self.table,contract['fitting_composition_ids'])
        state=ns['seal_text_state'](self.cache['direct'],proto,contract)
        model=ns['CachedFeatureBackbone']().eval();adapter=ns['ResidualVisualAdapter'](.2)
        optimizer=ns['make_adapter_optimizer'](adapter,cfg)
        original=adapter.up.weight.clone()
        with patch.object(optimizer,'step',wraps=optimizer.step) as step:
            result=ns['train_adapter_epoch'](model,adapter,torch.utils.data.DataLoader(Tiny(),batch_size=2),optimizer,state,frame,'cpu',False)
        self.assertEqual(step.call_count,2);self.assertEqual(result['sample_count'],3)
        self.assertFalse(torch.equal(original,adapter.up.weight))
        self.assertEqual(list(model.parameters()),[])

    def test_frozen_reference_parity_guard_is_exact(self):
        cache=copy.deepcopy(self.cache)
        rows=[]
        for role in ['val_seen','val_unseen']:
            part=cache['folds'][role]
            part['features']=part['features'][:2];part['filenames']=part['filenames'][:2]
            ranks=self.ns['rank_global_ids'](self.ns['cosine_candidate_scores'](part['features'],cache['direct']),self.split.candidate_ids)
            for i,name in enumerate(part['filenames']):
                rows.append({'epoch':0,'fold':role,'image_name':name,**{'control_rank_'+str(k+1):int(v) for k,v in enumerate(ranks[i,:5])}})
        table=self.ns['pd'].DataFrame(rows)
        with patch.object(self.ns['pd'],'read_csv',return_value=table):
            self.assertTrue(self.ns['verify_cached_frozen_reference'](cache,self.split))
        table.loc[0,'control_rank_1']=int(self.split.candidate_ids[0]) if table.loc[0,'control_rank_1']!=self.split.candidate_ids[0] else self.split.candidate_ids[-1]
        with patch.object(self.ns['pd'],'read_csv',return_value=table):
            with self.assertRaises(ValueError):self.ns['verify_cached_frozen_reference'](cache,self.split)

    def test_paired_cache_and_cap_budget_gate(self):
        def summary(cap,seen,unseen,diseased,false_healthy):
            return {'protocol':{'ids':list(self.split.candidate_ids)},
                'config':{**self.config,'residual_cap':cap,'feature_cache_path':'same.pt','feature_cache_sha256':'a'*64},
                'history':[{'epoch':i,'train_sample_count':None if i==0 else 4868} for i in range(6)],
                'selected':{'seen_top1':seen/1238,'unseen_top1':unseen/1894,'h1':self.ns['harmonic_mean'](seen/1238,unseen/1894)},
                'diagnostics':{'val_seen':{'count':1238,'correct1':seen},'val_unseen':{'count':1894,'correct1':unseen,'diseased_correct':diseased,'false_healthy':false_healthy}}}
        control=summary(None,224,233,71,873);new=summary(.2,300,260,100,300)
        self.assertTrue(self.ns['compare_pilot_summaries'](control,new)['supports_exploratory_success'])
        bad=copy.deepcopy(new);bad['config']['feature_cache_sha256']='b'*64
        with self.assertRaises(ValueError):self.ns['compare_pilot_summaries'](control,bad)
        bad=copy.deepcopy(new);bad['history'][1]['train_sample_count']=12
        with self.assertRaises(ValueError):self.ns['compare_pilot_summaries'](control,bad)
        unsafe=summary(.2,400,260,100,600)
        self.assertFalse(self.ns['compare_pilot_summaries'](control,unsafe)['supports_exploratory_success'])

    def test_one_shot_runner_declares_exactly_two_fresh_identical_budget_arms(self):
        ns=self.ns;calls=[]
        def baseline(split,model,adapter,preprocess,state,config):
            result={}
            for role,n in [('val_seen',1238),('val_unseen',1894)]:
                result[role]={'sample_count':n,'loss':1.,'metrics':{'top1':0.,'top5':0.},
                    'ranking':torch.tensor([[10,20]]),'control_ranking':torch.tensor([[10,20]])}
            return result
        def pilot(split,model,adapter,preprocess,state,config,contract,directory,provenance,record,validation,checkpoint):
            calls.append({'cap':adapter.residual_cap,'config':dict(config),'params':[v.detach().clone() for v in adapter.parameters()]})
            return {'best_record':record,'history':[{'epoch':i,'train_sample_count':None if i==0 else 4868} for i in range(6)]}
        def export(directory,contract,pilot,split):
            summary={'selected':pilot['best_record'],'history':pilot['history'],'config':contract['hyperparameters'],
                'protocol':{'candidate_ids':contract['candidate_ids']},
                'diagnostics':{'val_seen':{'count':1238,'correct1':0},'val_unseen':{'count':1894,'correct1':0,'diseased_correct':0,'false_healthy':0}}}
            path=Path(directory)/'adapter_summary.json';path.write_text(json.dumps(summary));return [str(path)]
        with reference.reference.workspace_temporary_directory() as directory:
            cache_path=Path(directory)/'features.pt';ns['save_feature_cache'](cache_path,self.cache)
            cfg={**self.config,'device':'cpu','output_directory':directory,
                 'feature_cache_path':str(cache_path),'feature_cache_sha256':ns['sha256_file'](cache_path)}
            with patch.dict(ns,{'FEATURE_CACHE':self.cache,'verify_prepared_state':lambda *a:True,
                    'evaluate_m4_folds':baseline,'run_adapter_pilot':pilot,'export_m4_results':export,
                    'save_adapter_checkpoint':lambda path,*a:str(path)}):
                result=ns['run_paired_cached_pilot'](self.split,self.table,cfg)
        self.assertEqual([c['cap'] for c in calls],[None,.2])
        self.assertTrue(all(c['config']['epochs']==5 and c['config']['conditional_weight']==1 for c in calls))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(calls[0]['params'],calls[1]['params'])))
        self.assertEqual(set(result['arms']),{'control','bounded'})

class StructureTests(unittest.TestCase):
    def test_json_syntax_and_default_off(self):
        b=json.loads(NOTEBOOK.read_text());self.assertEqual(len(b['cells']),len({c['id'] for c in b['cells']}))
        ns=namespace()
        for flag in ['BUILD_FEATURE_CACHE','LOAD_FEATURE_CACHE','PREDECLARE_RUN','RUN_TRAINING','RUN_DEVELOPMENT_EVALUATION','SAVE_CHECKPOINTS','EXPORT_RESULTS','VERIFY_CHECKPOINT','RUN_RECORD_DIAGNOSTICS','RUN_COMPARISON_REPORT','RUN_PAIRED_PILOT']:self.assertIs(ns[flag],False)
        for c in b['cells']:
            self.assertFalse(c.get('outputs'));self.assertIsNone(c.get('execution_count'))
            if c['cell_type']=='code':compile(''.join(c['source']),c['id'],'exec')
    def test_cold_default_run_all_has_no_image_model_or_writes(self):
        b=json.loads(NOTEBOOK.read_text());ns={'__name__':__name__,'display':lambda *a,**k:None}
        with patch('clip.load',side_effect=AssertionError('no model')),patch('PIL.Image.open',side_effect=AssertionError('no image')),patch('torch.save',side_effect=AssertionError('no save')),patch.object(Path,'write_text',side_effect=AssertionError('no write')),reference.reference.redirect_stdout(io.StringIO()):
            for c in b['cells']:
                if c['cell_type']=='code':exec(compile(''.join(c['source']),c['id'],'exec'),ns)
        self.assertIsNone(ns['FEATURE_CACHE']);self.assertIsNone(ns['FROZEN_CLIP']);self.assertIsNone(ns['RUN_DIRECTORY'])
    def test_no_hidden_processing_or_test_image_access(self):
        text=NOTEBOOK.read_text();self.assertNotIn('DPD_Small_test_',text);self.assertNotIn('%run',text)
        for c in json.loads(text)['cells']:
            if c['cell_type']=='code':
                for n in ast.walk(ast.parse(''.join(c['source']))):
                    if isinstance(n,ast.ImportFrom):self.assertNotIn((n.module or '').split('.')[0],['dataset','development_data','train_multitask','evaluation'])
                    if isinstance(n,ast.Call):self.assertNotIn(ast.unparse(n.func),['exec','eval','compile','importlib.import_module'])

    def test_paired_and_individual_execution_cannot_mix(self):
        book=json.loads(NOTEBOOK.read_text());source=next(''.join(c['source']) for c in book['cells'] if c['id']=='m4-owner-paired-run')
        ns={'RUN_PAIRED_PILOT':True,'PREDECLARE_RUN':True,'RUN_TRAINING':False,'RUN_DIRECTORY':None,'PAIRED_RESULTS':None}
        with self.assertRaises(RuntimeError):exec(compile(source,'paired-owner-cell','exec'),ns)
        ns.update(PREDECLARE_RUN=False,PAIRED_STARTED=True)
        with self.assertRaises(RuntimeError):exec(compile(source,'paired-owner-cell','exec'),ns)

if __name__=='__main__':unittest.main()
