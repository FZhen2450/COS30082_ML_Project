
import ast,copy,json,unittest,tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
NOTEBOOK=Path(__file__).resolve().parents[1]/'notebooks/wp2_selected_development_export.ipynb'
class ExportTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.nb=json.loads(NOTEBOOK.read_text());cls.code='\n'.join(''.join(c['source']) for c in cls.nb['cells'] if c['cell_type']=='code');cls.ns={}
  exec(compile(cls.code,str(NOTEBOOK),'exec'),cls.ns)
 def test_notebook_guards_and_outputs(self):
  self.assertIs(self.ns['RUN_EXPORT'],False)
  for c in self.nb['cells']:
   if c['cell_type']=='code':self.assertEqual(c['outputs'],[]);self.assertIsNone(c['execution_count']);ast.parse(''.join(c['source']))
  self.assertNotIn('train_one_epoch',self.code);self.assertNotIn('.backward(',self.code);self.assertNotIn('optimizer.step',self.code)
 def fixture(self,n=5):
  rows=[{'image_name':f'image_{i}.jpg','crop_id':0,'disease_id':0,'composition_id':2} for i in range(n)]
  class Items(Dataset):
   def __len__(self):return n
   def __getitem__(self,i):return torch.full((3,2,2),float(i)),{'image_name':f'image_{i}.jpg','crop':0,'disease':0,'composition':2}
  class Reused(nn.Module):
   def __init__(self):super().__init__();self.register_buffer('c',torch.zeros(2,55));self.register_buffer('d',torch.zeros(2,175))
   def forward(self,x):
    count=len(x);self.c.zero_();self.d.zero_();self.c[:count,0]=x[:,0,0,0];self.d[:count,0]=x[:,0,0,0]
    return {'crop_logits':self.c[:count],'disease_logits':self.d[:count]}
  return rows,Items(),Reused()
 def test_reused_cpu_buffers_are_snapshotted_with_partial_batch(self):
  rows,ds,model=self.fixture();r=self.ns['collect_named_logits'](model,DataLoader(ds,batch_size=2),rows,'cpu',precision='fp32',channels_last=False)
  self.assertEqual(r['crop_logits'][:,0].tolist(),[0.,1.,2.,3.,4.]);self.assertEqual(r['disease_logits'][:,0].tolist(),[0.,1.,2.,3.,4.]);self.assertEqual(r['image_names'],[x['image_name'] for x in rows]);self.assertFalse(model.training)
 def test_same_label_reordered_names_rejected(self):
  rows,ds,model=self.fixture()
  with self.assertRaises(ValueError):self.ns['collect_named_logits'](model,DataLoader(ds,batch_size=2,sampler=list(reversed(range(5)))),rows,'cpu',precision='fp32',channels_last=False)
 def test_missing_or_empty_loader_rejected(self):
  rows,ds,model=self.fixture()
  for loader in [[],DataLoader(ds,batch_size=2,drop_last=True)]:
   with self.assertRaises(ValueError):self.ns['collect_named_logits'](model,loader,rows,'cpu',precision='fp32',channels_last=False)
 def test_nonfinite_output_rejected(self):
  rows,ds,model=self.fixture()
  def bad(x):return {'crop_logits':torch.full((len(x),55),float('nan')),'disease_logits':torch.zeros(len(x),175)}
  model.forward=bad
  with self.assertRaises(ValueError):self.ns['collect_named_logits'](model,DataLoader(ds,batch_size=2),rows,'cpu',precision='fp32',channels_last=False)
 def test_canonical_scorer_and_exact_ties(self):
  c=torch.zeros(2,55);d=torch.zeros(2,175);ids=[2,8,331];pairs=[[0,0],[1,1],[2,2]]
  scores=self.ns['score_components'](c,d,ids,pairs)
  expected=torch.nn.functional.log_softmax(c.double(),1)[:,[0,1,2]]+torch.nn.functional.log_softmax(d.double(),1)[:,[0,1,2]]
  self.assertTrue(torch.equal(scores,expected));self.assertEqual(self.ns['rank_global_ids'](scores,ids)[0].tolist(),ids)
  scores[0,2]=torch.nextafter(scores[0,0],torch.tensor(float('inf'),dtype=torch.float64));self.assertEqual(self.ns['rank_global_ids'](scores,ids)[0,0].item(),331)
 def test_extreme_logits_and_reordered_candidates(self):
  c=torch.full((1,55),-1e20);d=torch.full((1,175),-1e20);c[:,0]=1e20;d[:,0]=1e20
  self.assertTrue(torch.isfinite(self.ns['score_components'](c,d,[2,8],[[0,0],[1,1]])).all())
  with self.assertRaises(ValueError):self.ns['score_components'](c,d,[8,2],[[0,0],[1,1]])
 def test_original_source_mappings_keep_checkpoint_integer_keys(self):
  _,_,folds,mappings,ids,pairs=self.ns['load_inputs'](self.ns['HANDOFF_DIR'])
  self.assertEqual(set(mappings['crop_id_to_name']),set(range(55)))
  self.assertEqual(set(mappings['disease_id_to_name']),set(range(175)))
  self.assertEqual([len(folds[f]) for f in ['fitting','seen','unseen']],[4868,1238,1894])
 def test_legacy_resolution_and_mappings_fail_closed(self):
  mappings={'crop_id_to_name':{'0':'apple'},'disease_id_to_name':{'0':'healthy'}}
  payload={'format_version':1,'architecture':'multitask_convnext_base','num_crops':55,'num_diseases':175,'feature_dim':1024,'model_dtype':'torch.float32','class_mappings':mappings,'metadata':{'epoch':1,'selection':{'epoch':1,'resolution':224}},'preprocessing':{'image_size':[320,320]}}
  self.ns['validate_selected_metadata'](payload,mappings)
  for key,value in [('feature_dim',768),('num_crops',54),('architecture','multitask_convnext_tiny'),('model_dtype','torch.float16')]:
   bad=copy.deepcopy(payload);bad[key]=value
   with self.assertRaises(ValueError):self.ns['validate_selected_metadata'](bad,mappings)
  bad=copy.deepcopy(payload);bad['metadata']['selection']['resolution']=320
  with self.assertRaises(ValueError):self.ns['validate_selected_metadata'](bad,mappings)
  with self.assertRaises(ValueError):self.ns['validate_selected_metadata'](payload,{'crop_id_to_name':{'0':'pear'}})
 def test_unique_export_directory_never_overwrites(self):
  with tempfile.TemporaryDirectory() as t:
   path=Path(t)/'run';self.ns['create_output_directory'](path)
   with self.assertRaises(FileExistsError):self.ns['create_output_directory'](path)
 def test_guarded_export_serializes_packets_accepted_by_wp5_using_synthetic_images(self):
  import sys,io,contextlib
  ns=self.ns
  class SyntheticImages(Dataset):
   def __init__(self,rows,image_directory):self.rows=rows
   def __len__(self):return len(self.rows)
   def __getitem__(self,i):
    r=self.rows[i]
    return torch.zeros(3,2,2),{'image_name':r['image_name'],'crop':r['crop_id'],'disease':r['disease_id'],'composition':r['composition_id']}
  class SyntheticModel(nn.Module):
   def forward(self,x):return {'crop_logits':torch.zeros(len(x),55),'disease_logits':torch.zeros(len(x),175)}
  info={'metadata':{'run_id':'SYNTHETIC_TEST_ONLY','code':{'synthetic_fixture':True},'selection':{'epoch':1,'resolution':224}},'preprocessing':{'image_size':[320,320]}}
  guarded=next(''.join(c['source']) for c in self.nb['cells'] if c['cell_type']=='code' and ''.join(c['source']).startswith('if RUN_EXPORT:'))
  reference=ns['ROOT']/'workspace/handoffs/TEAM_WP5_HANDOFF_20261107/wp5_reference'
  # Isolate the pinned WP5 package from the older merged evaluation package
  # already imported by other tests; restore both namespace and path afterward.
  with patch.object(sys,'path',[str(reference),*sys.path]),patch.dict(sys.modules):
   for name in list(sys.modules):
    if name=='evaluation' or name.startswith('evaluation.'):sys.modules.pop(name)
   from evaluation.reporting import EvaluationManifest,RunProvenance,build_report
   from evaluation.model_adapters import ModelOutputAdapter
   with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
    changes={'RUN_EXPORT':True,'DEVICE':'cpu','PRECISION':'fp32','CHANNELS_LAST':False,'BATCH_SIZE':64,'OUTPUT_PARENT':Path(temp),'NamedDevelopmentImages':SyntheticImages,'load_selected_checkpoint':lambda *a:(SyntheticModel(),info)}
    with patch.dict(ns,changes):exec(compile(guarded,'synthetic_guarded_export','exec'),ns)
    directory=next(Path(temp).iterdir());receipt=json.loads((directory/'completion.json').read_text())
    self.assertEqual(receipt['counts'],{'seen':1238,'unseen':1894});self.assertFalse(receipt['training_performed']);self.assertFalse(receipt['official_tests_used'])
    manifest=EvaluationManifest.from_dict(json.loads((ns['HANDOFF_DIR']/'identity_manifest.json').read_text()))
    prov=RunProvenance(**receipt['model_provenance']);adapter=ModelOutputAdapter(manifest,prov,'wp2_convnext_base');records=[]
    for fold in ['seen','unseen']:
     packet=json.loads((directory/(fold+'_packet.json')).read_text());records.extend(adapter.adapt(packet))
     self.assertEqual(len(packet['independent_crop_ids']),receipt['counts'][fold])
    report=build_report(manifest,records);self.assertEqual(report['seen']['count'],1238);self.assertEqual(report['unseen']['count'],1894)
 def test_no_project_processing_import_or_hidden_execution(self):
  for node in ast.walk(ast.parse(self.code)):
   if isinstance(node,ast.ImportFrom):self.assertNotIn((node.module or '').split('.')[0],['dataset','data_utils','train_multitask','development_data','multitask_backbones','evaluation'])
   if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):self.assertNotIn(node.func.id,['exec','eval','__import__'])
if __name__=='__main__':unittest.main(verbosity=2)
