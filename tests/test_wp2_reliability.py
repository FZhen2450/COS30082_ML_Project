"""CPU-only regressions for output retention and saved notebook provenance."""
import ast,hashlib,json,tempfile,unittest
from pathlib import Path
from contextlib import nullcontext
from unittest.mock import patch
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
ROOT=Path(__file__).resolve().parents[1]

def definitions(path):
    notebook=json.loads(path.read_text())
    nodes=[]
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            nodes.extend(n for n in ast.parse(''.join(cell['source'])).body if isinstance(n,ast.FunctionDef) and n.name in ['d_collect_logits','capture_notebook_source'])
    ns={'Path':Path,'ast':ast,'hashlib':hashlib,'json':json,'torch':torch,'nn':nn,'Dataset':Dataset,
        'DataLoader':DataLoader,'pd':pd,'amp_context':lambda device,precision:nullcontext()}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    return ns

class CpuSnapshotTests(unittest.TestCase):
    def test_collector_retains_reused_cpu_buffers_across_partial_batches(self):
        ns=definitions(ROOT/'notebooks/wp2_backbone_comparison.ipynb')
        ns['amp_context']=lambda device,precision:nullcontext()
        class Items(Dataset):
            def __len__(self):return 5
            def __getitem__(self,i):return torch.full((3,2,2),float(i)),{'crop':0,'disease':0,'composition':0}
        class Predictor(nn.Module):
            def __init__(self):
                super().__init__();self.register_buffer('crop',torch.zeros(2,55));self.register_buffer('disease',torch.zeros(2,175))
            def forward(self,x):
                n=len(x);self.crop.zero_();self.disease.zero_();self.crop[:n,0]=x[:,0,0,0];self.disease[:n,0]=x[:,0,0,0]
                return {'crop_logits':self.crop[:n],'disease_logits':self.disease[:n]}
        frame=pd.DataFrame({'crop_id':[0]*5,'disease_id':[0]*5,'composition_id':[0]*5})
        c,d=ns['d_collect_logits'](Predictor(),DataLoader(Items(),batch_size=2),frame,'cpu',precision='fp32',channels_last=False)
        self.assertEqual(c[:,0].tolist(),[0.,1.,2.,3.,4.]);self.assertEqual(d[:,0].tolist(),[0.,1.,2.,3.,4.])
        self.assertEqual(c.dtype,torch.float32);self.assertFalse(c.requires_grad)

class NotebookProvenanceTests(unittest.TestCase):
    def capture(self,path,directory):
        ns=definitions(ROOT/'notebooks/wp2_selected_development_export.ipynb')
        self.assertTrue(callable(ns.get('capture_notebook_source')),'Saved-source capture must be visible in the notebook')
        return ns['capture_notebook_source'](path,directory)
    def raw(self,code='value=1',output='first'):
        return json.dumps({'nbformat':4,'nbformat_minor':5,'metadata':{'note':output},'cells':[{'cell_type':'code','source':[code],'outputs':[{'text':output}],'execution_count':1}]}).encode()
    def test_source_capture_hashes_exact_same_bytes_written_despite_autosave(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);source=p/'input.ipynb';original=self.raw();source.write_bytes(original);out=p/'run';out.mkdir()
            real=Path.read_bytes
            def autosave(path):
                raw=real(path)
                if path==source:source.write_bytes(self.raw(output='autosaved'))
                return raw
            with patch.object(Path,'read_bytes',autosave):record=self.capture(source,out)
            saved=(out/record['source_snapshot']).read_bytes()
            self.assertEqual(saved,original)
            self.assertEqual(record['export_notebook_sha256'],hashlib.sha256(saved).hexdigest())
            self.assertNotEqual(source.read_bytes(),saved)
    def test_code_digest_ignores_outputs_but_detects_processing_changes(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);source=p/'input.ipynb';records=[]
            for i,(code,output) in enumerate([('value=1','first'),('value=1','changed output'),('value=2','changed output')]):
                source.write_bytes(self.raw(code,output));out=p/str(i);out.mkdir();records.append(self.capture(source,out))
            self.assertNotEqual(records[0]['export_notebook_sha256'],records[1]['export_notebook_sha256'])
            self.assertEqual(records[0]['export_notebook_ast_sha256'],records[1]['export_notebook_ast_sha256'])
            self.assertNotEqual(records[1]['export_notebook_ast_sha256'],records[2]['export_notebook_ast_sha256'])
    def test_source_snapshot_cannot_overwrite_existing_file(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);source=p/'input.ipynb';source.write_bytes(self.raw());out=p/'run';out.mkdir()
            record=self.capture(source,out);target=out/record['source_snapshot'];before=target.read_bytes()
            with self.assertRaises(FileExistsError):self.capture(source,out)
            self.assertEqual(target.read_bytes(),before)
if __name__=='__main__':unittest.main()
