"""Explicitly synthetic added-class GUI exercise; no model inference or accuracy."""
import argparse
import ctypes
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


class SyntheticAddedCandidatePredictor:
    """Deterministic test scores that place a declared added candidate first."""
    def __init__(self,catalog,candidate_id):
        self.catalog,self.candidate_id=catalog,candidate_id
        self.provenance={'variant':'synthetic_added_candidate_fixture','synthetic':True,
                         'candidate_id':candidate_id,'native_model_inference':False}

    def predict(self,images,topk=5):
        import torch
        if len(images)!=1 or images[0].mode!='RGB':
            raise ValueError('Synthetic fixture expects one RGB image')
        ids=torch.tensor(list(self.catalog.candidates),dtype=torch.int64)
        scores=-torch.arange(len(ids),dtype=torch.float64)[None,:]
        scores[0,list(self.catalog.candidates).index(self.candidate_id)]=1.
        order=scores.argsort(dim=1,descending=True,stable=True)[:,:topk]
        return {'candidate_ids':ids,'scores':scores,'topk_composition_ids':ids[order],
                'candidate_sha256':self.catalog.protocol['candidate_sha256'],
                'score_semantics':'synthetic_fixture_no_model_inference',
                'model_provenance':self.provenance}


def rehearse(protocol,source,images,output,query='PWV2_peach_leaf_curl_6128.jpg',candidate_id=39,screenshots=False):
    import tkinter as tk
    from PIL import ImageGrab
    from wp4.gui import PlantApp
    output.mkdir(parents=True,exist_ok=False)
    root=tk.Tk()
    if not screenshots:
        root.withdraw()
    app=PlantApp(root,protocol,images,source)
    errors=[]
    root.report_callback_exception=lambda *exc:errors.append(str(exc))

    def wait(condition):
        deadline=time.monotonic()+90
        while time.monotonic()<deadline:
            root.update()
            if errors:
                raise RuntimeError(str(errors))
            if condition():
                root.update_idletasks()
                return
            time.sleep(.02)
        raise TimeoutError('Synthetic GUI exercise timed out')

    try:
        with patch('wp4.gui.messagebox.showerror',side_effect=lambda *a,**k:errors.append(str(a))):
            wait(lambda:app.catalog is not None and not app.busy)
            fitting_ids={row['composition_id'] for row in app.catalog.fitting}
            if not app.catalog.is_final or candidate_id not in app.catalog.candidates or candidate_id in fitting_ids:
                raise ValueError('Select an added final candidate absent from full training')
            if Path(query).name!=query or query not in app.catalog.protocol['training_images']:
                raise ValueError('Only an original training-source query is accepted')
            fixture=output/'SYNTHETIC_NOT_MODEL.pt'
            fixture.write_bytes(b'Synthetic path-validation fixture only; not serialized model weights.')
            predictor=SyntheticAddedCandidatePredictor(app.catalog,candidate_id)
            panel=app.model_panel
            panel.choice.current(panel.kinds.index('wp1'))
            panel.render_fields()
            for key in ('crop','disease'):
                panel.paths[key].set(str(fixture))
            with patch('wp4.gui.load_model',return_value=predictor):
                panel.load()
                wait(lambda:app.predictor is predictor and not app.busy)
            app.model_label='SYNTHETIC ADDED-CLASS FIXTURE / NO MODEL INFERENCE'
            app.model_status.configure(text=app.model_label)
            app.scope_caption='SYNTHETIC / UI CHECK'
            app.update_scope_caption(app.scope_caption)
            with patch('wp4.gui.filedialog.askopenfilename',return_value=str(images/query)):
                app.open_image()
            wait(lambda:app.query is not None and not app.busy)
            app.identify_button.invoke()
            wait(lambda:app.export_data is not None and not app.busy and app.reference_note.cget('text').startswith('Selected match.'))
            assert app.predictions[0]['composition_id']==candidate_id
            assert app.table.selection()==(str(candidate_id),)
            assert app.class_ids[app.class_choice.current()]==candidate_id
            assert 'Same disease; different crop' in app.reference_note.cget('text')
            refs=app.catalog.references(candidate_id,app.query,app.query_path)
            assert refs['items'] and len(app.photos)==len(refs['items'])
            by_name={row['image_name']:row for row in app.catalog.fitting}
            assert all(row['filename'] in by_name and by_name[row['filename']]['composition_id']!=candidate_id
                       and row['filename']!=query for row in refs['items'])
            app.result_note.configure(text='Synthetic scores only. No model inference or accuracy claim.')
            exported=output/'SYNTHETIC_added_candidate_prediction.json'
            with patch('wp4.gui.filedialog.asksaveasfilename',return_value=str(exported)):
                app.export()
            packet=json.loads(exported.read_text(encoding='utf-8'))
            assert packet==app.export_data and packet['model_provenance']['synthetic'] is True
            assert packet['predictions'][0]['composition_id']==candidate_id
            if screenshots:
                root.state('normal')
                root.geometry('1280x840+20+20')
                root.update()
                time.sleep(.2)
                ImageGrab.grab(window=int(root.wm_frame(),0)).save(output/'SYNTHETIC_added_candidate.png')
            receipt={'schema':'wp4-synthetic-added-candidate-v1','synthetic':True,
                     'native_model_inference':False,'official_images_read':False,
                     'added_candidate':candidate_id,'selected_top1':app.predictions[0],
                     'candidate_count':333,'training_reference_count':8000,'query':query,
                     'selected_result_path_exercised':True,'dropdown_alignment_verified':True,
                     'fallback_note':app.reference_note.cget('text'),
                     'reference_names':[row['filename'] for row in refs['items']],
                     'fitting_only':True,'export_roundtrip_verified':True,'errors':errors}
            (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
            return receipt
    finally:
        app.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'artifacts/data/final_full8000_333_v1.json')
    parser.add_argument('--source',type=Path,default=ROOT/'DPD_Small_train.csv')
    parser.add_argument('--images',type=Path,default=ROOT/'DPD_Small_dataset')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--screenshots',action='store_true')
    args=parser.parse_args()
    if sys.platform=='win32' and args.screenshots:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    print(json.dumps(rehearse(args.protocol,args.source,args.images,args.output,screenshots=args.screenshots),indent=2))


if __name__=='__main__':
    main()
