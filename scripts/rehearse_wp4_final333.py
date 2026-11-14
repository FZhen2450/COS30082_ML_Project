"""Rehearse the final-333 GUI on one declared training image, never official data."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=('wp1','frozen_clip','m6'),required=True)
    parser.add_argument('--protocol',type=Path,default=ROOT/'artifacts/data/final_full8000_333_v1.json')
    parser.add_argument('--images',type=Path,default=ROOT/'DPD_Small_dataset')
    parser.add_argument('--query',default='PWV2_peach_leaf_curl_6128.jpg')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--screenshots',action='store_true')
    args=parser.parse_args()
    import torch
    import tkinter as tk
    from PIL import ImageGrab
    from wp4.gui import PlantApp
    from wp4.model_loader import MODEL_SPECS
    from wp4.service import present_prediction
    torch.set_num_threads(2)
    if sys.platform=='win32':
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    args.output.mkdir(parents=True,exist_ok=False)
    root=tk.Tk()
    app=PlantApp(root,args.protocol,args.images,ROOT/'DPD_Small_train.csv')
    failures=[]
    root.report_callback_exception=lambda *exc:failures.append(str(exc))

    def wait(condition):
        deadline=time.monotonic()+600
        while time.monotonic()<deadline:
            root.update()
            if failures:
                raise RuntimeError(str(failures))
            if condition():
                root.update_idletasks()
                return
            time.sleep(.04)
        raise TimeoutError('GUI operation timed out')

    def screenshot(name):
        if args.screenshots:
            root.update()
            time.sleep(.2)
            ImageGrab.grab(window=int(root.wm_frame(),0)).save(args.output/name)

    try:
        with patch('wp4.gui.messagebox.showerror',side_effect=lambda *a,**k:failures.append(str(a))):
            wait(lambda:app.catalog is not None and not app.busy)
            assert app.catalog.is_final and len(app.catalog.candidates)==333 and len(app.catalog.fitting)==8000
            if Path(args.query).name!=args.query or args.query not in app.catalog.protocol['training_images']:
                raise ValueError('Only an original training-source query is accepted')
            panel=app.model_panel
            panel.choice.current(panel.kinds.index(args.model))
            panel.render_fields()
            wait(lambda:all(panel.paths[key].get() for key,_ in MODEL_SPECS[args.model]['fields']))
            panel.device.set('cpu')
            panel.load()
            wait(lambda:app.predictor is not None and not app.busy)
            with patch('wp4.gui.filedialog.askopenfilename',return_value=str(args.images/args.query)):
                app.open_image()
            wait(lambda:app.query is not None and not app.busy)
            app.identify_button.invoke()
            wait(lambda:app.export_data is not None and not app.busy and app.reference_note.cget('text').startswith('Selected match.'))
            canonical=app.predictor.predict([app.query])
            assert tuple(canonical['scores'].shape)==(1,333)
            assert canonical['candidate_ids'].tolist()==list(range(333))
            assert torch.isfinite(canonical['scores']).all()
            order=canonical['scores'].argsort(dim=1,descending=True,stable=True)[:,:5]
            assert canonical['topk_composition_ids'].tolist()==canonical['candidate_ids'][order].tolist()
            assert present_prediction(canonical,app.catalog)==app.predictions
            (args.output/'canonical_scores.json').write_text(json.dumps({
                'candidate_ids':canonical['candidate_ids'].tolist(),
                'scores':canonical['scores'].tolist(),
                'topk_composition_ids':canonical['topk_composition_ids'].tolist(),
                'score_dtype':str(canonical['scores'].dtype),
                'candidate_sha256':canonical['candidate_sha256']},allow_nan=False)+'\n',encoding='utf-8')
            refs=app.catalog.references(app.predictions[0]['composition_id'],app.query,app.query_path)
            training=set(app.catalog.protocol['training_images'])
            assert all(row['filename'] in training and row['filename']!=args.query for row in refs['items'])
            export=args.output/'prediction.json'
            with patch('wp4.gui.filedialog.asksaveasfilename',return_value=str(export)):
                app.export()
            assert json.loads(export.read_text())==app.export_data
            screenshot('final333-'+args.model+'.png')
            root.state('normal')
            root.geometry('1280x840+20+20')
            root.update()
            screenshot('minimum-final333-'+args.model+'.png')
            fitting_ids={row['composition_id'] for row in app.catalog.fitting}
            unseen=next(cid for cid in app.class_ids if cid not in fitting_ids)
            app.class_choice.current(app.class_ids.index(unseen))
            app.browse_class()
            wait(lambda:not app.busy and app.reference_note.cget('text').startswith('Class browser'))
            assert 'Same disease; different crop' in app.reference_note.cget('text')
            screenshot('unseen-labelled-fallback.png')
            receipt={'schema':'wp4-final333-rehearsal-v1','model':args.model,'query':args.query,
                     'candidate_count':333,'reference_training_count':8000,'native_image_inference':True,
                     'canonical_outputs_match':True,'export_matches':True,'model_panel_inline':True,
                     'model_label':app.model_label,'score_shape':list(canonical['scores'].shape),
                     'score_dtype':str(canonical['scores'].dtype),'stable_top5_verified':True,
                     'reference_names':[row['filename'] for row in refs['items']],
                     'query_sha256':hashlib.sha256((args.images/args.query).read_bytes()).hexdigest(),
                     'source_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [*sorted((ROOT/'wp4').glob('*.py')),Path(__file__)]},
                     'official_test_images_read':False,'model_provenance':app.export_data['model_provenance'],
                     'fallback_candidate':unseen,'fallback_note':app.reference_note.cget('text'),
                     'predictions':app.predictions,'errors':failures}
            (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
            print(json.dumps(receipt,indent=2),flush=True)
    finally:
        app.close()


if __name__=='__main__':
    main()
