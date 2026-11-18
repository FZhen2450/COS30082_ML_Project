"""Explicit synthetic result exercises added-candidate selection and export."""
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from scripts.rehearse_wp4_added_candidate import rehearse
from wp4.service import Catalog

ROOT=Path(__file__).resolve().parents[1]


class AddedCandidateGUITests(unittest.TestCase):
    def test_synthetic_prediction_selects_added_class_and_exports_labelled_fallback(self):
        import tkinter as tk
        try:
            probe=tk.Tk()
            probe.destroy()
        except tk.TclError as exc:
            self.skipTest('A local Tk display is required: '+str(exc))
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            protocol=ROOT/'artifacts/data/final_full8000_333_v1.json'
            source=ROOT/'DPD_Small_train.csv'
            catalog=Catalog(protocol,source,folder)
            query='PWV2_peach_leaf_curl_6128.jpg'
            Image.new('RGB',(20,18),(220,30,50)).save(folder/query)
            target=catalog.candidates[39]
            rows=[row for row in catalog.fitting if row['disease_id']==target['disease_id']][:4]
            for index,row in enumerate(rows):
                Image.new('RGB',(18,14),(20+index*20,130,60)).save(folder/row['image_name'])
            receipt=rehearse(protocol,source,folder,folder/'output')
            self.assertTrue(receipt['synthetic'])
            self.assertFalse(receipt['native_model_inference'])
            self.assertEqual(receipt['selected_top1']['composition_id'],39)
            self.assertTrue(receipt['selected_result_path_exercised'])
            self.assertTrue(receipt['export_roundtrip_verified'])
            self.assertEqual(len(receipt['reference_names']),4)


if __name__=='__main__':
    unittest.main()
