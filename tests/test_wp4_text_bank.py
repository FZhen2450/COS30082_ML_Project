"""Official-bank tests use explicit NPZ fixtures, never model/network downloads."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from wp4.text_bank import load_official_text_bank


class OfficialTextBankTests(unittest.TestCase):
    def fixture(self,path,features,ids,extra=False):
        values={'features':features,'candidate_ids':ids}
        if extra:
            values['unexpected']=np.array([1])
        np.savez(path,**values)
        return (hashlib.sha256(path.read_bytes()).hexdigest(),
                hashlib.sha256(np.ascontiguousarray(features).tobytes()).hexdigest())

    def test_exact_stored_tensor_is_returned_without_renormalization(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'official-fixture.npz'
            features=np.full((333,768),1/np.sqrt(768),dtype=np.float64)
            features[0,0]=np.nextafter(features[0,0],np.inf)
            file_sha,tensor_sha=self.fixture(path,features,np.arange(333,dtype=np.int64))
            with patch('wp4.text_bank.OFFICIAL_BANK_SHA256',file_sha),patch('wp4.text_bank.OFFICIAL_FEATURES_SHA256',tensor_sha):
                bank,identity=load_official_text_bank(path)
            self.assertTrue(np.array_equal(bank.numpy(),features))
            self.assertEqual(bank.dtype,torch.float64)
            self.assertEqual(identity['text_bank_file_sha256'],file_sha)
            self.assertEqual(identity['direct_text_sha256'],tensor_sha)

    def test_wrong_file_hash_schema_columns_precision_and_nonfinite_rows_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fixture.npz'
            for change in ('file_hash','extra','ids','float32','nan','norm'):
                features=np.full((333,768),1/np.sqrt(768),dtype=np.float64)
                ids=np.arange(333,dtype=np.int64)
                if change=='ids': ids=ids[::-1]
                if change=='float32': features=features.astype(np.float32)
                if change=='nan': features[0,0]=np.nan
                if change=='norm': features[0,:]=0
                file_sha,tensor_sha=self.fixture(path,features,ids,extra=change=='extra')
                with self.subTest(change=change),patch('wp4.text_bank.OFFICIAL_BANK_SHA256',
                        '0'*64 if change=='file_hash' else file_sha),patch('wp4.text_bank.OFFICIAL_FEATURES_SHA256',tensor_sha),self.assertRaises(ValueError):
                    load_official_text_bank(path)


if __name__=='__main__':
    unittest.main()
