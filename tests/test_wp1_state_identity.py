"""Tensor-state identities must be explicit, stable and independent of metadata."""
from collections import OrderedDict
from pathlib import Path
import tempfile
import unittest

import torch

from checkpoint_identity import checkpoint_identity,state_dict_identity


class StateIdentityTests(unittest.TestCase):
    def test_key_order_and_serialization_do_not_change_state_identity(self):
        state=OrderedDict([('weight',torch.arange(6,dtype=torch.float32).reshape(2,3)),
                           ('bn.num_batches_tracked',torch.tensor(7,dtype=torch.int64))])
        expected=state_dict_identity(state)
        self.assertEqual(expected['state_dict_sha256'],state_dict_identity(OrderedDict(reversed(list(state.items()))))['state_dict_sha256'])
        self.assertEqual(expected['tensor_count'],2)
        with tempfile.TemporaryDirectory() as directory:
            first,second=Path(directory)/'first.pt',Path(directory)/'second.pt'
            torch.save({'state_dict':state,'metadata':{'note':'first'}},first)
            torch.save({'state_dict':state,'metadata':{'note':'different metadata'}},second)
            a,b=checkpoint_identity(first),checkpoint_identity(second)
            self.assertNotEqual(a['checkpoint_file_sha256'],b['checkpoint_file_sha256'])
            self.assertEqual(a['state_dict_sha256'],b['state_dict_sha256'])

    def test_name_dtype_shape_and_values_are_part_of_the_identity(self):
        base=torch.arange(6,dtype=torch.float32).reshape(2,3)
        expected=state_dict_identity({'weight':base})['state_dict_sha256']
        for state in ({'renamed':base},{'weight':base.double()},{'weight':base.reshape(3,2)},
                      {'weight':base+1}):
            self.assertNotEqual(state_dict_identity(state)['state_dict_sha256'],expected)

    def test_invalid_entries_and_wrong_file_pins_are_rejected(self):
        for state in ({},{'weight':'not a tensor'},{'weight':torch.tensor([float('nan')])}):
            with self.assertRaises(ValueError):
                state_dict_identity(state)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fixture.pt'
            torch.save({'state_dict':{'weight':torch.ones(2)}},path)
            with self.assertRaises(ValueError):
                checkpoint_identity(path,'0'*64)


if __name__=='__main__':
    unittest.main()
