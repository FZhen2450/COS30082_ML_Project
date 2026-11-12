"""Synthetic export corruption checks; no weights, images or official inputs."""
import copy
import unittest
from types import SimpleNamespace

import numpy as np

from scripts.verify_wp1_official_export import validate_packet


class OfficialPacketTests(unittest.TestCase):
    def setUp(self):
        self.ids = (2, 7, 11)
        self.samples = (SimpleNamespace(sample_id='a.jpg', truth_id=2),
                        SimpleNamespace(sample_id='b.jpg', truth_id=7))
        self.packet = {
            'scores': np.array([[0, 0, -1], [-2, 3, 1]], dtype=np.float32),
            'candidate_ids': np.array(self.ids, dtype=np.int64),
            'filenames': np.array(['a.jpg', 'b.jpg']),
            'truths': np.array([2, 7], dtype=np.int64),
            'top5_ids': np.array([[2, 7, 11], [7, 11, 2]], dtype=np.int64),
            'independent_crop_ids': np.array([0, 1], dtype=np.int64),
            'independent_disease_ids': np.array([3, 4], dtype=np.int64),
        }

    def test_noncontiguous_ids_and_ascending_id_ties(self):
        metrics, ranks = validate_packet(self.packet, self.samples, self.ids)
        self.assertEqual(metrics['hits_top1'], 2)
        self.assertEqual(ranks[0].tolist(), [2, 7, 11])

    def test_changed_packet_fields_precision_or_nonfinite_score_rejected(self):
        for change in ('extra', 'missing', 'float64', 'nan'):
            packet = copy.deepcopy(self.packet)
            if change == 'extra':
                packet['extra'] = np.array([1])
            elif change == 'missing':
                del packet['independent_crop_ids']
            elif change == 'float64':
                packet['scores'] = packet['scores'].astype(np.float64)
            else:
                packet['scores'][0, 0] = np.nan
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_packet(packet, self.samples, self.ids)

    def test_reordered_names_truths_or_candidate_columns_rejected(self):
        for field in ('filenames', 'truths', 'candidate_ids'):
            packet = copy.deepcopy(self.packet)
            packet[field] = packet[field][::-1]
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_packet(packet, self.samples, self.ids)

    def test_rank_tampering_and_component_ids_rejected(self):
        for field, value in [('top5_ids', 11), ('independent_crop_ids', 55),
                             ('independent_disease_ids', -1)]:
            packet = copy.deepcopy(self.packet)
            packet[field].flat[0] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_packet(packet, self.samples, self.ids)

    def test_boolean_truths_cannot_pass_as_integer_ids(self):
        packet = copy.deepcopy(self.packet)
        packet['truths'] = packet['truths'].astype(bool)
        with self.assertRaises(ValueError):
            validate_packet(packet, self.samples, self.ids)


if __name__ == '__main__':
    unittest.main()
