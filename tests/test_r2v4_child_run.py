"""Amendment-3 child runner (research_r2_v4/child_run.py): synthetic end-to-end and negative tests (CPU, no images)."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

from research_r2_v4 import child_gate as cg  # noqa: E402
from research_r2_v4 import child_inputs as ci  # noqa: E402
from research_r2_v4 import child_run as cr  # noqa: E402
from research_r2_v4 import jrun  # noqa: E402
from research_r2_v4.common import INNER, env  # noqa: E402

PARENT, CHILD = "Jsyn_P", "Jsyn_C"
_S = {}


def pcfg():
    return jrun.JConfig(PARENT, backbone="synthetic", trainable="synthetic", epochs=1, batch=256, lr_backbone=1e-2, workers=0)


def ccfg():
    c = pcfg(); c.row = CHILD; c.source_balanced = True
    return c


class FrozenMutating(jrun.SyntheticBackend):
    @staticmethod
    def embed(model, x):
        with torch.no_grad():
            model[1].weight.add_(1e-6)
        return model(x.float()).float()


def setUpModule():
    frame, bench, F = env()
    tmp = Path(tempfile.mkdtemp(prefix="r2v4_child_run_"))
    acc = jrun.ImageAccess(frame, None, loader=jrun.synthetic_loader(frame))
    st = jrun.run_row(pcfg(), 0, [F[f] for f in INNER], frame, bench, jrun.SyntheticBackend(), acc, tmp / "parent", device="cpu", log=lambda *_: None)
    assert all(v == "ok" for v in st.values()), st
    _S.update(frame=frame, bench=bench, F=F, tmp=tmp)


def tearDownModule():
    shutil.rmtree(_S["tmp"], ignore_errors=True)


def run(out, backend=None, clean=lambda: True):
    acc = jrun.ImageAccess(_S["frame"], None, loader=jrun.synthetic_loader(_S["frame"]))
    return cr._run(ccfg(), 0, out, _S["frame"], _S["bench"], _S["F"], backend or jrun.SyntheticBackend(), acc, parent_row=PARENT,
                   device="cpu", worktree_check=clean, log=lambda *_: None)


class TestChildRunner(unittest.TestCase):
    def test_end_to_end_runner_then_verified_gate(self):
        out = _S["tmp"] / "child_ok"
        st = run(out)
        self.assertEqual(st, {fn: "ok" for fn in INNER})
        for fn in INNER:
            a = out / CHILD / "seed0" / fn / "attempt1"
            s = json.loads((a / ci.SIDECAR).read_text())
            self.assertEqual((s["row"], s["seed"], s["fold"], s["parent_row"]), (CHILD, 0, fn, PARENT))
        C, P, meta = ci._pair(out, {0: _S["tmp"] / "parent"}, CHILD, PARENT, ccfg(), pcfg(), "screen", _S["frame"], _S["F"])
        self.assertEqual(meta["modification"], ("source_balanced", True))
        dec = cg.promotion_decision(C, P, "screen")
        self.assertIn(dec["pass"], (True, False))
        self.assertEqual(run(out), {fn: "ok (existing)" for fn in INNER})  # rerun: nothing retrained, nothing overwritten

    def test_dirty_worktree_refused_before_any_write(self):
        out = _S["tmp"] / "child_dirty"
        with self.assertRaisesRegex(cr.ChildRunError, "worktree is not clean"):
            run(out, clean=lambda: False)
        self.assertFalse(out.exists())
        calls = iter([True] + [False] * 100)  # worktree changes during the run -> reported
        st = run(_S["tmp"] / "child_changed", clean=lambda: next(calls))
        self.assertEqual(st["_worktree"], "CHANGED DURING THE RUN")

    def test_protected_evidence_directories_refused(self):
        for p in (ci.PARENT_BASES[0], ci.PARENT_BASES[1] / "J2_clipL_last4", cr.V4 / "stage_hrt_20261116" / "runs", cr.V4):
            with self.assertRaisesRegex(cr.ChildRunError, "overlaps protected evidence"):
                cr.check_out_dir(p)
        cr.check_out_dir(cr.V4 / "stage_j4b_test_only_not_created")

    def test_failure_kept_and_retry_in_new_attempt(self):
        out = _S["tmp"] / "child_fail"
        st = run(out, backend=FrozenMutating())
        self.assertTrue(all(v.startswith("failed") for v in st.values()))
        a1 = out / CHILD / "seed0" / "I1" / "attempt1"
        self.assertTrue((a1 / "failure.json").exists()); self.assertFalse((a1 / ci.SIDECAR).exists())
        st2 = run(out)
        self.assertEqual(st2, {fn: "ok" for fn in INNER})
        self.assertTrue((a1 / "failure.json").exists(), "failed attempt preserved")
        self.assertTrue((out / CHILD / "seed0" / "I1" / "attempt2" / ci.SIDECAR).exists())

    def test_rejected_attempt_gets_no_sidecar(self):
        out = _S["tmp"] / "child_reject"
        with mock.patch.object(ci, "rederive_packet", side_effect=ci.ChildInputError("simulated re-derivation mismatch")):
            st = run(out)
        self.assertTrue(all(v.startswith("rejected") for v in st.values()))
        a = out / CHILD / "seed0" / "I3" / "attempt1"
        self.assertTrue((a / "child_rejected.json").exists()); self.assertFalse((a / ci.SIDECAR).exists())

    def test_production_entry_refuses_without_declaration_or_authorisation(self):
        with self.assertRaisesRegex(cr.ChildRunError, "not an authorised child row"):
            cr.run_child("J5_best_source_balanced", 0, _S["tmp"] / "x", None)
        with mock.patch.object(cr, "DECLARATION", _S["tmp"] / "missing.json"):
            with self.assertRaisesRegex(cr.ChildRunError, "not declared"):
                cr.run_child("J4b_best_photometric_aug", 0, _S["tmp"] / "x", None)
        fake = _S["tmp"] / "amend.json"; fake.write_text(json.dumps({"authorised_now": {"rows": ["J4b_best_photometric_aug"], "seeds": [0]}}))
        with mock.patch.object(cr, "AMENDMENT", fake), mock.patch.object(cr, "DECLARATION", fake), \
                mock.patch.object(cr.subprocess, "run", return_value=mock.Mock(returncode=0, stdout="OK", stderr="")):
            with self.assertRaisesRegex(cr.ChildRunError, "seed 1 is not authorised"):
                cr.check_declaration("J4b_best_photometric_aug", 1)
            cr.check_declaration("J4b_best_photometric_aug", 0)
        with mock.patch.object(cr, "AMENDMENT", fake), mock.patch.object(cr, "DECLARATION", fake), \
                mock.patch.object(cr.subprocess, "run", return_value=mock.Mock(returncode=1, stdout="CHANGED: [x]", stderr="")):
            with self.assertRaisesRegex(cr.ChildRunError, "does not verify"):
                cr.check_declaration("J4b_best_photometric_aug", 0)


if __name__ == "__main__":
    unittest.main()
