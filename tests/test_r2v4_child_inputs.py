"""Amendment 3 v3 PROPOSAL: verified-input boundary (research_r2_v4/child_inputs.py) on real synthetic runs.

A synthetic parent and a one-modification synthetic child (source_balanced) are trained at seed 0 on the six inner
folds with jrun.run_fold (CPU, synthetic backend, no images); the child runner writes child_identity.json. Negative
tests cover identity (including a same-label image swap), provenance, seeds, checkpoints, code and configuration."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "tests"))

from research_r2_v4 import child_gate as cg  # noqa: E402
from research_r2_v4 import child_inputs as ci  # noqa: E402
from research_r2_v4 import jrun  # noqa: E402
from research_r2_v4.common import INNER, env  # noqa: E402

_S = {}
PARENT, CHILD = "Jsyn_P", "Jsyn_C"


def pcfg():
    return jrun.JConfig(PARENT, backbone="synthetic", trainable="synthetic", epochs=1, batch=256, lr_backbone=1e-2, workers=0)


def ccfg():
    c = pcfg(); c.row = CHILD; c.source_balanced = True
    return c


def setUpModule():
    frame, bench, F = env()
    tmp = Path(tempfile.mkdtemp(prefix="r2v4_child_inputs_"))
    lin = ci.lineage_groups()
    acc = jrun.ImageAccess(frame, None, loader=jrun.synthetic_loader(frame))
    for cfg, base in ((pcfg(), tmp / "parent"), (ccfg(), tmp / "child")):
        st = jrun.run_row(cfg, 0, [F[f] for f in INNER], frame, bench, jrun.SyntheticBackend(), acc, base, device="cpu", log=lambda *_: None)
        assert all(v == "ok" for v in st.values()), st
    for fn in INNER:
        adir, rec = jrun.completed_attempt(tmp / "child", CHILD, 0, fn)
        ci.write_child_identity(adir, rec, frame, F[fn], parent_row=PARENT, lineage=lin)
    _S.update(frame=frame, F=F, tmp=tmp, lin=lin)


def tearDownModule():
    shutil.rmtree(_S["tmp"], ignore_errors=True)


def copy_child():
    d = Path(tempfile.mkdtemp(prefix="r2v4_ci_case_"))
    shutil.copytree(_S["tmp"] / "child", d / "child")
    return d / "child"


def pair(child_base, mode="screen", **kw):
    return ci.verified_pair(child_base, _S["tmp"] / "parent", CHILD, mode, _S["frame"], _S["F"], child_cfg=ccfg(), parent_cfg=pcfg(),
                            parent_row=PARENT, **kw)


def attempt(base, fn):
    return jrun.completed_attempt(base, CHILD, 0, fn)[0]


def edit_json(p, fn):
    x = json.loads(Path(p).read_text()); fn(x); Path(p).write_text(json.dumps(x))


def reseal(adir):
    """Refresh the receipt hash and the packet hash in the sidecar, so only the intended check can trip."""
    rec = json.loads((adir / "receipt.json").read_text())
    edit_json(adir / ci.SIDECAR, lambda s: s.update(receipt_sha256=jrun.sha256_file(adir / "receipt.json"), packet_sha256=rec["packet_sha256"]))


class TestVerifiedPair(unittest.TestCase):
    def test_good_pair_feeds_the_gate(self):
        C, P, meta = pair(_S["tmp"] / "child")
        self.assertEqual(meta["modification"], ("source_balanced", True))
        self.assertTrue(C[0].identity["verified"]); self.assertEqual(C[0].identity["seed"], 0)
        for fn in INNER:
            self.assertEqual(C[0].identity["folds"][fn]["identity_sha256"], P[0].identity["folds"][fn]["identity_sha256"])
        out = cg.primary_gate(C, P, INNER, "screen", reps=20)
        self.assertEqual(out["role"], "primary"); self.assertIn("pass", cg.promotion_decision(out))

    def test_same_label_swap_inside_a_packet_is_refused(self):
        d = copy_child(); a = attempt(d, "I3")
        z = dict(np.load(a / "I3_scores.npz")); t = z["seen_truth"]
        i = next(i for i in range(len(t)) if (t[i + 1:] == t[i]).any()); j = i + 1 + int(np.flatnonzero(t[i + 1:] == t[i])[0])
        z["seen"][[i, j]] = z["seen"][[j, i]]
        self.assertTrue(np.array_equal(z["seen_truth"], np.load(a / "I3_scores.npz")["seen_truth"]))  # labels unchanged
        np.savez_compressed(a / "I3_scores.npz", **z)
        edit_json(a / "receipt.json", lambda r: r.update(packet_sha256=jrun.sha256_file(a / "I3_scores.npz"))); reseal(a)
        pair(d, rederive=False)  # receipt, contract, labels and sidecar checks alone are blind to the exchange ...
        with self.assertRaisesRegex(ci.ChildInputError, "does not re-derive"):
            pair(d)  # ... the re-derivation from the images' own features is not

    def test_same_label_swap_in_the_fold_order_is_refused(self):
        from test_r2v4_child_gate import same_label_swap
        d = copy_child(); a = attempt(d, "I3")
        swapped, _ = same_label_swap("I3")
        rec = json.loads((a / "receipt.json").read_text())
        ci.write_child_identity(a, rec, _S["frame"], swapped, parent_row=PARENT, lineage=_S["lin"])
        with self.assertRaisesRegex(ci.ChildInputError, r"child provenance does not match \(\['identity_sha256'\]\)"):
            pair(d)
        with self.assertRaisesRegex(ci.ChildInputError, "seen_val_rows differs"):
            ci.check_registered_fold(swapped, _S["F"]["I3"])
        ci.check_registered_fold(dict(_S["F"]["I3"]), _S["F"]["I3"])

    def test_missing_or_invalid_provenance_is_refused(self):
        d = copy_child(); (attempt(d, "I2") / ci.SIDECAR).unlink()
        with self.assertRaisesRegex(ci.ChildInputError, "missing child provenance"):
            pair(d)
        d = copy_child(); edit_json(attempt(d, "I2") / ci.SIDECAR, lambda s: s.update(parent_row="J3_clipL_lora8_mlp"))
        with self.assertRaisesRegex(ci.ChildInputError, "parent_row"):
            pair(d)
        d = copy_child(); edit_json(attempt(d, "I5") / "receipt.json", lambda r: r.update(contract_sha256="0" * 64)); reseal(attempt(d, "I5"))
        with self.assertRaisesRegex(ci.ChildInputError, "registered fold contract"):
            pair(d)
        d = copy_child(); edit_json(attempt(d, "I1") / "receipt.json", lambda r: r.update(seconds=1.0))  # receipt changed after sidecar
        with self.assertRaisesRegex(ci.ChildInputError, "receipt_sha256"):
            pair(d)

    def test_stale_or_foreign_feature_cache_is_refused(self):
        d = copy_child(); a, b = attempt(d, "I3"), attempt(d, "I2")
        shutil.copy(b / "features.pt", a / "features.pt")
        edit_json(a / "receipt.json", lambda r: r.update(features_sha256=jrun.sha256_file(a / "features.pt"))); reseal(a)
        with self.assertRaisesRegex(jrun.JCheckError, "feature cache metadata mismatch"):
            pair(d)

    def test_seed_and_checkpoint_identity(self):
        d = copy_child(); a = attempt(d, "I4"); edit_json(a / "receipt.json", lambda r: r.update(seed=1)); reseal(a)
        with self.assertRaisesRegex(jrun.JCheckError, "receipt seed=1, expected 0"):
            pair(d)
        d = copy_child(); a = attempt(d, "I4"); edit_json(a / "receipt.json", lambda r: r.update(torch_seed=1003)); reseal(a)
        with self.assertRaisesRegex(jrun.JCheckError, "torch_seed"):
            pair(d)
        parent_ck = jrun.completed_attempt(_S["tmp"] / "parent", PARENT, 0, "I4")[1]["checkpoint_sha256"]
        d = copy_child(); a = attempt(d, "I4"); edit_json(a / "receipt.json", lambda r: r.update(checkpoint_sha256=parent_ck)); reseal(a)
        with self.assertRaisesRegex(jrun.JCheckError, "checkpoint.pt missing or changed"):
            pair(d)  # a parent checkpoint hash cannot be claimed by a child run
        with self.assertRaisesRegex(ci.ChildInputError, "seed directories"):
            pair(_S["tmp"] / "child", mode="confirm")

    def test_training_code_must_match(self):
        d = copy_child()
        for fn in INNER:
            edit_json(attempt(d, fn) / "receipt.json", lambda r: r["code"].update(jrun_sha256="f" * 64)); reseal(attempt(d, fn))
        with self.assertRaisesRegex(ci.ChildInputError, "different training code"):
            pair(d)


class TestConfigurationAndLineage(unittest.TestCase):
    def test_exactly_one_registered_modification(self):
        j2 = asdict(jrun.j_rows()[ci.PARENT_ROW])
        want = {"J4a_best_geometric_aug": ("augment", "geometric"), "J4b_best_photometric_aug": ("augment", "photometric"),
                "J5_best_source_balanced": ("source_balanced", True), "J6_best_crosscrop_supcon": ("extra", "supcon"),
                "J7_best_crop_adversary": ("extra", "adversary")}
        for row, mod in want.items():
            self.assertEqual(ci.modification(asdict(ci.declared_child_config(row)), j2), mod)
        with self.assertRaisesRegex(ci.ChildInputError, "not a registered J4a-J7 child"):
            ci.declared_child_config("J1_clipL_last2")
        with self.assertRaisesRegex(ci.ChildInputError, "exactly one"):
            ci.modification(dict(j2, row="X"), j2)  # null child: no modification
        with self.assertRaisesRegex(ci.ChildInputError, "exactly one"):
            ci.modification(dict(j2, row="X", augment="photometric", source_balanced=True), j2)
        with self.assertRaisesRegex(ci.ChildInputError, "exactly one"):
            ci.verified_pair(_S["tmp"] / "child", _S["tmp"] / "parent", CHILD, "screen", _S["frame"], _S["F"],
                             child_cfg=ccfg(), parent_cfg=ccfg(), parent_row=PARENT)

    def test_lineage_shared_between_fitting_and_evaluation_is_refused(self):
        F = _S["F"]; lin = _S["lin"].copy()
        ci.check_lineage(F["I1"], lin)
        lin[F["I1"]["seen_val_rows"][0]] = lin[F["I1"]["fitting_rows"][0]]
        with self.assertRaisesRegex(ci.ChildInputError, "lineage group is shared"):
            ci.check_lineage(F["I1"], lin)


if __name__ == "__main__":
    unittest.main()
