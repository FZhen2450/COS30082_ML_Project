"""Second Run final model (research_r2_v5/final.py, scripts/r2v5_final.py): CPU tests with the synthetic backend.
Candidate payload and frozen configuration, full-data training contract, 333-candidate scoring, calibration, ranking,
metric equivalence with First Run's evaluation/metrics.py, and the guarded one-shot official evaluation."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from evaluation import metrics as fr_metrics  # noqa: E402  (First Run scoring contract, read-only)
from evaluation.reporting import FINAL_CANDIDATE_SHA256, FINAL_UNSEEN_IDS, TIE_RULE  # noqa: E402
from research_r2_v4 import confirm, jrun  # noqa: E402
from research_r2_v4.common import env  # noqa: E402
from research_r2_v5 import final as fm  # noqa: E402

_S = {}


def setUpModule():
    frame, _, _ = env()
    tmp = Path(tempfile.mkdtemp(prefix="r2v5_final_"))
    cfg = jrun.JConfig("Jsyn_full", backbone="synthetic", trainable="synthetic", epochs=1, batch=256, lr_backbone=1e-2, workers=0)
    acc = jrun.ImageAccess(frame, None, loader=jrun.synthetic_loader(frame))
    rec = fm.train_final(cfg, 0, frame, jrun.SyntheticBackend(), acc, tmp / "run", device="cpu")
    feats, rows_ = jrun.load_feature_cache(tmp / "run" / "features.pt", rec["feature_cache_meta"])
    _S.update(frame=frame, tmp=tmp, cfg=cfg, rec=rec, feats=feats, frows=rows_, rows=fm.load_payload(frame))


def tearDownModule():
    shutil.rmtree(_S["tmp"], ignore_errors=True)


class TestCandidatesAndConfig(unittest.TestCase):
    def test_payload_matches_first_run_contract_and_training_csv(self):
        rows = _S["rows"]
        self.assertEqual(fm.FINAL_CANDIDATE_SHA256, FINAL_CANDIDATE_SHA256)
        self.assertEqual(fm.FINAL_UNSEEN_IDS, FINAL_UNSEEN_IDS)
        self.assertEqual([r["composition_id"] for r in rows], list(range(333)))
        self.assertEqual(int(fm.seen_mask(rows).sum()), 323)
        self.assertEqual(TIE_RULE, "stable_column_order_ascending_global_id")

    def test_tampered_payload_refused(self):
        p = _S["tmp"] / "payload.json"
        d = json.loads(fm.PAYLOAD.read_text()); d["rows"][39]["disease_id"] = 0
        p.write_text(json.dumps(d))
        with self.assertRaisesRegex(fm.FinalError, "sha256"):
            fm.load_payload(_S["frame"], p)

    def test_frozen_config_is_validated_j2_and_development_gamma(self):
        import r2v5_final as cli
        c = fm.load_config()
        cfg = cli.cfg_of(c)
        self.assertEqual((c["row"], cfg.trainable, cfg.augment, cfg.epochs, c["seed"], c["gamma"]), ("J2_clipL_last4", "last4", "flip", 5, 0, 8.75))
        self.assertEqual(c["d1_cfg"], jrun.D1_CFG)


class TestTrainingContract(unittest.TestCase):
    def test_full_fold_trains_on_every_training_row_and_nothing_else(self):
        f = fm.full_fold(_S["frame"])
        self.assertEqual(f["fitting_rows"], list(range(8000)))
        self.assertEqual((f["seen_val_rows"], f["unseen_val_rows"], f["heldout_composition_ids"]), ([], [], []))
        rec = _S["rec"]
        self.assertEqual((rec["status"], rec["n_fitting"], rec["torch_seed"], rec["reload_max_abs_diff"]), ("ok", 8000, 0, 0.0))
        self.assertEqual(confirm.receipt_problems(rec, "Jsyn_full", 0, fm.FOLD_NAME, 0, _S["cfg"]), [])
        self.assertEqual(_S["frows"], list(range(8000)))
        self.assertIsNotNone(jrun.verify_receipt(_S["tmp"] / "run"))

    def test_training_access_refuses_non_training_images(self):
        frame = _S["frame"].copy()
        frame.loc[0, "file"] = "DPD_Small_test_seen_x.jpg"
        acc = jrun.ImageAccess(_S["frame"], "/nonexistent"); acc.allow("train", [0])
        acc.train_files = set()  # a file that is not a training-CSV image
        with self.assertRaisesRegex(jrun.JCheckError, "not a training-CSV image"):
            acc.check(0)


class TestScoringAndRanking(unittest.TestCase):
    def scores(self, idx=None):
        idx = list(range(0, 8000, 400)) if idx is None else idx
        return fm.raw_scores(_S["frame"], _S["feats"], _S["rows"], _S["feats"][idx, 0])

    def test_scores_finite_333_columns_deterministic(self):
        a, b = self.scores(), self.scores()
        self.assertEqual(a.shape, (20, 333)); self.assertTrue(np.isfinite(a).all()); self.assertTrue(np.array_equal(a, b))

    def test_calibration_penalises_exactly_the_seen_columns(self):
        s = self.scores(); cal = fm.calibrate(s, _S["rows"], 8.75)
        d = s - cal
        self.assertTrue(np.allclose(d[:, list(FINAL_UNSEEN_IDS)], 0.0))
        seen = [i for i in range(333) if i not in FINAL_UNSEEN_IDS]
        self.assertTrue(np.allclose(d[:, seen], 8.75))

    def test_stable_ties_and_top1_top5_consistency(self):
        cal = np.zeros((2, 333)); cal[1, [5, 3, 200]] = 1.0
        t = fm.topk(cal, 5)
        self.assertEqual(t[0].tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(t[1].tolist(), [3, 5, 200, 0, 1])
        cal = fm.calibrate(self.scores(), _S["rows"], 8.75)
        self.assertTrue(np.array_equal(fm.topk(cal, 5)[:, 0], fm.topk(cal, 1)[:, 0]))

    def test_metrics_equal_first_run_contract(self):
        rng = np.random.default_rng(0)
        cal = rng.normal(size=(50, 333)); cal[:, 7] = cal[:, 8]  # include ties
        truth = rng.integers(0, 333, 50)
        m = fm.metrics(cal, truth)
        self.assertAlmostEqual(m["top1"], fr_metrics.topk_accuracy(cal, truth, list(range(333)), 1), places=12)
        self.assertAlmostEqual(m["top5"], fr_metrics.topk_accuracy(cal, truth, list(range(333)), 5), places=12)
        self.assertAlmostEqual(fm.harmonic(0.3, 0.1), fr_metrics.harmonic_mean(0.3, 0.1), places=12)

    def test_non_finite_features_refused(self):
        q = _S["feats"][:3, 0].float().clone(); q[0, 0] = float("nan")
        with self.assertRaisesRegex(fm.FinalError, "non-finite"):
            fm.raw_scores(_S["frame"], _S["feats"], _S["rows"], q)


class TestOfficialEvaluationGuard(unittest.TestCase):
    def run_cli(self, env_extra, extra_args):
        e = dict(os.environ, PYTHONDONTWRITEBYTECODE="1"); e.pop("R2V5_OFFICIAL_EVAL_AUTHORIZED", None); e.update(env_extra)
        return subprocess.run([sys.executable, str(ROOT / "scripts/r2v5_final.py"), "official-eval", "--run", "/nonexistent",
                               "--verification", "/nonexistent/v.json", "--seen-csv", "/nonexistent/a.csv", "--unseen-csv", "/nonexistent/b.csv",
                               "--out", str(_S["tmp"] / "o")] + extra_args,
                              cwd=ROOT, capture_output=True, text=True, env=e)

    def test_refused_without_explicit_authorisation(self):
        for env_extra, args in (({}, []), ({}, ["--owner-authorization", "yes"]), ({"R2V5_OFFICIAL_EVAL_AUTHORIZED": "1"}, [])):
            r = self.run_cli(env_extra, args)
            self.assertNotEqual(r.returncode, 0); self.assertIn("REFUSED", r.stdout + r.stderr)
        self.assertFalse((_S["tmp"] / "o").exists())

    def test_non_frozen_checkpoint_refused_before_inputs_are_read(self):
        import r2v5_final as cli
        v = _S["tmp"] / "ver.json"; v.write_text(json.dumps({"checkpoint_sha256_now": "0" * 64}))
        with self.assertRaisesRegex(fm.FinalError, "not the frozen candidate"):
            cli.check_frozen(_S["tmp"], v)  # the synthetic run is not the frozen checkpoint


def synthetic_parts(rows, n_seen=2000, n_unseen=1448):
    """Fabricated official-format records (file names that exist nowhere); seen labels cycle over the 323 seen
    compositions and unseen labels over the 10 unseen ones, with crop/disease taken from the payload."""
    seen_ids = [r for r in rows if r["composition_id"] not in FINAL_UNSEEN_IDS]
    uns = [r for r in rows if r["composition_id"] in FINAL_UNSEEN_IDS]
    mk = lambda tag, i, r: {"file": f"SYN_{tag}_{i:05d}.jpg", "crop_id": r["crop_id"], "disease_id": r["disease_id"], "comp_id": r["composition_id"]}  # noqa: E731
    return {"seen": [mk("seen", i, seen_ids[i % len(seen_ids)]) for i in range(n_seen)],
            "unseen": [mk("uns", i, uns[i % len(uns)]) for i in range(n_unseen)]}


def write_csv(path, recs, rows):
    import csv
    by = {r["composition_id"]: r for r in rows}
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        for r in recs:
            b = by.get(r["comp_id"], {"crop_name": "x", "disease_name": "y", "composition_name": "x_y"})
            w.writerow([r["file"], r["crop_id"], r["disease_id"], r["comp_id"], b["crop_name"], b["disease_name"], b["composition_name"]])


class TestOfficialInputValidation(unittest.TestCase):
    """The official evaluation validates BOTH parts before any model is loaded; every failure stops it."""

    def validate(self, mutate=None, image_root=None):
        parts = synthetic_parts(_S["rows"])
        if mutate:
            mutate(parts)
        return fm.validate_official_inputs(parts, _S["frame"], _S["rows"], image_root=image_root)

    def test_valid_synthetic_parts_pass_and_csv_round_trip(self):
        s = self.validate()
        self.assertEqual(s, {"seen": {"rows": 2000, "compositions": 323}, "unseen": {"rows": 1448, "compositions": 10}})
        parts = synthetic_parts(_S["rows"])
        p = _S["tmp"] / "seen.csv"; write_csv(p, parts["seen"], _S["rows"])
        self.assertEqual(fm.read_assignment_csv(p), parts["seen"])

    def test_every_integrity_failure_stops(self):
        tr = _S["frame"].file.iloc[0]
        cases = {
            "seen: 1999 rows": lambda p: p["seen"].pop(),
            "unseen: 1449 rows": lambda p: p["unseen"].append(dict(p["unseen"][0], file="SYN_extra.jpg")),
            "seen: duplicate": lambda p: p["seen"].__setitem__(1, dict(p["seen"][1], file=p["seen"][0]["file"])),
            "also in the training CSV": lambda p: p["seen"].__setitem__(0, dict(p["seen"][0], file=tr)),
            "not a seen composition": lambda p: p["seen"].__setitem__(0, dict(p["unseen"][0], file="SYN_x.jpg")),
            "not a unseen composition": lambda p: p["unseen"].__setitem__(0, dict(p["seen"][0], file="SYN_y.jpg")),
            "outside the 333-candidate universe": lambda p: p["unseen"].__setitem__(0, dict(p["unseen"][0], comp_id=333)),
            "disagree with the candidate payload": lambda p: p["seen"].__setitem__(0, dict(p["seen"][0], disease_id=p["seen"][0]["disease_id"] + 1)),
            "share image files": lambda p: p["unseen"].__setitem__(0, dict(p["unseen"][0], file=p["seen"][5]["file"])),
        }
        for msg, mut in cases.items():
            with self.subTest(msg), self.assertRaisesRegex(fm.FinalError, msg.replace("(", r"\(").replace(")", r"\)")):
                self.validate(mut)
        with self.assertRaisesRegex(fm.FinalError, "image file\\(s\\) missing"):
            self.validate(image_root=_S["tmp"])

    def test_malformed_csv_rows_refused(self):
        bad = {"header.csv": "image_name,crop_id,disease_id,composition_id,crop_name,disease_name,composition_name\n",
               "six.csv": "a.jpg,1,2,3,x,y\n", "nonint.csv": "a.jpg,1,two,3,x,y,z\n", "traversal.csv": "../a.jpg,1,2,3,x,y,z\n"}
        for name, text in bad.items():
            p = _S["tmp"] / name; p.write_text(text)
            with self.subTest(name), self.assertRaises(fm.FinalError):
                fm.read_assignment_csv(p)

    def test_confusion_matrix_dimensions(self):
        cm = fm.confusion([0, 5, 332, 5], [0, 7, 332, 5])
        self.assertEqual(cm.shape, (333, 333)); self.assertEqual(int(cm.sum()), 4); self.assertEqual(int(cm[5, 7]), 1)


if __name__ == "__main__":
    unittest.main()
