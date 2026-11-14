"""Stage J workflow tests (CPU, synthetic backend, no images): fail-closed checks, negative tests and a synthetic
end-to-end six-fold dry run that also proves no official-test file and no image file is opened."""
from __future__ import annotations

import builtins
import json
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research_r2.packets import PacketError, fold_contract, validate_packet  # noqa: E402
from research_r2_v4 import jrun  # noqa: E402
from research_r2_v4.common import INNER, env  # noqa: E402

_E = {}


def E():
    if not _E:
        _E["frame"], _E["bench"], _E["F"] = env()
    return _E["frame"], _E["bench"], _E["F"]


def syn_cfg(row="Jt", **kw):
    return jrun.JConfig(row, backbone="synthetic", trainable="synthetic", epochs=1, batch=256, lr_backbone=1e-2, workers=0, **kw)


class TestImageAccess(unittest.TestCase):
    def test_evaluation_rows_refused_during_training(self):
        frame, bench, F = E()
        acc = jrun.ImageAccess(frame, None, loader=jrun.synthetic_loader(frame))
        acc.allow("train", F["I1"]["fitting_rows"])
        for r in (F["I1"]["seen_val_rows"][0], F["I1"]["unseen_val_rows"][0], bench["outer_exclusion_rows"][0]):
            with self.assertRaises(jrun.JCheckError):
                acc.get(r)
        self.assertEqual(acc.opened, 0, "a refused row must not be loaded")
        acc.get(F["I1"]["fitting_rows"][0])
        self.assertEqual(acc.opened, 1)

    def test_nothing_allowed_before_a_phase(self):
        frame, _, _ = E()
        with self.assertRaises(jrun.JCheckError):
            jrun.ImageAccess(frame, None, loader=jrun.synthetic_loader(frame)).get(0)

    def test_training_filenames_never_trip_the_test_name_guard(self):
        frame, _, _ = E()
        self.assertFalse(frame.file.str.lower().str.contains("test").any())

    def test_layout_chunks_load_only_allowed_rows(self):
        frame, _, F = E()
        acc = jrun.ImageAccess(frame, None, loader=jrun.synthetic_loader(frame))
        rows = sorted(F["I1"]["fitting_rows"] + F["I1"]["seen_val_rows"] + F["I1"]["unseen_val_rows"])
        acc.allow("extract", rows)
        ds = jrun.LayoutChunks(acc, rows, lambda t: t, len(frame), batch=32, shape=(3, 8, 8))
        seen = []
        for i in range(len(ds)):
            x, keep, rr = ds[i]
            self.assertEqual(len(rr), 32)
            self.assertEqual(int(rr[0]) % 32, 0, "chunks must follow the v3 batch layout")
            self.assertTrue(bool((x[~keep] == 0).all()), "non-fold slots must be zero, never loaded")
            seen += rr[keep].tolist()
        self.assertEqual(sorted(seen), rows)
        self.assertEqual(acc.opened, len(rows))

    def test_run_fold_rejects_overlapping_fitting_rows(self):
        frame, bench, F = E()
        bad = dict(F["I1"], fitting_rows=F["I1"]["fitting_rows"] + F["I1"]["seen_val_rows"][:1])
        acc = jrun.ImageAccess(frame, None, loader=jrun.synthetic_loader(frame))
        with tempfile.TemporaryDirectory() as d, self.assertRaises(jrun.JCheckError):
            jrun.run_fold(syn_cfg(), 0, bad, 0, frame, bench, jrun.SyntheticBackend(), acc, Path(d), device="cpu")


class TestFeatureCache(unittest.TestCase):
    def test_stale_or_mismatched_cache_refused(self):
        meta = {"row": "J1", "seed": 0, "fold": "I1", "config_sha256": "a", "checkpoint_sha256": "b",
                "rows_sha256": jrun.sha256_json([1, 2]), "backbone": "clip_l14", "schema": "r2v4-j-fold-features-v1"}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.pt"
            torch.save({"meta": meta, "rows": [1, 2], "features": torch.zeros(2, 2, 4)}, p)
            jrun.load_feature_cache(p, meta)  # exact match loads
            for k, v in (("checkpoint_sha256", "other"), ("seed", 1), ("fold", "I2"), ("config_sha256", "x")):
                with self.assertRaises(jrun.JCheckError):
                    jrun.load_feature_cache(p, dict(meta, **{k: v}))
            torch.save({"meta": meta, "rows": [1, 3], "features": torch.zeros(2, 2, 4)}, p)
            with self.assertRaises(jrun.JCheckError):
                jrun.load_feature_cache(p, meta)
            p.write_bytes(b"not a torch file")
            with self.assertRaises(jrun.JCheckError):
                jrun.load_feature_cache(p, meta)


class TestCheckpoint(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.cfg = syn_cfg()
        self.be = jrun.SyntheticBackend()
        self.model, _, _ = self.be.build(self.cfg, "cpu")
        with torch.no_grad():
            self.model[3].weight.add_(0.1)  # a "trained" state that differs from a fresh build
        self.heads = jrun.Heads(16, 55, 175)
        x = torch.randn(4, 3, 8, 8)
        self.probe = {"x": x, "z": self.be.embed(self.model.eval(), x)}
        self.meta = {"row": "Jt", "seed": 0, "fold": "I1", "config_sha256": self.cfg.sha()}
        self.ck = Path(self.d) / "checkpoint.pt"
        jrun.save_checkpoint(self.ck, self.model, self.heads, self.meta)

    def tearDown(self):
        shutil.rmtree(self.d)

    def load(self):
        return jrun.load_verified(self.ck, self.be, self.cfg, (55, 175), self.probe, self.meta, "cpu")

    def test_good_checkpoint_reloads_exactly(self):
        self.assertEqual(self.load()[3], 0.0)

    def test_corrupted_file(self):
        self.ck.write_bytes(self.ck.read_bytes()[:200])
        with self.assertRaises(jrun.JCheckError):
            self.load()

    def test_missing_key(self):
        st = torch.load(self.ck, weights_only=False)
        st["backbone"].pop(sorted(st["backbone"])[0]); torch.save(st, self.ck)
        with self.assertRaises(jrun.JCheckError):
            self.load()

    def test_altered_weight_fails_output_check(self):
        st = torch.load(self.ck, weights_only=False)
        k = sorted(st["backbone"])[0]; st["backbone"][k] = st["backbone"][k] + 1e-3; torch.save(st, self.ck)
        with self.assertRaises(jrun.JCheckError):
            self.load()

    def test_metadata_mismatch(self):
        with self.assertRaises(jrun.JCheckError):
            jrun.load_verified(self.ck, self.be, self.cfg, (55, 175), self.probe, dict(self.meta, fold="I2"), "cpu")


class TestGradMonitor(unittest.TestCase):
    def params(self):
        return [("a", torch.nn.Parameter(torch.ones(3))), ("b", torch.nn.Parameter(torch.ones(2)))]

    def test_missing_gradient(self):
        ps = self.params(); ps[0][1].grad = torch.ones(3)
        with self.assertRaises(jrun.JCheckError):
            jrun.GradMonitor(ps).check()

    def test_three_consecutive_nonfinite(self):
        ps = self.params(); m = jrun.GradMonitor(ps)
        ps[0][1].grad = torch.tensor([float("nan")] * 3); ps[1][1].grad = torch.ones(2)
        m.check(); m.check()
        with self.assertRaises(jrun.JCheckError):
            m.check()

    def test_never_nonzero_and_skips(self):
        ps = self.params(); m = jrun.GradMonitor(ps)
        ps[0][1].grad = torch.ones(3); ps[1][1].grad = torch.zeros(2)
        m.check()
        with self.assertRaises(jrun.JCheckError):
            m.finish(0, 10)  # tensor b never received a non-zero gradient
        ps[1][1].grad = torch.ones(2); m.check()
        with self.assertRaises(jrun.JCheckError):
            m.finish(6, 100)  # 6% of steps skipped
        self.assertEqual(m.finish(5, 100)["tensors"], 2)


class FrozenMutatingBackend(jrun.SyntheticBackend):
    @staticmethod
    def embed(model, x):
        with torch.no_grad():
            model[1].weight.add_(1e-6)  # illegal: changes a frozen parameter
        return model(x.float()).float()


class NoTrainableBackend(jrun.SyntheticBackend):
    def build(self, cfg, device="cpu", lora_seed=0):
        m, tf, w = super().build(cfg, device, lora_seed)
        for p in m.parameters():
            p.requires_grad_(False)
        return m, tf, w


class TestFailClosedFold(unittest.TestCase):
    def run_fold(self, backend):
        frame, bench, F = E()
        acc = jrun.ImageAccess(frame, None, loader=jrun.synthetic_loader(frame))
        with tempfile.TemporaryDirectory() as d:
            try:
                jrun.run_fold(syn_cfg(max_steps=3), 0, F["I1"], 0, frame, bench, backend, acc, Path(d), device="cpu")
            finally:
                self.assertFalse((Path(d) / "receipt.json").exists(), "no receipt may be written for a failed fold")

    def test_frozen_parameter_change_fails(self):
        with self.assertRaisesRegex(jrun.JCheckError, "frozen parameters changed"):
            self.run_fold(FrozenMutatingBackend())

    def test_no_trainable_parameters_fails(self):
        with self.assertRaisesRegex(jrun.JCheckError, "no trainable"):
            self.run_fold(NoTrainableBackend())


class TestPacketsAndReceipts(unittest.TestCase):
    def test_incorrect_candidate_order_refused(self):
        frame, _, F = E()
        con = fold_contract(frame, F["I1"])
        k = len(con["cand_ids"])
        pk = {"cand_ids": con["cand_ids"].copy(), "seen": np.zeros((len(con["seen_truth"]), k), np.float32),
              "unseen": np.zeros((len(con["unseen_truth"]), k), np.float32), "seen_truth": con["seen_truth"], "unseen_truth": con["unseen_truth"]}
        validate_packet(pk, con)
        pk["cand_ids"] = pk["cand_ids"][::-1].copy()
        with self.assertRaises(PacketError):
            validate_packet(pk, con)


class TestNestedParentRule(unittest.TestCase):
    """The J4a-J7 parent used in a held fold is chosen on the other five folds only."""

    def tables(self, perfect=None):
        from research_r2_v4 import evaluate as ev
        from research_r2_v4.common import scorer
        frame, _, F = E()

        def arm(fn, seed, good=False):
            con = fold_contract(frame, F[fn]); rng = np.random.default_rng(seed); k = len(con["cand_ids"])
            s = rng.normal(size=(len(con["seen_truth"]), k)); u = rng.normal(size=(len(con["unseen_truth"]), k))
            if good:
                s[np.arange(len(s)), con["seen_truth"]] += 50; u[np.arange(len(u)), con["unseen_truth"]] += 50
            return scorer(frame, {"seen": s, "unseen": u, "seen_truth": con["seen_truth"], "unseen_truth": con["unseen_truth"],
                                  "cand_ids": con["cand_ids"], "fold": F[fn]})
        T = {}
        for i, r in enumerate(["D1_synthComp_norm", "J1_clipL_last2", "J2_clipL_last4", "J3_clipL_lora8_mlp", "J5_best_source_balanced"]):
            T[r] = ev.RowTable(r, {fn: [arm(fn, 100 * i + j, good=(perfect == (r, fn)))] for j, fn in enumerate(INNER)}, ["single"])
        return T

    def test_held_fold_does_not_choose_its_parent(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import r2v4_stage_j as sj
        cands = ["J1_clipL_last2", "J2_clipL_last4", "J3_clipL_lora8_mlp", "J5_best_source_balanced"]
        base = sj.nested_with_parent_rule(self.tables(), "D1_synthComp_norm", cands, INNER, {"J5_best_source_balanced": "J1_clipL_last2"})
        for k in INNER:  # make J1 perfect ONLY on the held fold k: k's own inner parent must not change
            alt = sj.nested_with_parent_rule(self.tables(perfect=("J1_clipL_last2", k)), "D1_synthComp_norm", cands, INNER,
                                             {"J5_best_source_balanced": "J1_clipL_last2"})
            hb = next(h for h in base["held"] if h["held_fold"] == k); ha = next(h for h in alt["held"] if h["held_fold"] == k)
            self.assertEqual(hb["inner_parent"], ha["inner_parent"], k)
        for h in base["held"]:  # children of J1 are selectable only where the inner parent is J1
            self.assertEqual("J5_best_source_balanced" in h["excluded_children"], h["inner_parent"] != "J1_clipL_last2")


class TestSyntheticSixFoldDryRun(unittest.TestCase):
    """End to end on CPU: run (2 synthetic rows x 6 folds) -> receipts -> evaluate -> H_RF (2 pairs), recording every
    file the process opens; then negative checks on missing / tampered packets."""

    def test_dry_run(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import r2v4_stage_j as sj
        opened = []
        real_open = builtins.open
        from PIL import Image
        real_img_open = Image.open

        def rec_open(file, *a, **k):
            opened.append(str(file)); return real_open(file, *a, **k)

        def rec_img(fp, *a, **k):
            opened.append("IMG:" + str(fp)); return real_img_open(fp, *a, **k)
        d = tempfile.mkdtemp()
        try:
            with mock.patch("builtins.open", rec_open), mock.patch.object(Image, "open", rec_img):
                statuses, loads = sj.cmd_dryrun(types.SimpleNamespace(out=d))
                res = sj.cmd_evaluate(types.SimpleNamespace(out=d, parent=None))
                sj.cmd_hrf(types.SimpleNamespace(out=d, row="Jsyn_A", seed=0, parent=None, max_pairs=2))
            for row, st in statuses.items():
                self.assertEqual(st, {f: "ok" for f in INNER}, row)
            self.assertGreater(loads, 0)
            self.assertEqual(set(res["rows"]), {"Jsyn_A", "Jsyn_B"})
            self.assertEqual(res["refused_incomplete"], {})
            for r in res["rows"].values():
                self.assertIn("gate_vs_D1", r["seed0"])
            hrf = [json.loads(l) for l in (Path(d) / "hrf" / "Jsyn_A__seed0" / "pairs.jsonl").read_text().splitlines()]
            self.assertEqual([h["status"] for h in hrf], ["ok", "ok"])
            for p in opened:
                self.assertNotIn("DPD_Small_test", p)
                self.assertNotIn("DPD_Small_dataset", p, f"an image file was opened: {p}")
                self.assertFalse(p.startswith("IMG:"), f"an image was opened: {p}")
            # fail-closed evaluation: a missing fold packet and a tampered packet make the row refused / raise
            a = jrun.completed_attempt(d, "Jsyn_B", 0, "I3")[0]
            (a / "I3_scores.npz").unlink()
            with self.assertRaises(jrun.JCheckError):
                jrun.row_packets(d, "Jsyn_B", 0, INNER)
            a2 = jrun.completed_attempt(d, "Jsyn_A", 0, "I2")[0]
            (a2 / "receipt.json").write_text((a2 / "receipt.json").read_text().replace('"status": "ok"', '"status": "failed"'))
            with self.assertRaises(jrun.JCheckError):
                jrun.row_packets(d, "Jsyn_A", 0, INNER)
            res2 = sj.cmd_evaluate(types.SimpleNamespace(out=d, parent=None))
            self.assertIn("Jsyn_A/seed0", res2["refused_incomplete"])
        finally:
            shutil.rmtree(d)


if __name__ == "__main__":
    unittest.main()
