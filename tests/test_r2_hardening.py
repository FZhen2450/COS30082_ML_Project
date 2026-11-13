"""v3 hardening: fold contract, strict score-packet validation, configuration identity, packet provenance.

Negative tests build packets from the fold contract and then corrupt exactly one property; each must be
rejected with its specific error. Positive tests check that valid v3 and legacy (v1/v2) packets pass.
"""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from research_r2.data import load_training_source
from research_r2.packets import (PacketError, committed_matrix_index, config_sha256, declared_rows, file_sha256,
                                 fold_contract, resolve_packets, row_config, tree_rows,
                                 validate_packet)

ROOT = Path(__file__).resolve().parents[1]
V1, V2, V3 = (ROOT / f"artifacts/r2/benchmark_v{i}.json" for i in (1, 2, 3))
PROTO = ROOT / "research_r2/protocol_v3.json"
EXP = ROOT.parents[1] / "experiments/r2-compositional"


def packet(con, seed=0):
    g = np.random.default_rng(seed)
    n = len(con["cand_ids"])
    return {"cand_ids": con["cand_ids"].copy(), "seen": g.normal(size=(len(con["seen_truth"]), n)).astype(np.float32),
            "unseen": g.normal(size=(len(con["unseen_truth"]), n)).astype(np.float32),
            "seen_truth": con["seen_truth"].copy(), "unseen_truth": con["unseen_truth"].copy()}


def independent_contract(frame, fold):
    """Second derivation from the CSV and benchmark JSON only (no research_r2 helper)."""
    all_ids = sorted(int(c) for c in frame.comp_id.unique())
    if "candidate_policy" in fold:
        fit = set(int(c) for c in frame.comp_id.loc[fold["fitting_rows"]])
        ids = [c for c in all_ids if c in fit or c in set(fold["heldout_composition_ids"])]
    else:
        ids = all_ids
    col = {c: i for i, c in enumerate(ids)}
    return (ids, [col[int(c)] for c in frame.comp_id.loc[fold["seen_val_rows"]]],
            [col[int(c)] for c in frame.comp_id.loc[fold["unseen_val_rows"]]])


@unittest.skipUnless(V1.exists() and V2.exists() and V3.exists(), "benchmarks not built")
class TestFoldContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = load_training_source(ROOT / "DPD_Small_train.csv")
        cls.b = {i: json.loads(p.read_text()) for i, p in ((1, V1), (2, V2), (3, V3))}
        cls.F3 = {f["name"]: f for f in cls.b[3]["folds"]}
        cls.I1 = fold_contract(cls.frame, cls.F3["I1"])

    def test_contract_equals_independent_derivation_all_folds(self):
        for v in (1, 2, 3):
            for f in self.b[v]["folds"]:
                c = fold_contract(self.frame, f)
                ids, st, ut = independent_contract(self.frame, f)
                self.assertEqual(c["cand_ids"].tolist(), ids, (v, f["name"]))
                self.assertEqual(c["seen_truth"].tolist(), st, (v, f["name"]))
                self.assertEqual(c["unseen_truth"].tolist(), ut, (v, f["name"]))
                self.assertEqual(c["schema"], "v3" if v == 3 else "legacy")
                self.assertEqual(len(c["cand_ids"]), 301 if f["role"] == "inner" and v == 3 else 323)

    def test_contract_hash_is_stable(self):
        self.assertEqual(fold_contract(self.frame, self.F3["I1"])["sha256"], self.I1["sha256"])
        self.assertNotEqual(fold_contract(self.frame, self.F3["I2"])["sha256"], self.I1["sha256"])

    # ---- positive -----------------------------------------------------------------------------
    def test_valid_v3_packet_passes(self):
        validate_packet(packet(self.I1), self.I1)

    def test_legacy_packet_without_cand_ids_passes_on_legacy_fold(self):
        for v in (1, 2):
            f = next(f for f in self.b[v]["folds"] if f["role"] in ("screen", "inner"))
            c = fold_contract(self.frame, f)
            p = packet(c); p.pop("cand_ids")
            validate_packet(p, c)

    # ---- negative: one corruption each -----------------------------------------------------------
    def assertRejected(self, p, pattern, con=None):
        with self.assertRaisesRegex(PacketError, pattern):
            validate_packet(p, self.I1 if con is None else con)

    def test_reject_packet_from_another_fold(self):
        self.assertRejected(packet(fold_contract(self.frame, self.F3["I2"])), "shape|candidate|truth")

    def test_reject_legacy_packet_on_v3_fold(self):
        p = packet(fold_contract(self.frame, self.b[2]["folds"][0])); p.pop("cand_ids")
        self.assertRejected(p, "packet arrays")

    def test_reject_reordered_candidates_even_if_scores_permuted_consistently(self):
        p = packet(self.I1); perm = np.random.default_rng(1).permutation(len(p["cand_ids"]))
        inv = np.argsort(perm)
        p.update(cand_ids=p["cand_ids"][perm], seen=p["seen"][:, perm], unseen=p["unseen"][:, perm],
                 seen_truth=inv[p["seen_truth"]], unseen_truth=inv[p["unseen_truth"]])
        self.assertRejected(p, "reordered")

    def _drop(self, cid):
        p = packet(self.I1); k = int(np.nonzero(p["cand_ids"] == cid)[0][0])
        keep = np.arange(len(p["cand_ids"])) != k
        p.update(cand_ids=p["cand_ids"][keep], seen=p["seen"][:, keep], unseen=p["unseen"][:, keep])
        return p

    def test_reject_truncated_heldout_candidate(self):
        self.assertRejected(self._drop(self.F3["I1"]["heldout_composition_ids"][0]), "300 candidate ids")

    def test_reject_truncated_fitting_candidate(self):
        fit = self.I1["cand_ids"][self.I1["fit_mask"]][0]
        self.assertRejected(self._drop(fit), "300 candidate ids")

    def test_reject_extra_excluded_candidate(self):
        p = packet(self.I1); e = self.F3["I1"]["excluded_composition_ids"][0]
        p.update(cand_ids=np.append(p["cand_ids"], e), seen=np.hstack([p["seen"], p["seen"][:, :1]]),
                 unseen=np.hstack([p["unseen"], p["unseen"][:, :1]]))
        self.assertRejected(p, "extra")

    def test_reject_duplicate_candidate(self):
        p = packet(self.I1); p["cand_ids"][5] = p["cand_ids"][4]
        self.assertRejected(p, "duplicate")

    def test_reject_substituted_candidate_same_length(self):
        p = packet(self.I1); p["cand_ids"][-1] = self.F3["I1"]["excluded_composition_ids"][0]
        self.assertRejected(p, "differ from the fold universe")

    def test_reject_one_wrong_truth_index(self):
        p = packet(self.I1); p["unseen_truth"][0] = (p["unseen_truth"][0] + 1) % len(p["cand_ids"])
        self.assertRejected(p, "unseen_truth differs.*1 mismatches")

    def test_reject_truncated_rows(self):
        p = packet(self.I1); p["seen"] = p["seen"][:-1]
        self.assertRejected(p, "seen scores shape")

    def test_reject_nan_score(self):
        p = packet(self.I1); p["unseen"][3, 7] = np.nan
        self.assertRejected(p, "non-finite")

    def test_reject_integer_scores(self):
        p = packet(self.I1); p["seen"] = p["seen"].astype(np.int32)
        self.assertRejected(p, "dtype")

    def test_reject_extra_array(self):
        p = packet(self.I1); p["note"] = np.zeros(1)
        self.assertRejected(p, "packet arrays")

    def test_reject_wrong_truth_on_legacy_fold(self):
        f = next(f for f in self.b[2]["folds"] if f["role"] == "inner"); c = fold_contract(self.frame, f)
        p = packet(c); p.pop("cand_ids"); p["seen_truth"][0] = (p["seen_truth"][0] + 1) % 323
        self.assertRejected(p, "seen_truth differs", con=c)

    def test_scorer_from_scores_rejects_misaligned_truth_on_derived_fold(self):
        """The bug class found in the v3 stress script: truth taken from the parent fold after deriving."""
        from research_r2.candidates import derive_fold
        from research_r2.runner import scorer_from_scores
        f = self.F3["I1"]
        src = self.frame.source.loc[f["fitting_rows"]]
        f2 = derive_fold(self.frame, f, [r for r, s in zip(f["fitting_rows"], src) if s != src.iloc[0]])
        c2 = fold_contract(self.frame, f2); p = packet(c2)
        scorer_from_scores(self.frame, f2, p["seen"], p["seen_truth"], p["unseen"], p["unseen_truth"], p["cand_ids"])
        if c2["cand_ids"].tolist() != self.I1["cand_ids"].tolist():
            with self.assertRaises(PacketError):
                scorer_from_scores(self.frame, f2, p["seen"], p["seen_truth"], p["unseen"], self.I1["unseen_truth"],
                                   p["cand_ids"])


class TestConfigurationIdentity(unittest.TestCase):
    def setUp(self):
        self.proto = json.loads(PROTO.read_text())

    def test_identity_ignores_descriptive_fields_only(self):
        r = {"id": "x", "backbones": ["clip_l14"], "recipe": "factorised", "hypothesis": "h", "control": "c"}
        self.assertEqual(config_sha256(r), config_sha256(dict(r, id="y", hypothesis="other", control=None)))
        self.assertNotEqual(config_sha256(r), config_sha256(dict(r, cfg={"fuse": 0.5})))
        self.assertNotEqual(config_sha256(r), config_sha256(dict(r, tta_hflip=True)))
        self.assertEqual(row_config(r)["cfg"], {})

    def test_tree_rows_may_be_a_subset_of_the_matrix(self):
        e = self.proto["phase2_screen"]
        rows = tree_rows(e["tree"], e["start"])
        d = declared_rows(e, rows)
        self.assertEqual(set(d), set(rows))
        matrix = json.loads((ROOT / next(iter(e["matrix"]))).read_text())
        self.assertLessEqual(set(rows), {r["id"] for r in matrix})

    def test_reject_row_missing_from_declared_matrices(self):
        with self.assertRaisesRegex(PacketError, "not declared"):
            declared_rows(self.proto["phase3_representation"], ["F10_concat_convnext", "D1_synthComp_norm"])

    def test_reject_wrong_matrix_hash(self):
        e = copy.deepcopy(self.proto["phase2_screen"]); k = next(iter(e["matrix"])); e["matrix"][k] = "0" * 64
        with self.assertRaisesRegex(PacketError, "declared"):
            declared_rows(e, [e["start"]])

    def test_reject_conflicting_configuration_across_matrices(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "research_r2/matrices") as td:
            src = json.loads((ROOT / "research_r2/matrices/stage_f_v3.json").read_text())
            alt = [dict(r, cfg={"fuse": 0.5}) if r["id"] == "D1_synthComp_norm" else r for r in src]
            p = Path(td) / "alt.json"; p.write_text(json.dumps(alt))
            e = {"matrices": {"research_r2/matrices/stage_f_v3.json": file_sha256(ROOT / "research_r2/matrices/stage_f_v3.json"),
                              str(p.relative_to(ROOT)): file_sha256(p)}}
            with self.assertRaisesRegex(PacketError, "conflicting configurations"):
                declared_rows(e, ["D1_synthComp_norm"])


@unittest.skipUnless(V3.exists(), "benchmark v3 not built")
class TestPacketProvenance(unittest.TestCase):
    """End-to-end input checks of scripts/r2_select.py on synthetic packets in a temporary directory."""

    @classmethod
    def setUpClass(cls):
        cls.frame = load_training_source(ROOT / "DPD_Small_train.csv")
        cls.bench = json.loads(V3.read_text())
        cls.folds = [f["name"] for f in cls.bench["folds"] if f["role"] == "inner"]
        cls.F = {f["name"]: f for f in cls.bench["folds"]}
        spec = importlib.util.spec_from_file_location("r2_select", ROOT / "scripts/r2_select.py")
        cls.sel = importlib.util.module_from_spec(spec); spec.loader.exec_module(cls.sel)
        mpath = "research_r2/matrices/stage_f_v3.json"
        cls.proto = {"benchmark_semantic_sha256": cls.bench["semantic_sha256"], "_sha256": "p" * 64,
                     "t": {"matrix": {mpath: file_sha256(ROOT / mpath)}, "start": "B1_clipL_factorised",
                           "tree": {"B1_clipL_factorised": ["D1_synthComp_norm"], "D1_synthComp_norm": []}}}
        cls.msha = file_sha256(ROOT / mpath)
        cls.mrows = {r["id"]: r for r in json.loads((ROOT / mpath).read_text())}

    def make_stage(self, td, name, seed=0, rows=("B1_clipL_factorised", "D1_synthComp_norm"), cfg_override=None,
                   matrix_sha=None):
        stage = Path(td) / name
        stage.mkdir(parents=True)
        with open(stage / "summaries.jsonl", "w") as fh:
            for rid in rows:
                r = self.mrows[rid]
                fh.write(json.dumps({"id": rid, "backbones": r["backbones"], "recipe": r["recipe"],
                                     "cfg": (cfg_override or {}).get(rid, r.get("cfg", {})),
                                     "matrix_sha256": matrix_sha or self.msha, "feature_identity": {"x": 1}}) + "\n")
        for i, rid in enumerate(rows):
            d = stage / "scores" / rid; d.mkdir(parents=True)
            for fn in self.folds:
                np.savez_compressed(d / f"{fn}_scores.npz", **packet(fold_contract(self.frame, self.F[fn]), seed + i))
        return stage / "scores"

    def registry_for(self, dirs):
        rows = {}
        for rid in ("B1_clipL_factorised", "D1_synthComp_norm"):
            res = resolve_packets(rid, self.folds, dirs)
            rows[rid] = {"config_sha256": config_sha256(self.mrows[rid]),
                         "folds": {fn: {"sha256": res[fn]["sha256"],
                                        "contract_sha256": fold_contract(self.frame, self.F[fn])["sha256"]}
                                   for fn in self.folds}}
        return {"benchmark_semantic_sha256": self.bench["semantic_sha256"], "protocol_sha256": "p" * 64,
                "_sha256": "r" * 64, "rows": rows}

    def check(self, dirs, registry):
        return self.sel.check_inputs(self.frame, self.bench, self.proto, "t", dirs, self.folds, registry)

    def test_valid_inputs_pass_and_identical_duplicates_are_recorded(self):
        with tempfile.TemporaryDirectory() as td:
            a = self.make_stage(td, "a"); b = Path(td) / "b"
            import shutil
            shutil.copytree(a.parent, b)
            paths, inputs = self.check([a, b / "scores"], self.registry_for([a]))
            self.assertEqual(inputs["provenance"], "registered")
            self.assertEqual(inputs["rows"]["D1_synthComp_norm"]["packets"]["I1"]["n_identical_copies"], 2)

    def test_reject_conflicting_duplicate_copy(self):
        with tempfile.TemporaryDirectory() as td:
            a = self.make_stage(td, "a"); b = self.make_stage(td, "b", seed=7)
            with self.assertRaisesRegex(PacketError, "conflicting packet copies"):
                self.check([a, b], None)
            with self.assertRaisesRegex(PacketError, "conflicting packet copies"):
                self.check([b, a], None)  # argument order no longer decides which copy is used

    def test_reject_substituted_packet_against_registry(self):
        with tempfile.TemporaryDirectory() as td:
            a = self.make_stage(td, "a"); reg = self.registry_for([a])
            np.savez_compressed(a / "D1_synthComp_norm" / "I3_scores.npz",
                                **packet(fold_contract(self.frame, self.F["I3"]), 99))
            with self.assertRaisesRegex(PacketError, "not the registered one"):
                self.check([a], reg)

    def test_reject_registered_configuration_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            a = self.make_stage(td, "a"); reg = self.registry_for([a])
            reg["rows"]["D1_synthComp_norm"]["config_sha256"] = config_sha256(dict(self.mrows["D1_synthComp_norm"], cfg={}))
            with self.assertRaisesRegex(PacketError, "registered configuration differs"):
                self.check([a], reg)

    def test_reject_runtime_record_with_different_configuration(self):
        with tempfile.TemporaryDirectory() as td:
            a = self.make_stage(td, "a", cfg_override={"D1_synthComp_norm": {"fuse": 0.5, "normalise": True}})
            with self.assertRaisesRegex(PacketError, "recorded configuration"):
                self.check([a], None)

    def test_reject_runtime_record_from_uncommitted_matrix(self):
        with tempfile.TemporaryDirectory() as td:
            a = self.make_stage(td, "a", matrix_sha="f" * 64)
            with self.assertRaisesRegex(PacketError, "not committed"):
                self.check([a], None)

    def test_reject_registry_for_another_protocol(self):
        with tempfile.TemporaryDirectory() as td:
            a = self.make_stage(td, "a"); reg = self.registry_for([a]); reg["protocol_sha256"] = "q" * 64
            with self.assertRaisesRegex(PacketError, "different protocol"):
                self.check([a], reg)

    def test_reject_missing_fold_packet(self):
        with tempfile.TemporaryDirectory() as td:
            a = self.make_stage(td, "a"); (a / "B1_clipL_factorised" / "I6_scores.npz").unlink()
            with self.assertRaisesRegex(PacketError, "packet missing"):
                self.check([a], None)


@unittest.skipUnless((EXP / "v3").exists() and V3.exists(), "saved v3 packets not available")
class TestRealPackets(unittest.TestCase):
    def test_all_registered_real_packets_pass(self):
        reg_path = ROOT / "artifacts/r2/v3/packet_registry_v3.json"
        if not reg_path.exists():
            self.skipTest("registry not yet created")
        frame = load_training_source(ROOT / "DPD_Small_train.csv")
        F = {f["name"]: f for f in json.loads(V3.read_text())["folds"]}
        reg = json.loads(reg_path.read_text())
        con = {fn: fold_contract(frame, F[fn]) for fn in reg["folds"]}
        for rid, r in reg["rows"].items():
            for fn, rec in r["folds"].items():
                self.assertEqual(rec["contract_sha256"], con[fn]["sha256"])
                for c in rec["copies"]:
                    self.assertEqual(file_sha256(EXP / c), rec["sha256"], c)
                validate_packet(np.load(EXP / rec["copies"][0]), con[fn])
        self.assertTrue(committed_matrix_index())


if __name__ == "__main__":
    unittest.main()
