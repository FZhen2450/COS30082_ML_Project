"""Amendment 3 v2 PROPOSAL: synthetic validation of the child-versus-parent gate (research_r2_v4/child_gate.py).
Synthetic score tables on the real fold contracts (candidate order, truth); no model, no image, no real child."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from research_r2.packets import fold_contract  # noqa: E402
from research_r2_v4 import child_gate as cg  # noqa: E402
from research_r2_v4 import evaluate as ev  # noqa: E402
from research_r2_v4.common import INNER, env, scorer  # noqa: E402

_E = {}


def E():
    if not _E:
        _E["frame"], _E["bench"], _E["F"] = env()
    return _E["frame"], _E["F"]


_CACHE = {}


_IDENT = {}


def identity_of(fn, fold=None):
    """The ordered evaluation identity of a registered fold (or of a modified copy), as child_inputs computes it."""
    from research_r2_v4 import child_inputs as ci
    frame, F = E()
    if fold is not None:
        return ci.eval_identity(frame, fold)
    if fn not in _IDENT:
        _IDENT[fn] = ci.eval_identity(frame, F[fn])
    return _IDENT[fn]


def table(name, boost, noise_seed, extra=None, folds=INNER, permute=None, seed=0, identity_folds=None):
    """Scores = N(0,1) noise (fixed by noise_seed and fold) + boost[k] (+ extra[k]) on the true column.
    Built once per content (RowTable construction is the slow part); each call returns its own named object carrying a
    synthetic VERIFIED identity record (what child_inputs.load_verified attaches to real tables)."""
    import copy
    key = (tuple(sorted(boost.items())), noise_seed, tuple(sorted((extra or {}).items())), tuple(folds), permute)
    if key not in _CACHE:
        _CACHE[key] = _build(boost, noise_seed, extra, folds, permute)
    t = copy.copy(_CACHE[key]); t.sc = dict(t.sc); t.name = name
    t.identity = {"verified": True, "row": name, "seed": seed,
                  "folds": {fn: (identity_folds or {}).get(fn, identity_of(fn)) for fn in folds}}
    return t


def _build(boost, noise_seed, extra=None, folds=INNER, permute=None):
    frame, F = E()
    sc = {}
    for j, fn in enumerate(folds):
        con = fold_contract(frame, F[fn]); k = len(con["cand_ids"])
        rng = np.random.default_rng(10_000 * noise_seed + j)
        s = rng.normal(size=(len(con["seen_truth"]), k)); u = rng.normal(size=(len(con["unseen_truth"]), k))
        b = boost[fn] + (extra or {}).get(fn, 0.0)
        s[np.arange(len(s)), con["seen_truth"]] += b; u[np.arange(len(u)), con["unseen_truth"]] += b
        cand, st, ut = con["cand_ids"], con["seen_truth"], con["unseen_truth"]
        if permute == fn:  # same scores, candidate columns in another order (truth re-indexed consistently)
            p = np.arange(k)[::-1]; inv = np.argsort(p)
            s, u, cand, st, ut = s[:, p], u[:, p], np.asarray(cand)[p], inv[st], inv[ut]
        sc[fn] = [scorer(frame, {"seen": s, "unseen": u, "seen_truth": st, "unseen_truth": ut, "cand_ids": cand, "fold": F[fn]})]
    return ev.RowTable("synthetic", sc, ["single"])


B = {f: 3.0 for f in INNER}


def seeds_of(name, boost, extra=None, seeds=(0, 1, 2), base=0, identity_folds=None):
    return {s: table(f"{name}/s{s}", boost, base + 7 * s, extra, seed=s, identity_folds=identity_folds) for s in seeds}


def same_label_swap(fn):
    """A copy of registered fold fn whose seen-val order exchanges two images with the same truth label."""
    frame, F = E()
    fold = dict(F[fn]); rows = list(fold["seen_val_rows"])
    comp = frame.comp_id.to_numpy()
    for i in range(len(rows)):
        j = next((j for j in range(i + 1, len(rows)) if comp[rows[j]] == comp[rows[i]]), None)
        if j is not None:
            rows[i], rows[j] = rows[j], rows[i]
            fold["seen_val_rows"] = rows
            return fold, (i, j)
    raise AssertionError("no same-label pair")


class TestPrimaryGate(unittest.TestCase):
    def test_null_child_fails_and_old_rule_passed_it(self):
        parent = seeds_of("J2", B)
        child = seeds_of("J5_null", B)  # identical scores, separate objects
        for mode in ("screen", "confirm"):
            g = cg.primary_gate(child, parent, INNER, mode=mode, reps=100)["gate"]
            self.assertFalse(g["pass"]); self.assertEqual(g["mean_gain_pp"], 0.0); self.assertEqual(g["folds_improved"], 0)
        # the registered (amendment 2) evaluation gates a child against D1: a weak D1 makes the null child "pass"
        import r2v4_stage_j as sj
        d1 = table("D1", {f: 1.0 for f in INNER}, 999)
        old = sj.nested_with_parent_rule({sj.D1: d1, "J5": child[0]}, sj.D1, ["J5"], INNER, {})
        self.assertTrue(old["meets_gate_thresholds"])

    def test_bootstrap_of_null_child_is_degenerate_at_zero(self):
        b = cg.primary_gate(seeds_of("c", B), seeds_of("p", B), INNER, mode="confirm", reps=60)["bootstrap"]
        self.assertEqual(b["ci95_pp"], [0.0, 0.0]); self.assertEqual(b["bootstrap_sign_proportion_le_0"], 1.0)
        self.assertNotIn("p_le_0", b); self.assertIn("not a p-value", b["note"])
        self.assertFalse(hasattr(cg, "holm"), "no multiplicity adjustment of a descriptive bootstrap proportion")

    def test_genuine_six_fold_improvement_can_pass(self):
        parent = seeds_of("J2", B)
        child = seeds_of("J5_good", B, extra={f: 1.5 for f in INNER})
        for mode in ("screen", "confirm"):
            out = cg.primary_gate(child, parent, INNER, mode=mode, reps=100)
            self.assertTrue(out["gate"]["pass"], out["gate"]); self.assertEqual(out["gate"]["folds_improved"], 6)
            self.assertGreater(out["bootstrap"]["ci95_pp"][0], 0)
            self.assertEqual(set(out["parent_per_fold"].values()), {"J2/s0"})

    def test_fold_count_threshold(self):
        parent = seeds_of("J2", B)
        five = cg.primary_gate(seeds_of("c5", B, extra={f: (1.5 if f != "I3" else -1.0) for f in INNER}), parent, INNER, "confirm", reps=40)["gate"]
        four = cg.primary_gate(seeds_of("c4", B, extra={f: (1.5 if f not in ("I3", "I4") else -1.0) for f in INNER}), parent, INNER, "confirm", reps=40)["gate"]
        self.assertEqual(five["folds_improved"], 5); self.assertEqual(four["folds_improved"], 4)
        self.assertFalse(four["pass"])
        self.assertEqual(five["pass"], five["mean_gain_pp"] >= 1.0)

    def test_ausuc_and_shortcut_constraints(self):
        ok = {s: {k: {"d": 0.02, "a": 0.0, "h": 0.0} for k in INNER} for s in (0, 1, 2)}
        self.assertTrue(cg.gate_from(ok, INNER)["pass"])
        low_ausuc = {s: {k: {"d": 0.02, "a": -0.001, "h": 0.0} for k in INNER} for s in (0, 1, 2)}
        self.assertFalse(cg.gate_from(low_ausuc, INNER)["pass"])
        shortcut = {s: {k: {"d": 0.02, "a": 0.0, "h": 0.051} for k in INNER} for s in (0, 1, 2)}
        self.assertFalse(cg.gate_from(shortcut, INNER)["pass"])
        # per-fold seed means: a fold counts as improved on its mean over seeds, not on any single seed
        mixed = {0: {k: {"d": 0.03, "a": 0, "h": 0} for k in INNER}, 1: {k: {"d": -0.02, "a": 0, "h": 0} for k in INNER},
                 2: {k: {"d": -0.02, "a": 0, "h": 0} for k in INNER}}
        self.assertEqual(cg.gate_from(mixed, INNER)["folds_improved"], 0)

    def test_incomplete_or_incompatible_seeds_fail(self):
        parent = seeds_of("J2", B)
        with self.assertRaisesRegex(cg.GateError, "exactly seeds"):
            cg.primary_gate(seeds_of("c", B, seeds=(0, 1)), parent, INNER, "confirm")
        with self.assertRaisesRegex(cg.GateError, "exactly seeds"):
            cg.primary_gate(seeds_of("c", B, seeds=(0, 1, 3)), parent, INNER, "confirm")
        with self.assertRaisesRegex(cg.GateError, "screen needs seed"):
            cg.primary_gate(seeds_of("c", B, seeds=(1,)), parent, INNER, "screen")
        with self.assertRaisesRegex(cg.GateError, "parent: confirmation"):
            cg.primary_gate(seeds_of("c", B), {0: parent[0], 1: parent[1]}, INNER, "confirm")

    def test_candidate_order_and_fold_identities(self):
        parent = seeds_of("J2", B)
        from research_r2.packets import PacketError
        with self.assertRaisesRegex(PacketError, "reordered"):  # a reordered packet cannot even be scored
            _build(B, 0, permute="I2")
        bad = {s: table(f"c{s}", B, 7 * s, seed=s) for s in (0, 1, 2)}
        import copy
        sc = copy.copy(bad[1].sc["I2"][0]); sc.cand_crop = sc.cand_crop[::-1].copy(); bad[1].sc["I2"] = [sc]
        with self.assertRaisesRegex(cg.GateError, "candidate order"):
            cg.primary_gate(bad, parent, INNER, "confirm")
        swapped = {s: table(f"c{s}", B, 7 * s, folds=["I2", "I1", "I3", "I4", "I5", "I6"], seed=s) for s in (0, 1, 2)}
        with self.assertRaisesRegex(cg.GateError, "fold identities"):
            cg.primary_gate(swapped, parent, INNER, "confirm")
        missing = {s: table(f"c{s}", B, 7 * s, folds=INNER[:5], seed=s) for s in (0, 1, 2)}
        with self.assertRaisesRegex(cg.GateError, "fold identities"):
            cg.primary_gate(missing, parent, INNER, "confirm")
        # a fold's data under another fold's key: truth differs -> refused
        relabel = table("x", B, 0)
        relabel.sc["I1"], relabel.sc["I2"] = relabel.sc["I2"], relabel.sc["I1"]
        with self.assertRaises(cg.GateError):
            cg.check_pair(relabel, parent[0], INNER)

    def test_same_label_image_swap_is_refused(self):
        """Exchanging two evaluation images with the same truth label leaves every label array equal but changes the
        ordered image identity: refused."""
        frame, F = E()
        parent = seeds_of("J2", B)
        swapped_fold, (i, j) = same_label_swap("I3")
        from research_r2.packets import fold_contract
        a, b = fold_contract(frame, F["I3"]), fold_contract(frame, swapped_fold)
        self.assertTrue(np.array_equal(a["seen_truth"], b["seen_truth"]) and a["sha256"] == b["sha256"])  # labels / contract blind to it
        child = seeds_of("c", B, identity_folds={"I3": identity_of("I3", swapped_fold)})
        self.assertTrue(np.array_equal(child[0].sc["I3"][0].parts["seen"]["truth"], parent[0].sc["I3"][0].parts["seen"]["truth"]))
        with self.assertRaisesRegex(cg.GateError, "image identity / order differs"):
            cg.primary_gate(child, parent, INNER, "confirm")

    def test_unverified_or_misidentified_inputs_are_refused(self):
        parent = seeds_of("J2", B)
        raw = seeds_of("c", B)
        del raw[0].identity
        with self.assertRaisesRegex(cg.GateError, "unverified input"):
            cg.primary_gate(raw, parent, INNER, "screen")
        forged = seeds_of("c", B); forged[0].identity = dict(forged[0].identity, verified="yes")
        with self.assertRaisesRegex(cg.GateError, "unverified input"):
            cg.primary_gate(forged, parent, INNER, "screen")
        wrong_seed = seeds_of("c", B); wrong_seed[0], wrong_seed[1] = wrong_seed[1], wrong_seed[0]
        with self.assertRaisesRegex(cg.GateError, "identified as seed"):
            cg.primary_gate(wrong_seed, parent, INNER, "confirm")
        other_fold = seeds_of("c", B, identity_folds={"I4": identity_of("I5")})
        with self.assertRaisesRegex(cg.GateError, "names another fold"):
            cg.primary_gate(other_fold, parent, INNER, "screen")

    def test_held_fold_does_not_influence_its_own_calibration(self):
        parent = table("J2", B, 0)
        for k in INNER:
            alt = table("J2", B, 0, extra={k: 4.0})  # change ONLY the held fold
            self.assertEqual(cg.fold_values(parent, k, INNER)["gi"], cg.fold_values(alt, k, INNER)["gi"], k)


def switching_parents():
    """J2 and J3 share their noise; J3 is slightly weaker everywhere except I1, where it is far stronger.
    -> pi(I1) = J2 (I1 excluded from its own choice), pi(k) = J3 for every other held fold."""
    j2 = {f: 3.0 for f in INNER}
    j3 = {f: (8.0 if f == "I1" else 2.9) for f in INNER}
    return {"J1": seeds_of("J1", {f: 1.0 for f in INNER}, base=100), "J2": seeds_of("J2", j2, base=200), "J3": seeds_of("J3", j3, base=200)}, j2, j3


class TestPipelineEstimate(unittest.TestCase):
    ORDER = ["J1", "J2", "J3"]

    def test_held_fold_does_not_choose_its_parent(self):
        P, _, _ = switching_parents()
        pi = cg.inner_parents({p: P[p][0] for p in self.ORDER}, INNER, self.ORDER)
        self.assertEqual(pi["I1"], "J2"); self.assertEqual({pi[k] for k in INNER if k != "I1"}, {"J3"})
        for k in INNER:  # making J1 perfect on the held fold only never changes that fold's parent
            Q = dict(P); Q["J1"] = {0: table("J1", {f: (50.0 if f == k else 1.0) for f in INNER}, 100)}
            self.assertEqual(cg.inner_parents({p: Q[p][0] for p in self.ORDER}, INNER, self.ORDER)[k], pi[k])

    def test_missing_parent_variant_fails_closed(self):
        P, j2, _ = switching_parents()
        with self.assertRaisesRegex(cg.GateError, "missing child variant trained on parent J3"):
            cg.pipeline_estimate({"J2": seeds_of("c_on_J2", j2, base=200)}, P, INNER, self.ORDER, "confirm")

    def test_parent_switching_is_not_credited_to_the_child(self):
        P, j2, j3 = switching_parents()
        null = {"J2": seeds_of("c_on_J2", j2, base=200), "J3": seeds_of("c_on_J3", j3, base=200)}
        d1 = table("D1", {f: 1.0 for f in INNER}, 999)
        out = cg.pipeline_estimate(null, P, INNER, self.ORDER, "confirm", d1=d1)
        diff = {(s, k): (d["child_row"], d["parent_row"], d["child"], d["parent"]) for s, v in out["detail"].items() for k, d in v.items()
                if d["child"]["E"] != d["parent"]["E"]}
        self.assertEqual(out["gate"]["mean_gain_pp"], 0.0, f"{out['inner_parent']} {list(diff.items())[:2]}"); self.assertFalse(out["gate"]["pass"])
        dec = out["decomposition_vs_D1"]
        self.assertTrue(all(v["child_attributable_pp"] == 0.0 for v in dec.values()))
        self.assertTrue(all(v["parent_attributable_pp"] > 0 for v in dec.values()))  # parent gains stay with the parent
        # a child that helps only on J2 is credited only where J2 is the held fold's inner parent (I1)
        part = {"J2": seeds_of("c_on_J2", j2, extra={f: 1.5 for f in INNER}, base=200), "J3": seeds_of("c_on_J3", j3, base=200)}
        o2 = cg.pipeline_estimate(part, P, INNER, self.ORDER, "confirm", d1=d1)
        self.assertGreater(o2["gate"]["per_fold_delta_pp"]["I1"], 0)
        self.assertTrue(all(o2["gate"]["per_fold_delta_pp"][k] == 0.0 for k in INNER if k != "I1"))
        self.assertEqual(o2["gate"]["folds_improved"], 1); self.assertFalse(o2["gate"]["pass"])

    def test_fallback_gain_is_never_counted_for_the_child(self):
        """The registered stage-nested rule falls back to a parent where a child is ineligible; the proposed estimate
        has no fallback: the gate is computed only from child-minus-its-own-parent differences."""
        import r2v4_stage_j as sj
        P, j2, j3 = switching_parents()
        perfect = table("J5", {f: 50.0 for f in INNER}, 500)
        T = {sj.D1: table("D1", {f: 1.0 for f in INNER}, 999), "J1_clipL_last2": P["J1"][0], "J2_clipL_last4": P["J2"][0],
             "J3_clipL_lora8_mlp": P["J3"][0], "J5_best_source_balanced": perfect}
        old = sj.nested_with_parent_rule(T, sj.D1, list(T)[1:], INNER, {"J5_best_source_balanced": "J2_clipL_last4"})
        fallback = [h["held_fold"] for h in old["held"] if h["selected"] != "J5_best_source_balanced"]
        self.assertTrue(fallback)  # the old stage estimate includes parent-only folds
        new = cg.pipeline_estimate({"J2": {0: perfect}, "J3": {0: perfect}}, {p: {0: P[p][0]} for p in self.ORDER}, INNER, self.ORDER, "screen")
        for k in INNER:
            d = new["detail"][0][k]
            self.assertTrue(d["child_row"] == "J5" and d["parent_row"] != "J5")
        self.assertEqual(set(new["inner_parent"].values()), {"J2", "J3"})


class TestPrimaryEndpointLock(unittest.TestCase):
    def test_alternative_endpoint_cannot_gate_or_promote(self):
        child, parent = seeds_of("c", B, extra={f: 1.5 for f in INNER}), seeds_of("p", B)
        for e in ("unseen_disease_top1", "unseen_crop_top1", "seen_top1", "accuracy"):
            with self.assertRaisesRegex(cg.GateError, "primary endpoint is H_bal"):
                cg.primary_gate(child, parent, INNER, "screen", endpoint=e)
        diag = cg.diagnostics(child, parent, INNER, "screen", endpoints=("unseen_disease_top1",))
        self.assertGreater(diag["endpoints"]["unseen_disease_top1"]["mean_delta_pp"], 0)
        self.assertNotIn("pass", str(diag.keys())); self.assertNotIn("gate", diag)
        with self.assertRaisesRegex(cg.GateError, "only pass on the primary H_bal gate"):
            cg.promotion_decision(diag)
        with self.assertRaisesRegex(cg.GateError, "only pass on the primary H_bal gate"):
            cg.promotion_decision(dict(cg.primary_gate(child, parent, INNER, "screen", reps=20), endpoint="unseen_disease_top1"))
        with self.assertRaisesRegex(cg.GateError, "not a diagnostic"):
            cg.diagnostics(child, parent, INNER, "screen", endpoints=("H_bal",))
        self.assertTrue(cg.promotion_decision(cg.primary_gate(child, parent, INNER, "screen", reps=20))["pass"])

    def test_secondary_estimate_cannot_promote(self):
        P, j2, j3 = switching_parents()
        good = {"J2": seeds_of("c2", j2, extra={f: 1.5 for f in INNER}, base=200), "J3": seeds_of("c3", j3, extra={f: 1.5 for f in INNER}, base=200)}
        out = cg.pipeline_estimate(good, P, INNER, ["J1", "J2", "J3"], "confirm")
        self.assertEqual(out["role"], "secondary")
        with self.assertRaises(cg.GateError):
            cg.promotion_decision(out)

    def test_calibration_matched_sensitivity_is_diagnostic(self):
        diag = cg.diagnostics(seeds_of("c", B), seeds_of("p", B), INNER, "screen", endpoints=())
        self.assertEqual(diag["calibration_matched_H_bal"]["mean_delta_pp"], 0.0); self.assertEqual(diag["role"], "diagnostic")


if __name__ == "__main__":
    unittest.main()
