"""Seed replication contracts; synthetic CPU only, no real notebook experiment."""
import ast
import copy
import io
import json
import math
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
import matplotlib.pyplot as plt
import test_wp3b_m4 as m4

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/wp3b_seed_replication.ipynb"


def namespace():
    if not NOTEBOOK.exists():
        raise AssertionError("The approved self-contained replication notebook is not implemented yet")
    nodes, values = [], {}
    for cell in json.loads(NOTEBOOK.read_text())["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                nodes.append(node)
            elif isinstance(node, ast.Assign):
                try:
                    value = ast.literal_eval(node.value)
                except (TypeError, ValueError):
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        values[target.id] = value
    ns = {**m4.namespace(), **values, "ROOT": ROOT, "__name__": __name__, "plt": plt}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<visible replication cells>", "exec"), ns)
    return ns


class SeedContractTests(unittest.TestCase):
    def setUp(self):
        self.ns = namespace()
        self.config = copy.deepcopy(self.ns["CONFIG"])

    def test_only_predeclared_training_seeds_and_fixed_split_are_accepted(self):
        self.assertTrue(self.ns["validate_replication_config"](self.config))
        for key, value in (("split_seed", 43), ("training_seeds", [42, 43]),
                           ("training_seeds", [44, 43]), ("epochs", 6),
                           ("conditional_weight", 0.), ("learning_rate", .001)):
            with self.subTest(key=key, value=value):
                bad = {**self.config, key: value}
                with self.assertRaises(ValueError):
                    self.ns["validate_replication_config"](bad)
        with self.assertRaises(ValueError):
            self.ns["validate_replication_config"]({**self.config, "seed": 43})

    def test_same_seed_arms_start_identically_and_other_seed_changes_initialization(self):
        a = self.ns["initialize_replication_adapter"](43, None)
        b = self.ns["initialize_replication_adapter"](43, .2)
        c = self.ns["initialize_replication_adapter"](44, .2)
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters())))
        self.assertFalse(torch.equal(a.down.weight, c.down.weight))
        x = torch.randn(2, 768)
        torch.testing.assert_close(a(x), x, rtol=0, atol=0)
        torch.testing.assert_close(b(x), x, rtol=0, atol=0)
        for seed in (42, True, 45):
            with self.assertRaises(ValueError):
                self.ns["initialize_replication_adapter"](seed, .2)

    def test_contract_separates_training_seed_and_preserves_split_candidates_prompts(self):
        split = self.ns["build_development_split"](ROOT / "DPD_Small_train.csv", seed=42)
        table = self.ns["build_prompt_table"](split)
        first = self.ns["make_replication_contract"](split, table, {**self.config, "training_seed": 43, "residual_cap": None})
        second = self.ns["make_replication_contract"](split, table, {**self.config, "training_seed": 44, "residual_cap": .2})
        self.assertEqual(first["split"], second["split"])
        self.assertEqual(first["candidate_ids"], second["candidate_ids"])
        self.assertEqual(len(first["candidate_ids"]), 323)
        self.assertNotEqual(first["candidate_ids"], list(range(323)))
        self.assertEqual(first["direct_prompt_sha256"], "753672c7964bcb20a1346145f21a72b40a19f55590a006b5783823f23c13f537")
        self.assertEqual(first["hyperparameters"]["split_seed"], 42)
        self.assertEqual(second["hyperparameters"]["training_seed"], 44)
        self.assertNotEqual(self.ns["json_hash"](first), self.ns["json_hash"](second))

    def test_canonical_objective_scoring_selection_and_adapter_AST_match_parent(self):
        def defs(path):
            return {n.name: ast.dump(n) for c in json.loads(path.read_text())["cells"] if c["cell_type"] == "code"
                    for n in ast.parse("".join(c["source"])).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
        parent, new = defs(ROOT / "notebooks/wp3b_bounded_cached.ipynb"), defs(NOTEBOOK)
        for name in ("ResidualVisualAdapter", "adapter_objective", "m3_training_objective", "same_crop_loss",
                     "cosine_candidate_scores", "rank_global_ids", "harmonic_mean", "selection_record",
                     "better_selection", "train_adapter_epoch", "validate_adapter", "save_adapter_checkpoint", "load_adapter_checkpoint"):
            with self.subTest(name=name):
                self.assertEqual(new[name], parent[name])


class ReplicationStructureTests(unittest.TestCase):
    def test_json_syntax_default_guards_and_sanitized_outputs(self):
        ns = namespace()
        book = json.loads(NOTEBOOK.read_text())
        self.assertEqual(len(book["cells"]), len({c["id"] for c in book["cells"]}))
        for flag in ("LOAD_FEATURE_CACHE", "RUN_REPLICATIONS", "BUILD_REPLICATION_REPORT"):
            self.assertIs(ns[flag], False)
        for cell in book["cells"]:
            self.assertFalse(cell.get("outputs"))
            self.assertIsNone(cell.get("execution_count"))
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), cell["id"], "exec")

    def test_no_hidden_project_processing_download_or_official_test_access(self):
        namespace()
        book = json.loads(NOTEBOOK.read_text())
        for cell in book["cells"]:
            if cell["cell_type"] != "code":
                continue
            text = "".join(cell["source"])
            self.assertNotIn("DPD_Small_test_", text)
            for node in ast.walk(ast.parse(text)):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn((node.module or "").split(".")[0],
                                     ("dataset", "development_data", "train_multitask", "evaluation", "test_wp3b_m4"))
                if isinstance(node, ast.Call):
                    self.assertNotIn(ast.unparse(node.func), ("exec", "eval", "compile", "clip.load", "subprocess.run", "torch.hub.load"))

    def test_cold_default_run_all_never_loads_cache_models_or_starts_training(self):
        namespace()
        ns = {"__name__": __name__, "display": lambda *args, **kwargs: None}
        with patch("clip.load", side_effect=AssertionError("no model")), \
             patch("torch.load", side_effect=AssertionError("no cache/checkpoint")), \
             patch("torch.save", side_effect=AssertionError("no save")), \
             patch.object(Path, "write_text", side_effect=AssertionError("no writes")), \
             patch("torch.optim.AdamW.step", side_effect=AssertionError("no optimizer")), \
             m4.reference.reference.redirect_stdout(io.StringIO()):
            for cell in json.loads(NOTEBOOK.read_text())["cells"]:
                if cell["cell_type"] == "code":
                    exec(compile("".join(cell["source"]), cell["id"], "exec"), ns)
        self.assertIsNone(ns["FEATURE_CACHE"])
        self.assertIsNone(ns["REPLICATION_RESULTS"])
        self.assertFalse(ns["REPLICATION_STARTED"])


class CachedLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = namespace()
        m4.CacheTests.setUpClass()
        cls.split, cls.cache = m4.CacheTests.split, m4.CacheTests.cache

    def test_fitting_order_is_paired_deterministic_and_validation_keeps_original_ids(self):
        ns = self.ns
        def names(seed, cap, role):
            config = {**ns["CONFIG"], "device": "cpu", "training_seed": seed, "residual_cap": cap}
            loader = ns["make_replication_loader"](self.split, role, config, self.cache)
            return [name for _, targets in loader for name in targets["image_name"]]
        first = names(43, None, "fit")
        self.assertEqual(first, names(43, .2, "fit"))
        self.assertNotEqual(first, names(44, .2, "fit"))
        self.assertEqual(set(first), set(self.split.fit.image_name))
        self.assertEqual(names(44, .2, "val_unseen"), self.split.val_unseen.image_name.tolist())

    def test_checkpoint_roundtrip_keeps_both_seed_identities_and_logits(self):
        ns = self.ns
        config = {**ns["CONFIG"], "device": "cpu", "training_seed": 43, "residual_cap": .2}
        table = ns["build_prompt_table"](self.split)
        contract = ns["make_replication_contract"](self.split, table, config)
        proto = ns["component_prototypes"](self.cache["direct"], table, contract["fitting_composition_ids"])
        state = ns["seal_text_state"](self.cache["direct"], proto, contract)
        adapter = ns["initialize_replication_adapter"](43, .2)
        with m4.reference.reference.workspace_temporary_directory() as directory:
            path = Path(directory) / "adapter.pt"
            ns["save_adapter_checkpoint"](path, adapter, state, contract, {"epoch": 0})
            restored, text, _ = ns["load_adapter_checkpoint"](path, contract)
            x = torch.randn(2, 768)
            torch.testing.assert_close(ns["cosine_candidate_scores"](adapter(x), state["direct"]),
                                       ns["cosine_candidate_scores"](restored(x), text["direct"]), rtol=0, atol=0)
            wrong = copy.deepcopy(contract)
            wrong["hyperparameters"]["training_seed"] = 44
            with self.assertRaises(ValueError):
                ns["load_adapter_checkpoint"](path, wrong)


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.ns = namespace()

    def table(self):
        rows = []
        for i, seed in enumerate((42, 43, 44)):
            for arm, offset in (("control", 0.), ("bounded", .02)):
                value = (.1, .2, .3)[i] + offset
                rows.append({"training_seed": seed, "arm": arm, "h1": value, "h5": value,
                             "seen_top1": value, "unseen_top1": value,
                             "unseen_diseased_accuracy": value, "unseen_healthy_accuracy": value,
                             "false_healthy_rate": value})
        return self.ns["pd"].DataFrame(rows)

    def test_three_seed_sample_standard_deviation_and_paired_differences(self):
        result = self.ns["replication_statistics"](self.table())
        stats = result["descriptive_statistics"]
        control = stats.loc[(stats.arm == "control") & (stats.metric == "h1")].iloc[0]
        self.assertEqual(control["n"], 3)
        self.assertAlmostEqual(control["mean"], .2)
        self.assertAlmostEqual(control["sample_std"], .1)
        self.assertEqual(result["paired_differences"].training_seed.tolist(), [42, 43, 44])
        self.assertTrue(all(abs(v - .02) < 1e-15 for v in result["paired_differences"].h1_delta))

    def test_missing_duplicate_and_nonfinite_records_are_rejected(self):
        for table in (self.table().iloc[:-1], self.ns["pd"].concat([self.table(), self.table().iloc[:1]])):
            with self.assertRaises(ValueError):
                self.ns["replication_statistics"](table)
        table = self.table()
        table.loc[0, "h1"] = float("nan")
        with self.assertRaises(ValueError):
            self.ns["replication_statistics"](table)

    def test_record_validator_rejects_partial_budget_and_changed_selected_epoch(self):
        split = self.ns["build_development_split"](ROOT / "DPD_Small_train.csv")
        summary = synthetic_summary(self.ns, split)
        self.assertTrue(self.ns["validate_replication_summary"](summary, split, historical=True))
        bad = copy.deepcopy(summary)
        bad["history"][1]["train_sample_count"] = 4867
        with self.assertRaises(ValueError):
            self.ns["validate_replication_summary"](bad, split, historical=True)
        bad = copy.deepcopy(summary)
        bad["selected"]["epoch"] = 5
        with self.assertRaises(ValueError):
            self.ns["validate_replication_summary"](bad, split, historical=True)

    def test_existing_failed_seed42_gate_cannot_be_pooled_away(self):
        decisions = [{"training_seed": seed, "decision": {"supports_exploratory_success": seed != 42}}
                     for seed in (42, 43, 44)]
        result = self.ns["summarize_gate_decisions"](decisions)
        self.assertFalse(result["all_seeds_pass"])
        self.assertEqual(result["failed_training_seeds"], [42])
        self.assertEqual(result["passing_training_seeds"], [43, 44])


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.ns = namespace()
        m4.CacheTests.setUpClass()
        self.split, self.cache = m4.CacheTests.split, m4.CacheTests.cache
        self.table = self.ns["build_prompt_table"](self.split)

    def test_all_four_contracts_exist_before_first_arm_and_initialization_is_paired(self):
        calls = []
        with m4.reference.reference.workspace_temporary_directory() as directory:
            cfg = {**self.ns["CONFIG"], "device": "cpu", "output_directory": directory}
            def fake(split, table, arm, cache, contract, folder, adapter):
                group = Path(folder).parent
                declaration = json.loads((group / "replication_declaration.json").read_text())
                self.assertEqual(len(declaration["arms"]), 4)
                self.assertTrue(all((Path(item["directory"]) / "declaration.json").is_file() for item in declaration["arms"]))
                calls.append((arm["training_seed"], arm["residual_cap"], adapter.down.weight.detach().clone()))
                summary = synthetic_summary(self.ns, split, arm, contract)
                path = Path(folder) / "adapter_summary.json"
                path.write_text(json.dumps(summary))
                return {"training_seed": arm["training_seed"], "arm": "control" if arm["residual_cap"] is None else "bounded",
                        "directory": str(folder), "summary_path": str(path), "summary_sha256": self.ns["sha256_file"](path)}
            with patch.dict(self.ns, {"verify_replication_cache": lambda *a: True, "run_replication_arm": fake}):
                result = self.ns["run_replications"](self.split, self.table, cfg, self.cache)
                self.assertTrue((Path(result["group_directory"]) / "completed.json").is_file())
                with self.assertRaises(RuntimeError):
                    self.ns["run_replications"](self.split, self.table, cfg, self.cache)
        self.assertEqual([(a, b) for a, b, _ in calls], [(43, None), (43, .2), (44, None), (44, .2)])
        torch.testing.assert_close(calls[0][2], calls[1][2], rtol=0, atol=0)
        torch.testing.assert_close(calls[2][2], calls[3][2], rtol=0, atol=0)
        self.assertFalse(torch.equal(calls[0][2], calls[2][2]))

    def test_interruption_preserves_failure_and_never_marks_group_complete(self):
        with m4.reference.reference.workspace_temporary_directory() as directory:
            cfg = {**self.ns["CONFIG"], "device": "cpu", "output_directory": directory}
            def stop(*args):
                raise KeyboardInterrupt("synthetic interruption")
            with patch.dict(self.ns, {"verify_replication_cache": lambda *a: True, "run_replication_arm": stop}):
                with self.assertRaises(KeyboardInterrupt):
                    self.ns["run_replications"](self.split, self.table, cfg, self.cache)
            groups = list(Path(directory).glob("*/replication_declaration.json"))
            self.assertEqual(len(groups), 1)
            self.assertTrue((groups[0].parent / "failed.json").is_file())
            self.assertFalse((groups[0].parent / "completed.json").exists())


def synthetic_summary(ns, split, config=None, contract=None):
    """Complete scalar-only fixture; it never claims image/model execution."""
    config = dict(config or {**ns["CONFIG"], "seed": 42, "residual_cap": None})
    if config.get("seed") == 42:
        config.pop("training_seeds", None)
        config.pop("split_seed", None)
    row = {"epoch": 0, "h1": 0., "h5": 0., "seen_top1": 0., "seen_top5": 0.,
           "unseen_top1": 0., "unseen_top5": 0., "seen_count": 1238, "unseen_count": 1894,
           "combined_validation_loss": 1., "val_seen_loss": 1., "val_unseen_loss": 1., "train_sample_count": None}
    history = [{**row, "epoch": i, "train_sample_count": None if i == 0 else 4868} for i in range(6)]
    selected = {k: v for k, v in row.items() if k not in ("val_seen_loss", "val_unseen_loss", "train_sample_count")}
    selected["selection_policy"] = copy.deepcopy(ns["SELECTION_POLICY"])
    table = ns["build_prompt_table"](split)
    protocol = {"split": ns["verify_split_identity"](split), "candidate_ids": list(split.candidate_ids),
                "candidate_pairs": [list(p) for p in split.candidate_pairs], "direct_prompt_sha256": ns["EXPECTED_DIRECT_PROMPT_SHA256"],
                "model": {"name": "ViT-L/14", "revision": ns["CLIP_REVISION"], **ns["weight_spec"]("ViT-L/14")},
                "preprocessing": ns["NATIVE_PREPROCESSING"], "code": {"fixture": True}, "conditional_objective": ns["M3_SPEC"]}
    if contract is not None:
        protocol = {k: contract[k] for k in protocol}
    diag = {}
    for role, n, healthy in (("val_seen", 1238, 238), ("val_unseen", 1894, 398)):
        diag[role] = {"count": n, "correct1": 0, "correct5": 0, "top1": 0., "top5": 0.,
                      "healthy_count": healthy, "healthy_correct": 0, "diseased_count": n - healthy,
                      "diseased_correct": 0, "false_healthy": 0, "predicted_fitting_composition": 0}
    return {"selected": selected, "history": history, "diagnostics": diag, "config": config, "protocol": protocol,
            "contract_sha256": ns["json_hash"](contract) if contract is not None else "0" * 64}


class AdditionalBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.ns = namespace()

    def test_fractional_integer_budget_and_boolean_cap_rejected(self):
        cfg = self.ns["CONFIG"]
        for key, value in (("epochs", 5.), ("batch_size", 8.), ("num_workers", 0.),
                           ("training_seed", True), ("residual_cap", False)):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    self.ns["validate_replication_config"]({**cfg, key: value})

    def test_negative_recorded_cross_entropy_and_boolean_selected_epoch_rejected(self):
        split = self.ns["build_development_split"](ROOT / "DPD_Small_train.csv")
        negative = synthetic_summary(self.ns, split)
        for row in negative["history"]:
            row["val_seen_loss"], row["val_unseen_loss"] = -1., 3.
            row["combined_validation_loss"] = (-1238 + 1894 * 3) / 3132
        negative["selected"]["combined_validation_loss"] = negative["history"][0]["combined_validation_loss"]
        boolean = synthetic_summary(self.ns, split)
        boolean["selected"]["epoch"] = False
        for summary in (negative, boolean):
            with self.assertRaises(ValueError):
                self.ns["validate_replication_summary"](summary, split, historical=True)

    def test_reopened_report_rejects_missing_or_tampered_interpretation(self):
        ns = self.ns
        table = ns["pd"].DataFrame({"value": [.1, .2]})
        report = {key: table.copy() for key in ("selected_table", "descriptive_statistics", "paired_differences",
                                               "per_pair", "epoch_subgroups", "trajectories")}
        report.update(gate_decisions=[{"seed": 42, "pass": False}], gate_summary={"all_seeds_pass": False},
                      interpretation="Synthetic report fixture")
        with m4.reference.reference.workspace_temporary_directory() as directory:
            with patch.dict(ns, {"load_historical_reference": lambda *a: [], "read_completed_replication": lambda *a: [],
                                 "build_replication_report": lambda *a: report}):
                ns["load_and_report_replications"](directory, None, None, ns["CONFIG"])
                path = Path(directory) / "replication_interpretation.json"
                path.write_text(json.dumps({"gate_summary": {"all_seeds_pass": True}}))
                with self.assertRaises(ValueError):
                    ns["load_and_report_replications"](directory, None, None, ns["CONFIG"])
                path.unlink()
                with self.assertRaises(ValueError):
                    ns["load_and_report_replications"](directory, None, None, ns["CONFIG"])

    def test_new_contract_runs_bounded_synthetic_backward_only_on_adapter(self):
        ns = self.ns
        m4.CacheTests.setUpClass()
        split, cache = m4.CacheTests.split, m4.CacheTests.cache
        config = {**ns["CONFIG"], "device": "cpu", "training_seed": 43, "residual_cap": .2}
        table = ns["build_prompt_table"](split)
        contract = ns["make_replication_contract"](split, table, config)
        proto = ns["component_prototypes"](cache["direct"], table, contract["fitting_composition_ids"])
        state = ns["seal_text_state"](cache["direct"], proto, contract)
        frame = split.fit.iloc[:3].copy()
        features = torch.randn(3, 768)
        class Tiny(torch.utils.data.Dataset):
            role = "fit"
            def __len__(self): return 3
            def __getitem__(self, i):
                row = frame.iloc[i]
                return features[i], {"crop": int(row.crop_id), "disease": int(row.disease_id),
                                     "composition": int(row.composition_id), "image_name": row.image_name}
        adapter = ns["initialize_replication_adapter"](43, .2)
        optimizer = ns["make_adapter_optimizer"](adapter, config)
        original = adapter.up.weight.detach().clone()
        frozen = ns["CachedFeatureBackbone"]().eval()
        with patch.object(optimizer, "step", wraps=optimizer.step) as step:
            result = ns["train_adapter_epoch"](frozen, adapter, torch.utils.data.DataLoader(Tiny(), batch_size=2),
                                               optimizer, state, frame, "cpu", False)
        self.assertEqual(step.call_count, 2)
        self.assertEqual(result["sample_count"], 3)
        self.assertFalse(torch.equal(original, adapter.up.weight))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters()))
        self.assertEqual(list(frozen.parameters()), [])


class CompletedGroupTests(unittest.TestCase):
    """Actual serializer/reader integration using synthetic predictions; zero training steps."""
    @classmethod
    def setUpClass(cls):
        cls.ns = namespace()
        m4.CacheTests.setUpClass()
        cls.split, cls.cache = m4.CacheTests.split, m4.CacheTests.cache
        cls.table = cls.ns["build_prompt_table"](cls.split)
        cls.temporary = m4.reference.reference.workspace_temporary_directory()
        cls.root = Path(cls.temporary.name)
        cls.config = {**cls.ns["CONFIG"], "device": "cpu", "output_directory": str(cls.root)}
        cls.group = cls.root / "synthetic_complete_group"
        cls.group.mkdir()
        cls.records, declarations = [], []
        candidates = list(cls.split.candidate_ids)
        validation = {}
        for role in ("val_seen", "val_unseen"):
            frame = getattr(cls.split, role)
            ranks = torch.tensor([[int(v)] + [x for x in candidates if x != int(v)][:4] for v in frame.composition_id])
            validation[role] = {"sample_count": len(frame), "loss": 1., "crop_loss": 1., "disease_loss": 1.,
                "identity_loss": 0., "elapsed_seconds": .01, "peak_allocated_bytes": None, "peak_reserved_bytes": None,
                "metrics": {"top1": 1., "top5": 1.}, "ranking": ranks, "control_ranking": ranks.clone(),
                "truths": torch.tensor(frame.composition_id.to_numpy()), "filenames": frame.image_name.tolist()}
        for seed in (43, 44):
            for label, cap in (("control", None), ("bounded", .2)):
                cfg = {**cls.config, "training_seed": seed, "residual_cap": cap}
                contract = cls.ns["make_replication_contract"](cls.split, cls.table, cfg)
                folder = cls.group / ("seed_" + str(seed) + "_" + label)
                folder.mkdir()
                cls.ns["write_exclusive_json"](folder / "declaration.json", {"contract": contract,
                    "contract_sha256": cls.ns["json_hash"](contract)})
                cls.table.to_csv(folder / "direct_prompts.csv", index=False)
                adapter = cls.ns["initialize_replication_adapter"](seed, cap)
                digest = cls.ns["json_hash"]({k: cls.ns["tensor_sha256"](v) for k, v in adapter.state_dict().items()})
                cls.ns["write_exclusive_json"](folder / "initial_state.json", {"training_seed": seed, "split_seed": 42, "sha256": digest})
                proto = cls.ns["component_prototypes"](cls.cache["direct"], cls.table, contract["fitting_composition_ids"])
                state = cls.ns["seal_text_state"](cls.cache["direct"], proto, contract)
                selected = cls.ns["selection_record"](0, validation["val_seen"], validation["val_unseen"])
                checkpoint = cls.ns["save_adapter_checkpoint"](folder / "adapter_epoch_000_fixture.pt", adapter, state, contract, selected)
                training = {"loss": 1., "crop_loss": 1., "disease_loss": 1., "identity_loss": 0., "conditional_loss": 0.,
                    "elapsed_seconds": .01, "peak_allocated_bytes": None, "peak_reserved_bytes": None, "sample_count": 4868}
                pilot = {"best_validation": validation, "best_record": selected, "baseline_record": selected,
                    "best_checkpoint": checkpoint, "strict_H1_gain_over_identity": False,
                    "history": [cls.ns["history_row"](i, None if i == 0 else training, validation) for i in range(6)],
                    "epoch_predictions": [row for i in range(6) for row in cls.ns["prediction_rows"](validation, i)]}
                cls.ns["export_replication_results"](folder, contract, pilot, cls.split)
                path = folder / "adapter_summary.json"
                cls.records.append({"training_seed": seed, "arm": label, "directory": str(folder),
                    "summary_path": str(path), "summary_sha256": cls.ns["sha256_file"](path),
                    "artifacts": {p.name: cls.ns["sha256_file"](p) for p in folder.iterdir() if p.is_file()}})
                declarations.append({"training_seed": seed, "arm": label, "directory": str(folder), "contract": contract,
                                     "contract_sha256": cls.ns["json_hash"](contract)})
        group = {"schema": "wp3b-m5-group-v1", "config": cls.config, "arms": declarations,
                 "historical_reference": cls.ns["HISTORICAL_REFERENCE"], "code_sha256": cls.ns["notebook_algorithm_hash"](),
                 "selection_policy": cls.ns["SELECTION_POLICY"]}
        cls.ns["write_exclusive_json"](cls.group / "replication_declaration.json", group)
        completed = {"schema": "wp3b-m5-completed-v1", "declaration_sha256": cls.ns["sha256_file"](cls.group / "replication_declaration.json"),
                     "records": cls.records}
        cls.ns["write_exclusive_json"](cls.group / "completed.json", completed)
        cls.original_completed = (cls.group / "completed.json").read_bytes()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_complete_cpu_group_builds_and_reopens_report_without_memory_measurements(self):
        ns = self.ns
        historical = []
        for record in self.records[:2]:
            summary = json.loads(Path(record["summary_path"]).read_text())
            summary["config"]["seed"] = 42
            for key in ("training_seed", "training_seeds", "split_seed"):
                summary["config"].pop(key)
            historical.append({"training_seed": 42, "arm": record["arm"], "source": "synthetic_historical_reference",
                "directory": record["directory"], "summary": summary, "epochs": []})
        with patch.dict(ns, {"load_historical_reference": lambda *a: historical}):
            report = ns["load_and_report_replications"](self.group, self.split, self.table, self.config)
            self.assertEqual(len(report["selected_table"]), 6)
            self.assertTrue(report["selected_table"].peak_fitting_allocated_bytes.isna().all())
            reopened = ns["load_and_report_replications"](self.group, self.split, self.table, self.config)
            self.assertTrue(report["selected_table"].equals(reopened["selected_table"]))
            plt = ns["plt"]
            plt.switch_backend("Agg")
            plt.close("all")
            with patch.object(plt, "show"):
                ns["plot_replication_report"](report)
            self.assertEqual(len(plt.get_fignums()), 2)
            for line in plt.figure(plt.get_fignums()[0]).axes[0].lines:
                self.assertTrue(all(v == 100 for v in line.get_ydata()))
            plt.close("all")

    def test_required_history_and_checkpoint_manifest_entries_cannot_be_omitted(self):
        path = self.group / "completed.json"
        try:
            for name in ("adapter_history.csv", "adapter_epoch_000_fixture.pt"):
                bad = json.loads(self.original_completed)
                bad["records"][0]["artifacts"].pop(name)
                path.write_text(json.dumps(bad))
                with self.subTest(name=name):
                    with self.assertRaises(ValueError):
                        self.ns["read_completed_replication"](self.group, self.split, self.table, self.config)
        finally:
            path.write_bytes(self.original_completed)

    def test_changed_checkpoint_bytes_and_truncated_predictions_are_rejected(self):
        record = self.records[0]
        checkpoint = Path(record["directory"]) / "adapter_epoch_000_fixture.pt"
        original = checkpoint.read_bytes()
        try:
            checkpoint.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
            with self.assertRaises(ValueError):
                self.ns["read_completed_replication"](self.group, self.split, self.table, self.config)
        finally:
            checkpoint.write_bytes(original)
        predictions = Path(record["directory"]) / "adapter_epoch_predictions.csv"
        original_predictions = predictions.read_bytes()
        completed = self.group / "completed.json"
        try:
            frame = self.ns["pd"].read_csv(predictions).iloc[:-1]
            frame.to_csv(predictions, index=False)
            changed = json.loads(self.original_completed)
            changed["records"][0]["artifacts"][predictions.name] = self.ns["sha256_file"](predictions)
            completed.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                self.ns["read_completed_replication"](self.group, self.split, self.table, self.config)
        finally:
            predictions.write_bytes(original_predictions)
            completed.write_bytes(self.original_completed)


class UnavailableMemoryRoundtripTests(unittest.TestCase):
    def test_existing_report_roundtrips_missing_memory_without_dtype_false_alarm(self):
        ns = namespace()
        table = ns["pd"].DataFrame({"memory": [None, None], "accuracy": [.1, .2]})
        report = {key: table.copy() for key in ("selected_table", "descriptive_statistics", "paired_differences",
                                              "per_pair", "epoch_subgroups", "trajectories")}
        report.update(gate_decisions=[], gate_summary={}, interpretation="Synthetic missing-memory fixture")
        with m4.reference.reference.workspace_temporary_directory() as directory:
            with patch.dict(ns, {"load_historical_reference": lambda *a: [], "read_completed_replication": lambda *a: [],
                                 "build_replication_report": lambda *a: report}):
                ns["load_and_report_replications"](directory, None, None, ns["CONFIG"])
                reopened = ns["load_and_report_replications"](directory, None, None, ns["CONFIG"])
        self.assertEqual(reopened["interpretation"], report["interpretation"])


if __name__ == "__main__":
    unittest.main()
