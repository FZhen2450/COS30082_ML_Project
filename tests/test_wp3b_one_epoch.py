"""One-epoch paired experiment contracts, exercised without real training."""
import ast
import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import torch

import test_wp3b_replication as parent

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/wp3b_one_epoch_replication.ipynb"


def read_code(path):
    return [ast.parse("".join(cell["source"])) for cell in json.loads(path.read_text())["cells"]
            if cell["cell_type"] == "code"]


def definitions(path):
    return {node.name: node for cell in read_code(path) for node in cell.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))}


class OneEpochTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = parent.namespace()
        cls.ns.update({
            target.id: ast.literal_eval(node.value)
            for cell in read_code(NOTEBOOK) for node in cell.body if isinstance(node, ast.Assign)
            for target in node.targets if isinstance(target, ast.Name)
            and target.id in {"CONFIG", "TRAINING_SEEDS", "M5_PARENT", "M5_CACHE_SHA256",
                              "LOAD_FEATURE_CACHE", "RUN_REPLICATIONS", "BUILD_REPLICATION_REPORT"}
        })
        module = ast.Module(body=list(definitions(NOTEBOOK).values()), type_ignores=[])
        exec(compile(module, str(NOTEBOOK), "exec"), cls.ns)

    def test_defaults_off_and_no_saved_execution(self):
        for flag in ("LOAD_FEATURE_CACHE", "RUN_REPLICATIONS", "BUILD_REPLICATION_REPORT"):
            self.assertIs(self.ns[flag], False)
        for cell in json.loads(NOTEBOOK.read_text())["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])

    def test_scientific_definitions_match_accepted_m5(self):
        old, new = definitions(parent.NOTEBOOK), definitions(NOTEBOOK)
        for name in ("build_development_split", "build_prompt_table", "rank_global_ids",
                     "cosine_candidate_scores", "topk_accuracy", "harmonic_mean",
                     "ResidualVisualAdapter", "adapter_objective", "m3_training_objective",
                     "selection_record", "better_selection", "train_adapter_epoch",
                     "validate_adapter", "load_feature_cache", "verify_cache_image_files",
                     "save_adapter_checkpoint", "load_adapter_checkpoint", "diagnose_predictions"):
            self.assertEqual(ast.dump(old[name]), ast.dump(new[name]), name)

    def test_fixed_budget_and_new_seeds(self):
        config = self.ns["CONFIG"]
        self.assertEqual(config["epochs"], 1)
        self.assertEqual(config["training_seeds"], [45, 46, 47])
        self.assertTrue(self.ns["validate_replication_config"](config))
        for key, value in (("epochs", 2), ("epochs", True), ("split_seed", 45),
                           ("training_seeds", [42, 43, 44]), ("training_seed", 44),
                           ("feature_cache_sha256", "a" * 64)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.ns["validate_replication_config"]({**config, key: value})

    def test_paired_initialization_and_original_identity(self):
        for seed in self.ns["TRAINING_SEEDS"]:
            left = self.ns["initialize_replication_adapter"](seed, None)
            right = self.ns["initialize_replication_adapter"](seed, .2)
            for a, b in zip(left.parameters(), right.parameters()):
                self.assertTrue(torch.equal(a, b))
            features = torch.randn(2, 768)
            self.assertTrue(torch.equal(left(features), features))
            self.assertTrue(torch.equal(right(features), features))

    def test_split_candidate_and_prompt_contracts(self):
        ns = self.ns
        split = ns["build_development_split"](ROOT / "DPD_Small_train.csv", seed=42)
        table = ns["build_prompt_table"](split)
        contract = ns["make_replication_contract"](
            split, table, {**ns["CONFIG"], "training_seed": 47, "residual_cap": .2})
        self.assertEqual([len(split.fit), len(split.val_seen), len(split.val_unseen)], [4868, 1238, 1894])
        self.assertEqual(len(contract["candidate_ids"]), 323)
        self.assertNotEqual(contract["candidate_ids"], list(range(323)))
        self.assertEqual(contract["direct_prompt_sha256"],
                         "753672c7964bcb20a1346145f21a72b40a19f55590a006b5783823f23c13f537")

    def test_one_epoch_endpoints_are_reported_even_when_epoch_zero_wins(self):
        summary = {"history": [{"epoch": 0, "h1": .2}, {"epoch": 1, "h1": .1}],
                   "selected": {"epoch": 0, "h1": .2}}
        row = self.ns["endpoint_row"](45, "control", summary,
            {"diseased_correct": 80, "false_healthy": 500})
        self.assertEqual(row["selected_epoch"], 0)
        self.assertEqual(row["selected_h1"], .2)
        self.assertEqual(row["epoch1_h1"], .1)
        self.assertEqual(row["epoch1_unseen_diseased_correct"], 80)

    def test_six_arms_are_declared_before_training_and_failure_stops_group(self):
        ns = self.ns
        attempts = []
        with parent.m4.reference.reference.workspace_temporary_directory() as tmp:
            config = {**ns["CONFIG"], "output_directory": tmp, "device": "cpu"}

            def run_arm(split, table, config, cache, contract, directory, adapter):
                declaration = json.loads((Path(directory).parent / "replication_declaration.json").read_text())
                self.assertEqual([(a["training_seed"], a["arm"]) for a in declaration["arms"]],
                                 [(seed, arm) for seed in (45, 46, 47) for arm in ("control", "bounded")])
                self.assertTrue(all(a["contract"]["hyperparameters"]["epochs"] == 1 for a in declaration["arms"]))
                attempts.append(config["training_seed"])
                raise RuntimeError("synthetic interruption")

            with patch.dict(ns, {"REPLICATION_STARTED": False, "require_external_path": Path,
                                 "verify_replication_cache": lambda *args: True,
                                 "make_replication_contract": lambda s, t, c: {"hyperparameters": c},
                                 "run_replication_arm": run_arm}):
                with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                    ns["run_replications"](None, pd.DataFrame({"prompt": ["dummy"]}), config, {})
                with self.assertRaisesRegex(RuntimeError, "already started"):
                    ns["run_replications"](None, None, config, {})
            self.assertEqual(attempts, [45])
            group, = Path(tmp).iterdir()
            self.assertTrue((group / "failed.json").is_file())
            self.assertFalse((group / "completed.json").exists())


    def test_synthetic_group_exports_and_reopens_both_checkpoints_per_arm(self):
        ns = self.ns
        parent.m4.CacheTests.setUpClass()
        split, cache = parent.m4.CacheTests.split, parent.m4.CacheTests.cache
        table = ns["build_prompt_table"](split)
        validation = {}
        for role in ("val_seen", "val_unseen"):
            frame = getattr(split, role)
            truths = torch.tensor(frame.composition_id.to_numpy())
            ranks = torch.tensor([[int(truth), *[i for i in split.candidate_ids if i != truth][:4]]
                                  for truth in truths.tolist()])
            validation[role] = {
                "sample_count": len(frame), "loss": 1., "crop_loss": 1., "disease_loss": 1.,
                "identity_loss": 0., "elapsed_seconds": .01, "peak_allocated_bytes": None,
                "peak_reserved_bytes": None, "metrics": {"top1": 1., "top5": 1.},
                "ranking": ranks, "control_ranking": ranks.clone(), "truths": truths,
                "filenames": frame.image_name.tolist(),
            }
        training = {"loss": 1., "crop_loss": 1., "disease_loss": 1., "identity_loss": 0.,
                    "conditional_loss": 0., "sample_count": 4868, "elapsed_seconds": .01,
                    "peak_allocated_bytes": None, "peak_reserved_bytes": None}
        with parent.m4.reference.reference.workspace_temporary_directory() as tmp:
            config = {**ns["CONFIG"], "output_directory": tmp, "device": "cpu"}
            train = Mock(return_value=training)
            with patch.dict(ns, {"REPLICATION_STARTED": False, "verify_replication_cache": lambda *a: True,
                                 "evaluate_replication_folds": lambda *a: validation, "train_adapter_epoch": train}):
                group = Path(ns["run_replications"](split, table, config, cache)["group_directory"])
                self.assertEqual(train.call_count, 6)
                records = ns["read_completed_replication"](group, split, table, config)
                rows, gates = ns["compare_one_epoch_records"](records)
                self.assertEqual(len(rows), 6)
                self.assertEqual(len(gates), 3)
                self.assertTrue((rows.selected_epoch == 0).all())  # exact tie keeps the earlier epoch
                self.assertTrue((rows.epoch1_h1 == 1.).all())
                self.assertTrue((gates.epoch1_h1_delta == 0.).all())
                checkpoint, = Path(records[0]["directory"]).glob("adapter_epoch_001_*.pt")
                checkpoint.unlink()
                with self.assertRaisesRegex(ValueError, "artifact manifest"):
                    ns["read_completed_replication"](group, split, table, config)
