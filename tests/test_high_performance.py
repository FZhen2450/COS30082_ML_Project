"""M5 preparation: synthetic CPU tests only, no full training or GPU experiments."""

import copy
import hashlib
import io
import json
import math
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from test_notebook import helper_namespace, code_sources, literal_assignments


class ToyMultiTask(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 4)
        self.crop_head = nn.Linear(4, 55)
        self.disease_head = nn.Linear(4, 175)

    def forward(self, inputs):
        features = self.backbone(inputs)
        return {"crop_logits": self.crop_head(features), "disease_logits": self.disease_head(features),
                "features": features}


class HighPerformanceNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = helper_namespace()

    def config(self, **updates):
        return {**copy.deepcopy(self.ns["CONFIG"]), **updates}

    def test_exactly_two_final_architectures_and_default_off_sequence(self):
        values = literal_assignments()
        self.assertEqual(set(values["APPROVED_ARCHITECTURES"]),
                         {"multitask_efficientnet_b3", "multitask_convnext_base"})
        for key in ("RUN_TRAINING", "RUN_EXPENSIVE_EVALUATION", "RUN_PLANNED_EXPERIMENTS",
                    "RUN_GPU_TUNING", "PREPARE_RUN"):
            self.assertIs(values[key], False)
        self.assertEqual(values["CONFIG"]["finetune_epochs"], 100)
        self.assertIn("EXPERIMENT_PLAN", code_sources()["configuration"])
        self.assertIn("run_experiment_sequence", code_sources()["training"])

    def test_new_output_dimensions_and_parameter_counts_without_downloads(self):
        expected = {"multitask_efficientnet_b3": (1536, 11049742),
                    "multitask_convnext_base": (1024, 87802214)}
        with patch("torch.hub.download_url_to_file", side_effect=AssertionError("download")):
            for architecture, (width, count) in expected.items():
                with self.subTest(architecture=architecture), torch.random.fork_rng(devices=[]):
                    model = self.ns["create_multitask_model"](architecture).eval()
                    self.assertEqual(model.feature_dim, width)
                    self.assertEqual(sum(p.numel() for p in model.parameters()), count)
                    for resolution in (32, 64):
                        with torch.inference_mode():
                            outputs = model(torch.randn(1, 3, resolution, resolution))
                        self.assertEqual(tuple(outputs["crop_logits"].shape), (1, 55))
                        self.assertEqual(tuple(outputs["disease_logits"].shape), (1, 175))
                        self.assertEqual(tuple(outputs["features"].shape), (1, width))
                        self.assertTrue(all(torch.isfinite(v).all() for v in outputs.values()))
                    del model

    def test_pretrained_identifiers_and_explicit_no_random_final_experiments(self):
        ids = self.ns["PRETRAINED_IDS"]
        self.assertEqual(ids["multitask_efficientnet_b3"], "EfficientNet_B3_Weights.IMAGENET1K_V1")
        self.assertEqual(ids["multitask_convnext_base"], "ConvNeXt_Base_Weights.IMAGENET1K_V1")
        with self.assertRaises(ValueError):
            self.ns["run_high_performance_experiment"](
                self.config(initialization="random"), None, Path("."), Path("."))

    def test_stage_resolution_transition_and_finite_budget(self):
        cfg = self.config(warmup_epochs=2, finetune_epochs=100, progressive_epochs=5)
        stage = self.ns["stage_for_epoch"]
        self.assertEqual(stage(cfg, 1), ("warmup", 1, 300))
        self.assertEqual(stage(cfg, 3), ("finetune", 1, 300))
        self.assertEqual(stage(cfg, 8), ("finetune", 6, 380))
        self.assertEqual(stage(cfg, 102), ("finetune", 100, 380))
        for epoch in (0, 103):
            with self.assertRaises(ValueError):
                stage(cfg, epoch)

    def test_frozen_head_stage_and_discriminative_full_stage(self):
        model = ToyMultiTask()
        cfg = self.config()
        optimizer, scheduler = self.ns["configure_training_stage"](model, cfg, "warmup")
        self.assertFalse(any(p.requires_grad for p in model.backbone.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.crop_head.parameters()))
        self.assertIsNone(scheduler)
        self.assertEqual(optimizer.param_groups[0]["lr"], cfg["warmup_head_lr"])
        optimizer, scheduler = self.ns["configure_training_stage"](model, cfg, "finetune")
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertEqual([g["name"] for g in optimizer.param_groups], ["backbone", "heads"])
        ratio = optimizer.param_groups[1]["lr"] / optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[1]["lr"] / optimizer.param_groups[0]["lr"], ratio)

    def test_accumulation_matches_full_batch_with_partial_microbatch(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(111)
            original = ToyMultiTask()
            inputs = torch.randn(5, 3)
        micro = copy.deepcopy(original)
        full = copy.deepcopy(original)
        pairs = [(inputs[i], {"crop": torch.tensor(i), "disease": torch.tensor(i),
                              "composition": torch.tensor(10+i)}) for i in range(5)]
        a = torch.optim.SGD(micro.parameters(), lr=0.1)
        b = torch.optim.SGD(full.parameters(), lr=0.1)
        with redirect_stdout(io.StringIO()):
            x = self.ns["train_one_epoch"](micro, DataLoader(pairs, batch_size=2), a, "cpu",
                                           accumulation_steps=3)
            y = self.ns["train_one_epoch"](full, DataLoader(pairs, batch_size=5), b, "cpu")
        self.assertEqual(x["optimizer_steps"], 1)
        self.assertEqual((x["sample_count"], x["batch_count"]), (5, 3))
        self.assertEqual(y["optimizer_steps"], 1)
        for key, value in micro.state_dict().items():
            torch.testing.assert_close(value, full.state_dict()[key], rtol=1e-5, atol=1e-7)

    def test_final_partial_accumulation_window_updates_once_more(self):
        model = ToyMultiTask()
        pairs = [(torch.ones(3), {"crop": torch.tensor(0), "disease": torch.tensor(0)})
                 for _ in range(5)]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        with redirect_stdout(io.StringIO()):
            result = self.ns["train_one_epoch"](model, DataLoader(pairs, batch_size=2),
                                                optimizer, "cpu", accumulation_steps=2)
        self.assertEqual(result["optimizer_steps"], 2)
        self.assertEqual(result["sample_count"], 5)

    def test_cpu_bf16_autocast_and_scaler_contract(self):
        model = ToyMultiTask()
        with self.ns["amp_context"]("cpu", "bf16"):
            outputs = model(torch.ones(2, 3))
            losses = self.ns["compute_multitask_loss"](
                outputs, {"crop": torch.zeros(2, dtype=torch.long),
                          "disease": torch.zeros(2, dtype=torch.long)})
        self.assertEqual(outputs["crop_logits"].dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertFalse(self.ns["make_scaler"]("cpu", "bf16").is_enabled())
        with self.assertRaises(ValueError):
            self.ns["amp_context"]("cpu", "fp16")

    def test_nonfinite_unscaled_gradients_reject_optimizer_update(self):
        model = ToyMultiTask()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, float("nan"))
        with patch.object(optimizer, "step", wraps=optimizer.step) as step:
            with self.assertRaises(FloatingPointError):
                self.ns["optimizer_update"](model, optimizer, self.ns["make_scaler"]("cpu", "fp32"), 1.0)
        step.assert_not_called()

    def test_early_stopping_uses_strict_primary_h1_and_saved_state(self):
        cfg = self.config(early_stopping_patience=2, min_finetune_epochs=3)
        state = {}
        stops = []
        for value in (0.1, 0.2, 0.2, 0.19):
            state, stop = self.ns["update_early_stopping"](state, value, cfg)
            stops.append(stop)
        self.assertEqual(stops, [False, False, False, True])
        state, stop = self.ns["update_early_stopping"](state, math.nextafter(0.2, 1), cfg)
        self.assertFalse(stop)
        self.assertEqual(state["bad_epochs"], 0)

    def test_rng_and_loader_generator_round_trip_is_safe_tensor_data(self):
        loader = DataLoader([1, 2, 3], generator=torch.Generator().manual_seed(5))
        state = self.ns["capture_rng_state"]([loader])
        expected = (random.random(), np.random.rand(), torch.rand(3),
                    torch.randperm(3, generator=loader.generator))
        self.ns["restore_rng_state"](state, [loader])
        actual = (random.random(), np.random.rand(), torch.rand(3),
                  torch.randperm(3, generator=loader.generator))
        self.assertEqual(actual[:2], expected[:2])
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
        torch.testing.assert_close(actual[3], expected[3], rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rng.pt"
            torch.save(state, path)
            loaded = torch.load(path, weights_only=True)
            self.assertIsInstance(loaded["numpy"]["keys"], list)

    def test_recovery_states_configuration_and_code_checks(self):
        cfg = self.config(device="cpu", num_workers=0)
        model = ToyMultiTask()
        optimizer, scheduler = self.ns["configure_training_stage"](model, cfg, "finetune")
        loss = model(torch.ones(1, 3))["crop_logits"].sum()
        loss.backward()
        optimizer.step()
        scheduler.step()
        loader = DataLoader([1, 2], generator=torch.Generator().manual_seed(1))
        split = SimpleNamespace(source_sha256="source", candidate_sha256="candidates")
        with tempfile.TemporaryDirectory() as directory:
            context = {"config": cfg, "model": model, "optimizer": optimizer, "loaders": (loader,),
                       "history": [{"epoch": 3}], "selected": None, "directory": Path(directory),
                       "base_metadata": {"split": {"source_sha256": "source", "candidate_sha256": "candidates"},
                                         "code": {"executable_code_sha256": "reviewed"}}}
            path = self.ns["save_recovery_checkpoint"](
                context, 3, "finetune", 300, scheduler, self.ns["make_scaler"]("cpu", "fp32"),
                {"best_h1": 0.2, "bad_epochs": 1, "fine_epochs": 1})
            saved = self.ns["read_recovery_checkpoint"](path, cfg, split, "reviewed")
            self.assertEqual(saved["completed_epoch"], 3)
            self.assertTrue(saved["optimizer_state_dict"]["state"])
            self.assertEqual(saved["scheduler_state_dict"], scheduler.state_dict())
            self.assertEqual(saved["early_state"]["bad_epochs"], 1)
            restored = ToyMultiTask()
            restored.load_state_dict(saved["model_state_dict"])
            for key, value in model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
            recovered_optimizer,recovered_scheduler=self.ns["configure_training_stage"](restored,cfg,"finetune")
            recovered_optimizer.load_state_dict(saved["optimizer_state_dict"])
            recovered_scheduler.load_state_dict(saved["scheduler_state_dict"])
            for active_model,active_optimizer,active_scheduler in (
                    (model,optimizer,scheduler),(restored,recovered_optimizer,recovered_scheduler)):
                active_optimizer.zero_grad()
                active_model(torch.ones(1,3))["crop_logits"].sum().backward()
                active_optimizer.step();active_scheduler.step()
            for key,value in model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key],value,rtol=0,atol=0)
            for changed, changed_split, sha in (
                ({**cfg, "precision": "fp32"}, split, "reviewed"),
                (cfg, SimpleNamespace(source_sha256="other", candidate_sha256="candidates"), "reviewed"),
                (cfg, split, "changed")):
                with self.assertRaises(ValueError):
                    self.ns["read_recovery_checkpoint"](path, changed, changed_split, sha)

    def test_atomic_recovery_failure_preserves_previous_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last_completed.pt"
            self.ns["atomic_torch_save"]({"epoch": 1}, path)
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(torch, "save", side_effect=OSError("synthetic write failure")):
                with self.assertRaises(OSError):
                    self.ns["atomic_torch_save"]({"epoch": 2}, path, overwrite=True)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_bounded_probe_budget_and_bn_checkpointing_rejection(self):
        cfg = self.config(gradient_checkpointing=True)
        with self.assertRaises(ValueError):
            self.ns["validate_config"](cfg)
        with self.assertRaises(ValueError):
            self.ns["bounded_gpu_tuning"](self.config(), None, Path("."), max_profiles=17)
        self.assertIn("max_profiles", code_sources()["bounded-tuning"])
        self.assertIn('range(warmup_steps+measured_steps)', code_sources()["gpu-probe-functions"])
        self.assertIn("use_reentrant=False", code_sources()["architecture"])
        self.assertIn("model.backbone.eval()", code_sources()["training-functions"])

    def test_effective_sample_cap_preserves_leftovers_and_updates(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(19);model=ToyMultiTask();inputs=torch.randn(9,3)
        reference=copy.deepcopy(model)
        rows=[(inputs[i],{"crop":torch.tensor(i),"disease":torch.tensor(i)}) for i in range(9)]
        a=torch.optim.SGD(model.parameters(),lr=0.1)
        b=torch.optim.SGD(reference.parameters(),lr=0.1)
        with redirect_stdout(io.StringIO()):
            result=self.ns["train_one_epoch"](model,DataLoader(rows,batch_size=4),a,"cpu",
                                               effective_batch_size=5)
            expected=self.ns["train_one_epoch"](reference,DataLoader(rows,batch_size=5),b,"cpu")
        self.assertEqual(result["sample_count"],9)
        self.assertEqual(result["optimizer_steps"],2)
        for key,value in model.state_dict().items():
            torch.testing.assert_close(value,reference.state_dict()[key],rtol=1e-5,atol=1e-7)

    def test_executable_identity_ignores_outputs_and_resume_controls_but_not_processing(self):
        notebook={"cells":[
            {"id":"configuration","cell_type":"code","source":['CONFIG={"resume_from":None}\n',
              'RUN_TRAINING=False\n','def validate_config(config):\n    return config\n'],
              "outputs":[],"execution_count":None},
            {"id":"loss","cell_type":"code","source":['def objective(x):\n    return x*2\n'],
              "outputs":[],"execution_count":None}]}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"notebook.ipynb";path.write_text(json.dumps(notebook))
            identity=self.ns["notebook_code_identity"](path)
            notebook["cells"][0]["source"][0]='CONFIG={"resume_from":"last_completed.pt"}\n'
            notebook["cells"][0]["source"][1]='RUN_TRAINING=True\n'
            notebook["cells"][1]["outputs"]=[{"text":"saved progress"}]
            notebook["cells"][1]["execution_count"]=9
            path.write_text(json.dumps(notebook))
            self.assertEqual(self.ns["notebook_code_identity"](path),identity)
            notebook["cells"][1]["source"]=['def objective(x):\n    return x*3\n']
            path.write_text(json.dumps(notebook))
            self.assertNotEqual(self.ns["notebook_code_identity"](path),identity)

    def test_owner_boundary_stop_is_not_a_completed_comparison(self):
        source=code_sources()["recovery-functions"]
        self.assertIn('context["completed"]=stop_reason != "owner_stop_file"',source)
        self.assertIn("if not complete:",source)
        with redirect_stdout(io.StringIO()):
            result=self.ns["compare_planned_runs"]([{"execution_status":"partial_or_interrupted"}])
        self.assertTrue(result.empty)


    def test_amp_cross_entropy_has_full_precision_known_values(self):
        with self.ns["amp_context"]("cpu","bf16"):
            losses=self.ns["compute_multitask_loss"](
                {"crop_logits":torch.zeros(2,55,dtype=torch.bfloat16),
                 "disease_logits":torch.zeros(2,175,dtype=torch.bfloat16)},
                {"crop":torch.zeros(2,dtype=torch.long),"disease":torch.zeros(2,dtype=torch.long)})
        self.assertEqual(losses["total"].dtype,torch.float32)
        self.assertAlmostEqual(losses["crop"].item(),math.log(55),places=5)
        self.assertAlmostEqual(losses["disease"].item(),math.log(175),places=5)

    def test_synthetic_owner_pause_and_epoch_boundary_resume_end_to_end(self):
        import pandas as pd
        from PIL import Image
        class ToyImage(ToyMultiTask):
            def forward(self, inputs):
                return super().forward(inputs.mean(dim=(-2,-1)))
        cfg=self.config(device="cpu",num_workers=0,pin_memory=False,persistent_workers=False,
                        precision="fp32",channels_last=False,batch_size=2,start_batch_size=2,
                        warmup_batch_size=2,effective_batch_size=4,warmup_epochs=1,
                        finetune_epochs=1,progressive_epochs=1,start_resolution=32,final_resolution=32,
                        min_finetune_epochs=1,run_id="synthetic_recovery",epochs=2)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/"notebooks").mkdir()
            actual=Path(__file__).resolve().parents[1]/"notebooks/wp2_backbone_comparison.ipynb"
            (root/"notebooks/wp2_backbone_comparison.ipynb").write_bytes(actual.read_bytes())
            images=root/"images";images.mkdir()
            for name in ("fit0.png","fit1.png","seen.png","unseen.png"):
                Image.new("RGB",(6,7),(100,140,80)).save(images/name)
            columns=self.ns["COLUMN_NAMES"]
            def frame(names, component):
                return pd.DataFrame([[name,component,component,10+component,
                                     f"c{component}",f"d{component}",f"p{component}"]
                                     for name in names],columns=columns)
            mappings={"crop_id_to_name":{i:f"c{i}" for i in range(55)},
                      "disease_id_to_name":{i:f"d{i}" for i in range(175)}}
            ids=tuple(range(10,16));pairs=tuple((i,i) for i in range(6))
            _,_,checksum=self.ns["validate_candidate_spec"]({"ids":ids,"pairs":pairs})
            split=self.ns["DevelopmentSplit"](frame(["fit0.png","fit1.png"],0),frame(["seen.png"],1),
                frame(["unseen.png"],2),mappings,ids,pairs,"synthetic.csv","source",checksum,42,0.2,0.2)
            cfg["output_directory"]=str(root/"runs")
            original_write=self.ns["write_new_json"]
            def stop_after_prepare(path,value):
                original_write(path,value)
                Path(path).parent.joinpath("STOP_AFTER_EPOCH").touch()
            def factory(architecture,**kwargs):
                model=ToyImage();model.architecture=architecture
                return model
            def portable_save(model,path,**kwargs):
                Path(path).write_bytes(b"synthetic portable checkpoint fixture")
                return Path(path)
            overrides={"create_multitask_model":factory,
                       "verified_backbone_state":lambda config:(None,{"mode":"local_pretrained","purpose":"synthetic test double"}),
                       "save_checkpoint":portable_save,
                       "ENVIRONMENT":{"python":"unit-test","pytorch":str(torch.__version__)},
                       "write_new_json":stop_after_prepare}
            with patch.dict(self.ns,overrides),redirect_stdout(io.StringIO()):
                paused=self.ns["run_high_performance_experiment"](cfg,split,images,root)
            self.assertFalse(paused["completed"])
            self.assertEqual(paused["stop_reason"],"owner_stop_file")
            self.assertEqual(len(paused["history"]),1)
            published=paused["directory"]/"last_completed.pt"
            self.assertTrue(published.is_file())
            # Each logged line and the derived summary must be actual JSON, not literal backslash-n text.
            lines=(paused["directory"]/"history.jsonl").read_text().splitlines()
            self.assertEqual(len(lines),1)
            self.assertEqual(json.loads(lines[0])["epoch"],1)
            summary=self.ns["summarize_high_performance_run"](paused)
            self.assertEqual(json.loads((paused["directory"]/"run_summary.json").read_text())["execution_status"],
                             "partial_or_interrupted")
            (paused["directory"]/"STOP_AFTER_EPOCH").unlink()
            resumed_cfg={**cfg,"resume_from":str(published)}
            overrides["write_new_json"]=original_write
            with patch.dict(self.ns,overrides),redirect_stdout(io.StringIO()):
                resumed=self.ns["run_high_performance_experiment"](resumed_cfg,split,images,root)
            self.assertTrue(resumed["completed"])
            self.assertEqual([row["epoch"] for row in resumed["history"]],[1,2])
            recovered=torch.load(published,map_location="cpu",weights_only=True)
            self.assertEqual(recovered["completed_epoch"],2)
            self.assertEqual(recovered["optimizer_stage"],"finetune")


if __name__ == "__main__":
    unittest.main()
