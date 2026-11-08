"""M5E regressions execute visible notebook definitions with synthetic inputs."""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler

from test_notebook import ROOT, helper_namespace, literal_assignments, code_sources
from test_high_performance import ToyMultiTask

class M5ETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns=helper_namespace()

    def config(self,**changes):
        return {**copy.deepcopy(self.ns["CONFIG"]),**changes}

    def fitting(self):
        rows=[]
        for disease,count in ((0,16),(1,1),(2,4)):
            for i in range(count):
                rows.append([f"fit_{disease}_{i}.png",0,disease,10+disease,"crop",
                             f"d{disease}",f"crop_d{disease}"])
        return pd.DataFrame(rows,columns=self.ns["COLUMN_NAMES"])

    def test_stage_specific_metadata_for_both_architectures(self):
        for architecture,start,final in (("multitask_efficientnet_b3",300,380),
                                         ("multitask_convnext_base",224,320)):
            cfg=self.config(architecture=architecture,start_resolution=start,final_resolution=final)
            for epoch,resolution,stage in ((1,start,"warmup"),(3,start,"progressive_finetune"),
                                           (8,final,"final_finetune")):
                with self.subTest(architecture=architecture,epoch=epoch):
                    record=self.ns["model_preprocessing_metadata"](cfg,epoch=epoch,selected=True)
                    self.assertEqual(record["metadata_version"],2)
                    self.assertEqual(record["actual_resolution"],resolution)
                    self.assertEqual(record["actual_selected_checkpoint_resolution"],resolution)
                    self.assertEqual(record["image_size"],[resolution,resolution])
                    self.assertEqual(record["stage"],stage)
                    self.assertEqual(record["final_planned_resolution"],final)
                    self.assertEqual(record["validation_transform"],
                        repr(self.ns["get_transforms"](resolution,False,True)))

    def test_plan_metadata_does_not_claim_an_executed_resolution(self):
        record=self.ns["model_preprocessing_metadata"](self.config())
        self.assertEqual(record["scope"],"run_plan")
        self.assertIsNone(record["actual_resolution"])
        self.assertIsNone(record["actual_selected_checkpoint_resolution"])

    def info(self,epoch=1):
        cfg=self.config()
        pre=self.ns["model_preprocessing_metadata"](cfg,epoch=epoch,selected=True)
        return {"architecture":cfg["architecture"],"preprocessing":pre,
                "metadata":{"configuration":cfg,"epoch":epoch,
                            "selection":{"epoch":epoch,"resolution":pre["actual_resolution"]}}}

    def test_new_checkpoint_resolution_is_authoritative_and_conflicts_rejected(self):
        info=self.info()
        resolved=self.ns["resolve_checkpoint_preprocessing"](info)
        self.assertEqual(resolved["actual_resolution"],300)
        transform=self.ns["checkpoint_inference_transform"](info)
        self.assertEqual(tuple(transform(Image.new("RGB",(11,9))).shape),(3,300,300))
        bad=copy.deepcopy(info);bad["preprocessing"]["image_size"]=[380,380]
        with self.assertRaises(ValueError):self.ns["resolve_checkpoint_preprocessing"](bad)
        bad=copy.deepcopy(info);bad["metadata"]["selection"]["resolution"]=380
        with self.assertRaises(ValueError):self.ns["resolve_checkpoint_preprocessing"](bad)
        bad=copy.deepcopy(info);bad["preprocessing"]["validation_transform"]="224 stale"
        with self.assertRaises(ValueError):self.ns["resolve_checkpoint_preprocessing"](bad)

    def test_legacy_resolution_requires_explicit_acknowledgement_and_preserves_raw(self):
        info={"preprocessing":{"image_size":[380,380],"validation_transform":"legacy224"},
              "metadata":{"selection":{"resolution":300}}}
        before=copy.deepcopy(info)
        with self.assertRaises(ValueError):self.ns["checkpoint_inference_transform"](info)
        resolved=self.ns["resolve_checkpoint_preprocessing"](info,legacy_policy="recorded_epoch_resolution")
        self.assertEqual(resolved["actual_resolution"],300)
        self.assertTrue(resolved["legacy_inconsistency_acknowledged"])
        self.assertEqual(info,before)
        with self.assertRaises(ValueError):
            self.ns["resolve_checkpoint_preprocessing"](info,legacy_policy="explicit_resolution",legacy_resolution=380)

    def test_legacy_weights_remain_loadable_without_silent_inference_policy(self):
        model=self.ns["create_multitask_model"]("multitask_efficientnet_b0").eval()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"legacy.pt"
            self.ns["save_checkpoint"](model,path,preprocessing={"image_size":[224,224]},
                                       metadata={"selection":{"resolution":224}})
            loaded,info=self.ns["load_checkpoint"](path)
            self.assertEqual(info["preprocessing_resolution_status"],"legacy_unresolved")
            with self.assertRaises(ValueError):self.ns["checkpoint_inference_transform"](info)
            loaded,info=self.ns["load_checkpoint"](path,legacy_policy="recorded_epoch_resolution")
            self.assertEqual(info["inference_preprocessing"]["actual_resolution"],224)
            for k,v in model.state_dict().items():
                torch.testing.assert_close(v,loaded.state_dict()[k],rtol=0,atol=0)

    def test_new_epoch_metadata_checkpoint_roundtrip(self):
        cfg=self.config()
        model=self.ns["create_multitask_model"](cfg["architecture"]).eval()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"new.pt";info=self.info(8)
            self.ns["save_checkpoint"](model,path,preprocessing=info["preprocessing"],metadata=info["metadata"])
            loaded,metadata=self.ns["load_checkpoint"](path)
            self.assertEqual(metadata["inference_preprocessing"]["actual_resolution"],380)
            self.assertEqual(metadata["preprocessing_resolution_status"],"epoch_specific_v2")
            for k,v in model.state_dict().items():
                torch.testing.assert_close(v,loaded.state_dict()[k],rtol=0,atol=0)

    def test_global_stopping_counts_only_fine_bad_epochs_and_honors_grace(self):
        cfg=self.config(early_stopping_patience=2,min_finetune_epochs=3)
        state,stop=self.ns["update_early_stopping"]({},.3,cfg,stage="warmup")
        self.assertFalse(stop);self.assertEqual(state["fine_epochs"],0)
        state,stop=self.ns["update_early_stopping"](state,.2,cfg,stage="warmup")
        self.assertEqual(state["bad_epochs"],0)
        for i,h in enumerate((.1,.2,.29),1):
            state,stop=self.ns["update_early_stopping"](state,h,cfg,stage="finetune")
            self.assertEqual(stop,i==3)
        self.assertEqual(state["best_h1"],.3)
        self.assertEqual(state["bad_epochs"],3)

    def test_exact_h1_ties_and_loss_only_selection_do_not_reset_patience(self):
        cfg=self.config(early_stopping_patience=3,min_finetune_epochs=1)
        state,_=self.ns["update_early_stopping"]({},.3,cfg,stage="warmup")
        state,_=self.ns["update_early_stopping"](state,.3,cfg)
        self.assertEqual(state["bad_epochs"],1)
        metrics={"top1_harmonic_mean":.3}
        old={"sample_count":2,"losses":{"total":2.}}
        new={"sample_count":2,"losses":{"total":1.}}
        self.assertLess(self.ns["checkpoint_selection_key"](metrics,new,new,2),
                        self.ns["checkpoint_selection_key"](metrics,old,old,1))
        state,_=self.ns["update_early_stopping"](state,.3,cfg)
        self.assertEqual(state["bad_epochs"],2)
        state,stop=self.ns["update_early_stopping"](state,math.nextafter(.3,1),cfg)
        self.assertEqual(state["bad_epochs"],0);self.assertFalse(stop)
        for value in (float("nan"),-1.,1.1):
            with self.assertRaises(ValueError):self.ns["update_early_stopping"]({},value,cfg)

    def test_ticket_formula_probabilities_and_singleton_cap(self):
        split=SimpleNamespace(fit=self.fitting())
        spec=self.ns["fitting_sampling_plan"](split,self.config(fitting_sampler="capped_disease_tickets",sampler_weight_cap=3))
        self.assertEqual(spec["pool_size"],27)
        rows={r["disease_id"]:r for r in spec["diseases"]}
        self.assertEqual([rows[i]["tickets_per_image"] for i in (0,1,2)],[1,3,2])
        self.assertAlmostEqual(rows[1]["effective_disease_probability"],3/27)
        self.assertAlmostEqual(sum(r["effective_disease_probability"] for r in rows.values()),1.)
        self.assertLessEqual(rows[1]["expected_epoch_draws"],3)
        self.assertLessEqual(spec["maximum_image_repeats"],3)

    def test_deterministic_ticket_draws_hard_cap_and_rng_restore(self):
        split=SimpleNamespace(fit=self.fitting())
        spec=self.ns["fitting_sampling_plan"](split,self.config(fitting_sampler="capped_disease_tickets",sampler_weight_cap=3))
        def make(seed):
            return self.ns["CappedDiseaseTicketSampler"](spec,generator=torch.Generator().manual_seed(seed))
        a,b=make(42),make(42)
        first=list(a);self.assertEqual(first,list(b))
        self.assertEqual(len(first),len(split.fit))
        self.assertLessEqual(max(torch.bincount(torch.tensor(first)).tolist()),3)
        saved=a.generator.get_state();second=list(a);a.generator.set_state(saved)
        self.assertEqual(second,list(a))
        self.assertNotEqual(first,list(make(43)))

    def test_cap_one_is_a_permutation_and_invalid_caps_rejected(self):
        split=SimpleNamespace(fit=self.fitting())
        cfg=self.config(fitting_sampler="capped_disease_tickets",sampler_weight_cap=1)
        spec=self.ns["fitting_sampling_plan"](split,cfg)
        draw=list(self.ns["CappedDiseaseTicketSampler"](spec,generator=torch.Generator().manual_seed(4)))
        self.assertEqual(sorted(draw),list(range(len(split.fit))))
        for cap in (0,5,True,2.5,float("nan")):
            with self.assertRaises(ValueError):
                self.ns["fitting_sampling_plan"](split,{**cfg,"sampler_weight_cap":cap})

    def test_sampling_plan_reads_only_fitting_rows(self):
        class FitOnly:
            fit=self.fitting()
            @property
            def val_seen(self):raise AssertionError("validation statistics accessed")
            @property
            def val_unseen(self):raise AssertionError("validation statistics accessed")
        split=FitOnly()
        before=split.fit.copy(deep=True)
        spec=self.ns["fitting_sampling_plan"](split,self.config(fitting_sampler="capped_disease_tickets",sampler_weight_cap=3))
        self.assertEqual(spec["samples_per_epoch"],len(before))
        pd.testing.assert_frame_equal(split.fit,before)

    def test_only_fitting_loader_has_optional_sampler(self):
        frame=self.fitting()
        split=SimpleNamespace(fit=frame,val_seen=frame.iloc[:2],val_unseen=frame.iloc[2:4],seed=42)
        cfg=self.config(num_workers=0,batch_size=2,fitting_sampler="capped_disease_tickets",sampler_weight_cap=3)
        loaders=self.ns["create_stage_loaders"](split,Path("."),cfg,32)
        self.assertIsInstance(loaders[0].sampler,self.ns["CappedDiseaseTicketSampler"])
        for loader in loaders[1:]:self.assertIsInstance(loader.sampler,SequentialSampler)
        control=self.ns["create_stage_loaders"](split,Path("."),{**cfg,"fitting_sampler":"uniform_shuffle"},32)
        self.assertIsInstance(control[0].sampler,RandomSampler)
        self.assertFalse(control[0].sampler.replacement)
        self.assertEqual(len(loaders[0]),len(control[0]))

    def test_pilot_plan_has_exactly_two_fresh_equal_budget_base_arms(self):
        plan=self.ns["make_m5e_pilot_plan"](self.ns["CONFIG"])
        self.ns["validate_m5e_pilot_plan"](plan)
        self.assertEqual(len(plan),2)
        self.assertEqual({p["architecture"] for p in plan},{"multitask_convnext_base"})
        self.assertEqual({p["warmup_epochs"] for p in plan},{2})
        self.assertEqual({p["finetune_epochs"] for p in plan},{15})
        self.assertTrue(all(p["resume_from"] is None for p in plan))
        ignored={"fitting_sampler","pilot_arm","run_id"}
        self.assertEqual({k:v for k,v in plan[0].items() if k not in ignored},
                         {k:v for k,v in plan[1].items() if k not in ignored})
        for change in ({"head_lr":.003},{"finetune_epochs":100},{"resume_from":"/old/last.pt"}):
            bad=copy.deepcopy(plan);bad[1].update(change)
            with self.assertRaises(ValueError):self.ns["validate_m5e_pilot_plan"](bad)

    def test_original_long_sequence_is_rejected_before_model_creation(self):
        with patch.dict(self.ns,{"run_high_performance_experiment":lambda *a:self.fail("must not run")}):
            with self.assertRaises(ValueError):
                self.ns["run_experiment_sequence"]([self.config(),self.config(architecture="multitask_convnext_base")],None,None,ROOT)

    def test_pilot_comparison_rejects_partial_or_unequal_actual_budgets(self):
        plan=self.ns["make_m5e_pilot_plan"](self.ns["CONFIG"])
        summaries=[]
        for cfg in plan:
            summaries.append({"configuration":cfg,"execution_status":"completed","completed_epochs":17,
                 "history":[{"epoch":i,"train_samples":4868,"optimizer_steps":77} for i in range(1,18)],
                 "selected":{"epoch":1},"initial_model_sha256":"same","initialization_verified":True})
        # Budget validation is independent of actual result scoring.
        self.ns["validate_m5e_actual_budgets"](summaries)
        summaries[1]["completed_epochs"]=16
        with self.assertRaises(ValueError):self.ns["validate_m5e_actual_budgets"](summaries)

        # A user interruption in the second arm returns both summaries without
        # attempting an invalid paired effectiveness comparison.
        with tempfile.TemporaryDirectory() as directory:
            paused_plan=self.ns["make_m5e_pilot_plan"](self.ns["CONFIG"])
            for cfg in paused_plan:cfg["output_directory"]=directory
            stages=iter((True,False))
            def fake_run(config,*args):
                return {"completed":next(stages),"config":config,"history":[]}
            def fake_summary(context):
                return {"configuration":context["config"],"execution_status":
                        "completed" if context["completed"] else "partial_or_interrupted"}
            with patch.dict(self.ns,{
                "verified_backbone_state":lambda cfg:(None,{"source":"test double"}),
                "run_high_performance_experiment":fake_run,
                "summarize_high_performance_run":fake_summary,
                "plot_run_history":lambda *args:None,
                "compare_m5e_pilot_results":lambda *args:self.fail("partial arm must not be compared")}):
                reports=self.ns["run_m5e_pilot_sequence"](paused_plan,
                    SimpleNamespace(fit=range(4868),val_seen=range(1238),val_unseen=range(1894)),Path("."),ROOT)
            self.assertEqual(len(reports),2)
            self.assertEqual(reports[1]["execution_status"],"partial_or_interrupted")

    def test_near_equal_scores_keep_canonical_float64_ranking_without_tolerance(self):
        c=torch.zeros(1,55,dtype=torch.float64);d=torch.zeros(1,175,dtype=torch.float64)
        spec={"ids":(10,30,50,70,90,110),"pairs":tuple((i,i) for i in range(6))}
        c[0,1]=1e-12
        scores=self.ns["composition_scores"](c,d,spec)
        self.assertEqual(scores.dtype,torch.float64)
        self.assertGreater(float(scores[0,1]),float(scores[0,0]))
        self.assertEqual(torch.argsort(scores,dim=1,descending=True,stable=True)[0,0].item(),1)
        c.zero_();scores=self.ns["composition_scores"](c,d,spec)
        self.assertEqual(torch.argsort(scores,dim=1,descending=True,stable=True)[0].tolist(),list(range(6)))
        # NumPy is an algebraic check, never the ranking authority or an exact tie oracle.
        x=c.numpy();y=d.numpy()
        ls=lambda a:a-(np.log(np.exp(a-a.max(1,keepdims=True)).sum(1,keepdims=True))+a.max(1,keepdims=True))
        alternative=np.stack([ls(x)[:,a]+ls(y)[:,b] for a,b in spec["pairs"]],1)
        np.testing.assert_allclose(scores.numpy(),alternative,rtol=0,atol=2e-14)

    def test_pilot_training_controls_default_off(self):
        values=literal_assignments()
        self.assertIs(values["RUN_M5E_PILOT"],False)
        self.assertIs(values["RUN_TRAINING"],False)
        self.assertIs(values["RUN_PLANNED_EXPERIMENTS"],False)
        source=code_sources()["m5e-execution"]
        self.assertIn("if RUN_M5E_PILOT:",source)
        self.assertNotIn("subprocess",source)

    def test_epoch_diagnostics_include_components_pairs_healthy_and_fitting_preference(self):
        columns=self.ns["COLUMN_NAMES"]
        def frame(rows):return pd.DataFrame(rows,columns=columns)
        fit=frame([["f0",0,9,10,"c0","healthy","p10"],["f1",1,1,20,"c1","d1","p20"]])
        seen=frame([["s0",0,9,10,"c0","healthy","p10"],["s1",1,1,20,"c1","d1","p20"]])
        unseen=frame([["u0",2,2,30,"c2","d2","p30"],["u1",5,5,60,"c5","d5","p60"]])
        ids=(10,20,30,40,50,60);pairs=((0,9),(1,1),(2,2),(3,3),(4,4),(5,5))
        _,_,digest=self.ns["validate_candidate_spec"]({"ids":ids,"pairs":pairs})
        split=SimpleNamespace(fit=fit,val_seen=seen,val_unseen=unseen,candidate_ids=ids,candidate_pairs=pairs,
                              candidate_sha256=digest,mappings={"disease_id_to_name":{i:("healthy" if i==9 else f"d{i}") for i in range(175)}})
        c=torch.zeros(2,55);d=torch.zeros(2,175);c[:,0]=10;d[:,9]=10
        def result(f):
            return {"sample_count":2,"crop_logits":c,"disease_logits":d,
                    "composition_ids":torch.tensor(f.composition_id.to_numpy())}
        result=self.ns["pilot_epoch_diagnostics"](result(seen),result(unseen),split)
        u=result["folds"]["unseen"]
        self.assertEqual(u["healthy_disease_prediction_count"],2)
        self.assertEqual(u["fitting_composition_prediction_count"],2)
        self.assertEqual(u["disease_top1_hits"],0)
        self.assertEqual(len(u["pairs"]),2)
        self.assertEqual(u["pairs"][0]["top1_accuracy"],0.)
        self.assertEqual(result["metrics"]["top1_harmonic_mean"],0.)

    def test_loss_improvement_alone_cannot_be_called_sampler_success(self):
        plan=self.ns["make_m5e_pilot_plan"](self.ns["CONFIG"])
        summaries=[]
        for i,cfg in enumerate(plan):
            metrics={"seen_top1":.5,"seen_top5":.7,"unseen_top1":.05,
                     "unseen_top5":.1,"top1_harmonic_mean":.09,"top5_harmonic_mean":.17}
            fold={"crop_top1":.7,"disease_top1":.1,"healthy_disease_prediction_fraction":.4,
                  "fitting_composition_prediction_fraction":.95,
                  "pairs":[{"composition_id":30,"crop":"c","disease":"d","sample_count":10,"top1_hits":1,"top5_hits":2}]}
            summaries.append({"configuration":cfg,"execution_status":"completed","completed_epochs":17,
                "history":[{"epoch":e,"train_samples":4868,"optimizer_steps":77} for e in range(1,18)],
                "selected":{"epoch":1,"path":"synthetic.pt","metrics":metrics},
                "initial_model_sha256":"same","initialization_verified":True,"duration_seconds":1.,
                "peak_gpu_allocated_mib":None,
                "selected_pilot_diagnostics":{"folds":{"seen":fold,"unseen":fold}},
                "train_loss":100. if i==0 else .01})
        result=self.ns["compare_m5e_pilot_results"](summaries)
        self.assertFalse(result["success_claimed"])
        self.assertEqual(result["assessment"],"no_joint_evidence_of_improvement")

    def test_tiny_core_run_publishes_actual_metadata_and_diagnostics(self):
        import io
        from contextlib import redirect_stdout
        class ToyImage(ToyMultiTask):
            def forward(self,x):return super().forward(x.mean((-2,-1)))
        cfg=self.config(architecture="multitask_convnext_base",device="cpu",num_workers=0,
            precision="fp32",channels_last=False,pin_memory=False,persistent_workers=False,
            warmup_epochs=1,finetune_epochs=1,epochs=2,progressive_epochs=1,
            start_resolution=32,final_resolution=32,min_finetune_epochs=1,
            batch_size=2,start_batch_size=2,warmup_batch_size=2,effective_batch_size=4,
            experiment_kind="m5e_pilot",pilot_arm="balanced",
            fitting_sampler="capped_disease_tickets",sampler_weight_cap=3,run_id="toy_m5e")
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/"notebooks").mkdir()
            (root/"notebooks/wp2_backbone_comparison.ipynb").write_bytes((ROOT/"notebooks/wp2_backbone_comparison.ipynb").read_bytes())
            images=root/"images";images.mkdir()
            for name in ("f0.png","f1.png","f2.png","f3.png","s.png","u.png"):
                Image.new("RGB",(8,8),(40,130,90)).save(images/name)
            def frame(names,label):
                return pd.DataFrame([[n,label,label,10+label,f"c{label}",f"d{label}",f"p{label}"] for n in names],
                                    columns=self.ns["COLUMN_NAMES"])
            ids=tuple(range(10,16));pairs=tuple((i,i) for i in range(6))
            _,_,checksum=self.ns["validate_candidate_spec"]({"ids":ids,"pairs":pairs})
            mappings={"crop_id_to_name":{i:f"c{i}" for i in range(55)},
                      "disease_id_to_name":{i:("healthy" if i==9 else f"d{i}") for i in range(175)}}
            split=SimpleNamespace(fit=frame(["f0.png","f1.png","f2.png","f3.png"],0),
                val_seen=frame(["s.png"],1),val_unseen=frame(["u.png"],2),seed=42,
                candidate_ids=ids,candidate_pairs=pairs,candidate_sha256=checksum,
                source_sha256="synthetic",mappings=mappings)
            cfg["output_directory"]=str(root/"runs")
            saved=[]
            def save(model,path,**kwargs):
                saved.append(kwargs);Path(path).write_bytes(b"synthetic-checkpoint-test-double")
            def factory(architecture,**kwargs):
                model=ToyImage();model.architecture=architecture;return model
            with patch.dict(self.ns,{"create_multitask_model":factory,
                "verified_backbone_state":lambda cfg:(None,{"source":"synthetic test double"}),
                "save_checkpoint":save,"ENVIRONMENT":{"python":"test"}}),redirect_stdout(io.StringIO()):
                context=self.ns["run_high_performance_experiment"](cfg,split,images,root)
                summary=self.ns["summarize_high_performance_run"](context)
            self.assertTrue(context["completed"]);self.assertEqual(len(context["history"]),2)
            self.assertTrue(saved);self.assertEqual(saved[0]["preprocessing"]["actual_resolution"],32)
            self.assertEqual(saved[0]["preprocessing"]["metadata_version"],2)
            self.assertIsNotNone(summary["selected_pilot_diagnostics"])
            recovered=torch.load(context["directory"]/"last_completed.pt",weights_only=True)
            self.assertEqual(recovered["preprocessing"]["actual_resolution"],32)
            self.assertEqual(recovered["early_state"]["reference"],"global_best_h1")
            self.assertEqual(sum(r["optimizer_steps"] for r in context["history"]),2)

if __name__=="__main__":
    unittest.main()
