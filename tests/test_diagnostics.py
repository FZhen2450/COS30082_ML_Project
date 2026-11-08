"""M5D notebook diagnostics: synthetic data and training metadata only."""
import ast
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from test_notebook import ROOT, NOTEBOOK, helper_namespace, load_notebook, literal_assignments

class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = helper_namespace()

    def frame(self):
        return pd.DataFrame([
            ["a.jpg",0,0,10,"c0","d0","c0_d0"],
            ["b.jpg",1,1,90,"c1","d1","c1_d1"],
            ["c.jpg",1,1,90,"c1","d1","c1_d1"],
        ],columns=self.ns["COLUMN_NAMES"])

    def spec(self):
        return {"ids":(10,45,90,130,170,220),"pairs":((0,0),(0,1),(1,1),(2,2),(3,3),(4,4))}

    def logits(self):
        c=torch.full((3,55),-12.)
        d=torch.full((3,175),-12.)
        c[0,0]=8;c[1,1]=8;c[2,0]=8
        d[0,0]=8;d[1,0]=8;d[2,1]=8
        return c,d

    def test_independent_scores_match_logsumexp_and_reference(self):
        g=torch.Generator().manual_seed(11)
        c=torch.randn(4,55,generator=g);d=torch.randn(4,175,generator=g)
        spec=self.spec()
        got=self.ns["d_independent_scores"](c,d,spec)
        want=torch.stack([(c.double()[:,a]-torch.logsumexp(c.double(),1))+
                          (d.double()[:,b]-torch.logsumexp(d.double(),1)) for a,b in spec["pairs"]],1)
        torch.testing.assert_close(got,want,rtol=0,atol=0)
        torch.testing.assert_close(got,self.ns["composition_scores"](c,d,spec))
        raw=torch.stack([c[:,a].double()+d[:,b].double() for a,b in spec["pairs"]],1)
        self.assertTrue(torch.equal(got.argsort(dim=1,descending=True,stable=True),
                                    raw.argsort(dim=1,descending=True,stable=True)))

    def test_extreme_logits_and_deterministic_global_ties(self):
        c=torch.zeros(2,55);d=torch.zeros(2,175)
        scores=self.ns["d_independent_scores"](c,d,self.spec())
        self.assertEqual(scores.argsort(dim=1,descending=True,stable=True)[0].tolist(),list(range(6)))
        c[0,0]=torch.finfo(torch.float32).max;c[0,1]=-torch.finfo(torch.float32).max
        self.assertTrue(torch.isfinite(self.ns["d_independent_scores"](c,d,self.spec())).all())

    def test_bad_candidate_order_pairs_shapes_and_values_rejected(self):
        c,d=self.logits()
        bad=[{"ids":(90,10),"pairs":((1,1),(0,0))},
             {"ids":(10,10),"pairs":((0,0),(0,1))},
             {"ids":(10,90),"pairs":((0,0),(0,0))},
             {"ids":(10,90),"pairs":((0,0),(55,1))}]
        for spec in bad:
            with self.subTest(spec=spec),self.assertRaises(ValueError):
                self.ns["d_independent_scores"](c,d,spec)
        for crop,disease in [(c[:,:3],d),(c,d[:2]),(c*float("nan"),d)]:
            with self.assertRaises(ValueError):
                self.ns["d_independent_scores"](crop,disease,self.spec())

    def test_component_counts_support_and_fitting_preference(self):
        f=self.frame();fit=f.iloc[[0]].copy()
        fit=pd.concat([fit,pd.DataFrame([["fit.jpg",1,1,90,"c1","d1","c1_d1"]],columns=f.columns)])
        c,d=self.logits()
        report=self.ns["d_fold_report"](c,d,f,fit,self.spec())
        s=report["summary"]
        self.assertEqual(s["sample_count"],3)
        self.assertEqual(s["crop_top1"]["hits"],2)
        self.assertEqual(s["disease_top1"]["hits"],2)
        self.assertEqual(s["joint_component_top1_hits"],1)
        self.assertEqual(s["composition_top1"]["hits"],1)
        self.assertEqual(s["predicted_fitting_composition_top1"]["hits"],2)
        self.assertEqual(sum(x["sample_count"] for x in report["pairs"]),3)
        self.assertEqual(sum(x["composition_top1_hits"] for x in report["pairs"]),1)
        self.assertEqual(len(report["samples"]),3)
        self.assertEqual(report["samples"][2]["crop_fit_support"],1)
        self.assertTrue(all(len(x["top5_global_ids"])==5 for x in report["samples"]))
        self.assertEqual({x["error_type"] for x in report["examples"]},{"disease_wrong","crop_wrong"})

    def test_truth_pair_disagreement_and_empty_data_rejected(self):
        f=self.frame();c,d=self.logits()
        bad=f.copy();bad.loc[0,"crop_id"]=2
        with self.assertRaises(ValueError):self.ns["d_fold_report"](c,d,bad,f,self.spec())
        with self.assertRaises(ValueError):self.ns["d_fold_report"](c[:0],d[:0],f.iloc[:0],f,self.spec())

    def test_all_real_training_candidate_and_split_records_independently_verified(self):
        split=self.ns["build_development_split"](ROOT/"DPD_Small_train.csv")
        audit=self.ns["d_audit_contract"](split)
        self.assertEqual(audit["candidate_count"],323)
        self.assertEqual(audit["image_counts"],{"fit":4868,"seen":1238,"unseen":1894})
        self.assertEqual(audit["fitting_crop_count"],55)
        self.assertEqual(audit["fitting_disease_count"],175)
        altered=copy.deepcopy(split)
        object.__setattr__(altered,"candidate_pairs",tuple(reversed(altered.candidate_pairs)))
        with self.assertRaises(ValueError):self.ns["d_audit_contract"](altered)

    def history(self):
        rows=[]
        for epoch,h in enumerate([.10,.06,.04,.05,.049,.048,.047],1):
            rows.append({"epoch":epoch,"stage":"warmup" if epoch<=2 else "finetune",
                         "top1_harmonic_mean":h,"combined_validation_loss":2.,
                         "seen_samples":2,"unseen_samples":3,"seen_total_loss":2.,
                         "unseen_total_loss":2.,"seen_top1":.5,"unseen_top1":h/(2-h/.5)})
        return {"history":rows,"completed_epochs":7,
                "configuration":{"early_stopping_patience":3,"min_finetune_epochs":2},
                "selected":{"epoch":1,"key":[-.1,2.,1]}}

    def test_selection_and_fine_only_stopping_are_replayed_without_mutation(self):
        s=self.history();before=copy.deepcopy(s)
        a=self.ns["d_audit_history"](s)
        self.assertEqual(a["global_selected_epoch"],1)
        self.assertEqual(a["fine_only_first_stop_epoch"],7)
        self.assertEqual(a["globally_anchored_first_stop_epoch"],5)
        self.assertEqual(a["fine_best_epoch"],4)
        self.assertEqual(s,before)
        broken=copy.deepcopy(s);broken["selected"]["epoch"]=4
        with self.assertRaises(ValueError):self.ns["d_audit_history"](broken)

    def test_counterfactual_retains_exact_tiny_h1_improvements(self):
        s=self.history();s["history"][4]["top1_harmonic_mean"]=.050000000000001
        a=self.ns["d_audit_history"](s)
        self.assertEqual(a["fine_best_epoch"],5)
        self.assertIn(5,[x["epoch"] for x in a["fine_improvements"]])
        self.assertIsNone(a["fine_only_first_stop_epoch"])

    def test_inference_is_no_grad_and_preserves_parameters_and_bn_buffers(self):
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__();self.bn=nn.BatchNorm1d(3);self.c=nn.Linear(3,55);self.d=nn.Linear(3,175)
            def forward(self,x):
                if torch.is_grad_enabled():raise AssertionError("inference must disable gradients")
                z=self.bn(x)
                return {"crop_logits":self.c(z),"disease_logits":self.d(z)}
        model=Tiny().eval()
        before={k:v.clone() for k,v in model.state_dict().items()}
        targets=torch.tensor([[0,0,10],[1,1,90],[1,1,90]])
        # Dictionary-label loader, including a partial final batch.
        loader=[(torch.randn(2,3),{"crop":targets[:2,0],"disease":targets[:2,1],"composition":targets[:2,2]}),
                (torch.randn(1,3),{"crop":targets[2:,0],"disease":targets[2:,1],"composition":targets[2:,2]})]
        c,d=self.ns["d_collect_logits"](model,loader,self.frame(),"cpu",precision="fp32",
                                       channels_last=False,label="synthetic")
        self.assertEqual(tuple(c.shape),(3,55));self.assertEqual(tuple(d.shape),(3,175))
        for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,before[k]),k)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_checkpoint_metadata_tampering_rejected_before_model_creation(self):
        split=self.ns["build_development_split"](ROOT/"DPD_Small_train.csv")
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/"bad.pt"
            torch.save({"architecture":"wrong","metadata":{},"model_state_dict":{}},p)
            with patch.dict(self.ns,{"create_multitask_model":lambda *a,**k:self.fail("must reject before allocation")}):
                with self.assertRaises(ValueError):
                    self.ns["d_read_model_state"](p,split,"selected")

    def test_default_off_diagnostics_and_no_training_in_new_cells(self):
        self.assertIs(literal_assignments()["RUN_M5D_DIAGNOSTICS"],False)
        self.assertIs(literal_assignments()["SHOW_M5D_ERROR_IMAGES"],False)
        cells=[c for c in load_notebook()["cells"] if c["id"].startswith("m5d-")]
        self.assertGreaterEqual(len(cells),4)
        source="\n".join("".join(c["source"]) for c in cells if c["cell_type"]=="code")
        tree=ast.parse(source)
        for n in ast.walk(tree):
            if isinstance(n,ast.Call):
                if isinstance(n.func,ast.Name):
                    self.assertNotIn(n.func.id,{"train_one_epoch","run_experiment_sequence","save_checkpoint",
                                               "save_recovery_checkpoint","exec","eval","acquire_official_weights"})
                if isinstance(n.func,ast.Attribute):self.assertNotIn(n.func.attr,{"backward","step"})
        self.assertIn("torch.inference_mode",source)
        self.assertNotIn("DPD_Small_test_",source)
        self.assertIn("if RUN_M5D_DIAGNOSTICS:",source)

    def test_state_fingerprints_detect_buffer_and_parameter_changes(self):
        model=nn.BatchNorm1d(3)
        before=self.ns["d_state_digest"](model.state_dict())
        with torch.no_grad():model.running_mean.add_(1)
        self.assertNotEqual(before,self.ns["d_state_digest"](model.state_dict()))
        self.assertEqual(self.ns["d_state_digest"](model.state_dict()),
                         self.ns["d_state_digest"]({k:v.clone() for k,v in model.state_dict().items()}))

    def test_support_structure_checks_every_heldout_pair(self):
        split=self.ns["build_development_split"](ROOT/"DPD_Small_train.csv")
        support=self.ns["d_support_structure"](split)
        self.assertEqual(support["unseen_images"],1894)
        self.assertTrue(support["all_components_have_fitting_examples"])
        self.assertTrue(support["all_heldout_pairs_absent_from_fitting"])
        self.assertEqual(sum(r["unseen_images"] for r in support["pairs"]),1894)
        self.assertEqual(sum(support["disease_heldout_image_support_bands"].values()),1894)

    def test_prediction_packet_recomputes_and_rejects_changed_truths(self):
        f=self.frame();c,d=self.logits();spec=self.spec()
        split=SimpleNamespace(fit=f,val_seen=f,val_unseen=f,candidate_ids=spec["ids"],
                              candidate_pairs=spec["pairs"],
                              candidate_sha256=self.ns["d_candidate_records"](spec)[2])
        report=self.ns["d_fold_report"](c,d,f,f,spec)["summary"]
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            (directory/"profile_summary.json").write_text(json.dumps({"profile":"synthetic",
                "folds":{"seen":report,"unseen":report}}))
            for fold in ("seen","unseen"):
                np.savez_compressed(directory/(fold+"_logits.npz"),crop_logits=c.numpy(),
                    disease_logits=d.numpy(),crop_truth=f.crop_id.to_numpy(),
                    disease_truth=f.disease_id.to_numpy(),composition_truth=f.composition_id.to_numpy())
            got=self.ns["d_review_prediction_packet"](directory,split)
            self.assertTrue(got["checks"]["unseen"]["all_summary_fields_match"])
            tie_audit=self.ns["d_audit_numerical_ties"](directory,split)
            self.assertFalse(tie_audit["policy_changed"])
            self.assertEqual(tie_audit["top5_hit_disagreement_rows"],0)
            np.savez_compressed(directory/"unseen_logits.npz",crop_logits=c.numpy(),
                    disease_logits=d.numpy(),crop_truth=f.crop_id.to_numpy(),
                    disease_truth=f.disease_id.to_numpy(),composition_truth=np.array([90,90,90]))
            with self.assertRaises(ValueError):
                self.ns["d_review_prediction_packet"](directory,split)

    def test_recorded_batch_check_uses_the_saved_epoch_microbatch(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);run=base/"run";run.mkdir()
            (run/"run_summary.json").write_text(json.dumps({
                "selected":{"path":"unused.pt","epoch":1,"metrics":{"h1":.1}},
                "history":[{"epoch":1,"physical_microbatch":64}]}))
            calls=[]
            def fake(*args,**kwargs):
                calls.append(kwargs["batch_size"])
                return {"metrics":{"h1":.1},"folds":{"seen":{"sample_count":2},"unseen":{"sample_count":3}}}
            with patch.dict(self.ns,{"d_run_profile":fake}):
                result=self.ns["d_run_recorded_batch_checks"]([run],None,None,base/"out",device="cpu")
            self.assertEqual(calls,[64])
            self.assertEqual(result["image_evaluations"],5)

    def test_preprocessing_description_defect_is_reported_without_rewriting(self):
        actual=repr(self.ns["get_transforms"](380,augment=False,normalize_imagenet=True))
        metadata={"image_size":[380,380],"validation_transform":
                  repr(self.ns["get_transforms"](224,augment=False,normalize_imagenet=True))}
        info={"architecture":"multitask_efficientnet_b3","kind":"final","epoch":23,
              "native_resolution":380,"preprocessing":metadata}
        before=copy.deepcopy(info)
        audit=self.ns["d_preprocessing_audit"](info)
        self.assertFalse(audit["metadata_consistent"])
        self.assertFalse(audit["validation_description_matches_actual_loader"])
        self.assertTrue(audit["image_size_describes_actual_epoch"])
        self.assertEqual(info,before)
        info["preprocessing"]["validation_transform"]=actual
        self.assertTrue(self.ns["d_preprocessing_audit"](info)["metadata_consistent"])
        info["native_resolution"]=300
        self.assertFalse(self.ns["d_preprocessing_audit"](info)["image_size_describes_actual_epoch"])

    def test_predicted_component_frequencies_are_compared_with_fitting_support(self):
        f=self.frame();c,d=self.logits();report=self.ns["d_fold_report"](c,d,f,f,self.spec())
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            for fold in ("seen","unseen"):
                pd.DataFrame(report["samples"]).to_csv(directory/(fold+"_samples.csv"),index=False)
            result=self.ns["d_prediction_support_summary"](directory,SimpleNamespace(fit=f))
            disease=result["folds"]["unseen"]["disease"]
            self.assertEqual(sum(x["predicted_argmax_images"] for x in disease["classes"]),3)
            self.assertEqual(disease["most_predicted_classes"][0]["class_id"],0)
            self.assertEqual(disease["most_predicted_classes"][0]["predicted_argmax_images"],2)
            self.assertEqual(disease["most_predicted_classes"][0]["fit_images"],1)

if __name__=="__main__":
    unittest.main()
