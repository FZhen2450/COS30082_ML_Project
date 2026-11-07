"""Presentation of validated four-model results; no inference or metric rescoring."""
import csv
import io
import json
from dataclasses import dataclass

from evaluation.model_adapters import MODEL_KINDS, import_model_packets
from evaluation.reporting import canonical_json, fingerprint


@dataclass(frozen=True, init=False)
class VerifiedResults:
    """Immutable snapshot minted by the existing complete-record validator.

    This guards the presentation API against accidental unvalidated dictionaries;
    it is not authentication of owner declarations or a Python security boundary.
    """
    _json: str

    def __init__(self):
        raise TypeError("Use VerifiedResults.from_packets with complete identified packets")

    @classmethod
    def from_packets(cls, manifest, provenances, packets, *, final_evaluation_reference=""):
        result = import_model_packets(manifest, provenances, packets,
                                      final_evaluation_reference=final_evaluation_reference)
        snapshot = object.__new__(cls)
        object.__setattr__(snapshot, "_json", canonical_json(result))
        return snapshot

    def to_dict(self):
        return json.loads(self._json)

    @property
    def sha256(self):
        return fingerprint(self.to_dict())


def result_scope(spec):
    if spec["synthetic"]:
        return "SYNTHETIC TEST FIXTURES — not empirical model performance"
    if spec["task_mode"] == "development":
        return "TRAINING-DERIVED DEVELOPMENT — internal seen/unseen validation"
    return "OFFICIAL TEST — separately authorized final evaluation"


def _csv(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def presentation_files(results):
    """Return fully prepared filename/bytes artifacts ONLY for verified results.

    Caller writes these to a new directory after all imports/plots have succeeded.
    Fractions stay on [0,1]; no significance or model-selection claims are made.
    """
    if type(results) is not VerifiedResults:
        raise TypeError("Presentation requires VerifiedResults.from_packets, not a result dictionary")
    imported = results.to_dict()
    first = imported["models"][MODEL_KINDS[0]]
    spec = first["candidate_spec"]
    scope = result_scope(spec)
    source = {"scope": scope, "manifest_sha256": imported["manifest_sha256"],
              "candidate_sha256": imported["candidate_sha256"], "results_sha256": results.sha256}
    comparison, health, classes, paired = [], [], [], []
    for kind in MODEL_KINDS:
        report = imported["models"][kind]
        comparison.append({"model": kind, "seen_n": report["seen"]["expected"],
                           "unseen_n": report["unseen"]["expected"],
                           **{fold + "_top" + str(k): report[fold]["top" + str(k)]
                              for fold in ("seen", "unseen") for k in (1, 5)},
                           "H@1": report["harmonic_mean_top1"], "H@5": report["harmonic_mean_top5"], **source})
        for fold in ("seen", "unseen"):
            for group, row in report[fold]["health"].items():
                health.append({"model": kind, "fold": fold, "truth_group": group,
                               "denominator": row["count"],
                               **{"errors_top" + str(k): row["count"] - row["correct_top" + str(k)] for k in (1, 5)},
                               "false_healthy_errors": row["false_healthy_error_count"], **source})
            for grouping in ("crop", "disease", "composition"):
                for row in report[fold]["per_" + grouping]:
                    classes.append({"model": kind, "fold": fold, "grouping": grouping,
                                    "class_id": row[grouping + "_id"], "class_name": row["name"],
                                    "denominator": row["count"],
                                    **{"errors_top" + str(k): row["count"] - row["correct_top" + str(k)] for k in (1, 5)},
                                    "fitting_support": row["fitting_support"], "low_support": row["low_support"], **source})
    for name, report in imported["paired"].items():
        for row in report["paired"]:
            if row["composition_id"] is None:
                paired.append({"pair": name, "fold": row["fold"], "k": row["k"],
                               "denominator": row["count"], "a_wins": row["a_only"], "b_wins": row["b_only"],
                               "both_correct": row["both_correct"], "neither_correct": row["neither"],
                               "b_minus_a_hits": row["b_minus_a_hits"], **source})
    footer = (f"Source: WP5 validated owner packets; identity source: {first['identity_source']}.\n"
              f"Manifest SHA-256: {imported['manifest_sha256']}\n"
              f"Candidate SHA-256: {imported['candidate_sha256']} ({spec['candidate_count']} ordered candidates)\n"
              f"Verified result SHA-256: {results.sha256}\n"
              f"Seen denominator: {first['seen']['expected']}; unseen denominator: {first['unseen']['expected']}.\n"
              "H@k is the harmonic mean of the two fold fractions, not pooled accuracy.\n"
              "Class/health errors are composition errors grouped by ground truth; zero support is not accuracy zero.\n"
              "Paired wins mean only that model is correct on the identical image; no significance claim.\n"
              "Full run/checkpoint/transform/scorer/environment provenance: each model's report.json.\n")
    columns = [k for k in comparison[0] if k not in source]
    markdown = [scope, "", "All accuracies are fractions [0,1].", "",
                "| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in comparison:
        markdown.append("| " + " | ".join(format(row[c], ".6f") if isinstance(row[c], float) else str(row[c])
                                          for c in columns) + " |")
    markdown += ["", footer]
    files = {"comparison.csv": _csv(comparison), "comparison.md": "\n".join(markdown).encode("utf-8"),
             "health_errors.csv": _csv(health), "per_class_errors.csv": _csv(classes),
             "paired_wins.csv": _csv(paired), "caption.txt": (scope + "\n\n" + footer).encode("utf-8")}
    # Existing plotting dependency, headless and in memory. Never install/download.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=(12, 7), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    labels = ("WP1 pair", "WP2 Base", "CLIP frozen", "CLIP adapter")
    for axis, k in zip(axes, (1, 5)):
        for offset, metric, label in ((-.24, f"seen_top{k}", "Seen"), (0, f"unseen_top{k}", "Unseen"),
                                       (.24, f"H@{k}", f"H@{k}")):
            axis.bar([i + offset for i in range(4)], [r[metric] for r in comparison], width=.24, label=label)
        axis.set(ylim=(0, 1), ylabel="Composition accuracy (fraction)", title=f"Top-{k}",
                 xticks=range(4), xticklabels=labels)
        axis.tick_params(axis="x", labelrotation=20)
        axis.legend()
    figure.suptitle(f"{scope}\nSeen n={first['seen']['expected']}; unseen n={first['unseen']['expected']}")
    figure.supxlabel(f"WP5 verified packets | {spec['candidate_count']} candidates | manifest {imported['manifest_sha256']}\n"
                     "Full provenance and interpretation: caption.txt and model report.json", fontsize=8)
    png = io.BytesIO()
    figure.savefig(png, format="png", dpi=150, metadata={"Software": "WP5 verified result presentation"})
    files["comparison.png"] = png.getvalue()
    return files
