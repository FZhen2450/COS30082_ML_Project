"""UI-independent prediction presentation and training-reference retrieval.

The reference browser is usable without checkpoints. Its colour histogram is
only a visual ordering of labelled examples, never a disease predictor.
"""
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageOps


def open_rgb(path):
    with Image.open(path) as image:
        image.load()
        return ImageOps.exif_transpose(image).convert("RGB")


def colour_signature(image):
    small = image.convert("RGB").resize((64, 64))
    histogram = small.histogram()
    bins = [sum(histogram[channel * 256 + start:channel * 256 + start + 32])
            for channel in range(3) for start in range(0, 256, 32)]
    total = sum(bins)
    return [value / total for value in bins]


class Catalog:
    def __init__(self, protocol_path, source_csv, images):
        # This validates filename partitions and candidate IDs against train.csv
        # without reading either official test CSV or loading a neural network.
        from shared_protocol import validate_protocol
        self.protocol_path = Path(protocol_path).resolve()
        self.source_csv = Path(source_csv).resolve()
        self.images = Path(images).resolve()
        self.protocol = json.loads(self.protocol_path.read_text(encoding="utf-8"))
        fitting, _, _ = validate_protocol(self.protocol, self.source_csv)
        self.fitting = fitting.to_dict("records")
        self.candidates = {int(c["composition_id"]): c for c in self.protocol["candidates"]}

    def image_path(self, name):
        path = (self.images / name).resolve()
        if path.parent != self.images:
            raise ValueError("Reference image must be inside the flat dataset folder")
        return path

    def references(self, composition_id, query=None, exclude_path=None, limit=4):
        candidate = self.candidates[int(composition_id)]
        rows = [r for r in self.fitting if r["composition_id"] == int(composition_id)]
        relation = "Same crop and disease"
        if not rows:
            rows = [r for r in self.fitting if r["disease_id"] == candidate["disease_id"]]
            relation = "Same disease; different crop"
        query_signature = colour_signature(query) if query is not None else None
        query_hash = hashlib.sha256(query.tobytes()).digest() if query is not None else None
        excluded = Path(exclude_path).resolve() if exclude_path else None
        found, unavailable = [], 0
        # Metadata order and filename ties are deterministic. No validation/test
        # image can enter this pool, even if present in the dataset directory.
        for row in sorted(rows, key=lambda r: r["image_name"]):
            path = self.image_path(row["image_name"])
            if path == excluded:
                continue
            try:
                image = open_rgb(path)
            except (OSError, ValueError, Image.DecompressionBombError):
                unavailable += 1
                continue
            if (query_hash is not None and image.size == query.size and
                    hashlib.sha256(image.tobytes()).digest() == query_hash):
                continue
            distance = (sum((a - b) ** 2 for a, b in
                            zip(query_signature, colour_signature(image)))
                        if query_signature is not None else 0.)
            image.thumbnail((230, 150))
            found.append({"filename": row["image_name"], "image": image,
                          "crop": row["crop_name"], "disease": row["disease_name"],
                          "relation": relation, "distance": distance})
        found.sort(key=lambda r: (r["distance"], r["filename"]))
        return {"items": found[:limit], "relation": relation, "unavailable": unavailable,
                "pool_size": len(rows), "query_ordered": query is not None}


def present_prediction(result, catalog):
    """Resolve global IDs by score columns, with stable global-ID tie breaks."""
    ids = result["candidate_ids"].tolist()
    matrix = result["scores"].tolist()
    expected = list(catalog.candidates)
    if ids != expected or len(matrix) != 1 or len(matrix[0]) != len(ids):
        raise ValueError("Model candidate columns do not match the loaded protocol")
    if result.get("candidate_sha256") != catalog.protocol["candidate_sha256"]:
        raise ValueError("Model and browser candidate fingerprints differ")
    if not all(math.isfinite(value) for value in matrix[0]):
        raise ValueError("Model returned non-finite scores")
    ranking = sorted(zip(ids, matrix[0]), key=lambda pair: (-pair[1], pair[0]))[:5]
    return [{**catalog.candidates[cid], "score": float(score)} for cid, score in ranking]


def load_independent(catalog, crop, disease, device="cpu"):
    # No automatic pretrained download; the existing loader rebuilds checkpoints.
    from independent_baseline import IndependentPredictor
    return IndependentPredictor(crop, disease, catalog.protocol_path,
                                catalog.source_csv, device=device)
