"""Import-safe WP2 losses, explicit epoch functions and model checkpoints."""

import json
import math
import os
import platform
import tempfile
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import torch
import torchvision
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from multitask_model import MultiTaskResNet18
from multitask_backbones import MODEL_FEATURE_DIMS, MultiTaskBackbone, create_multitask_model


_FLOAT_DTYPES = {
    str(dtype): dtype
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
}


def _validate_weight(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    weight = float(value)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return weight


def _classification_loss(logits: Tensor, labels: Tensor, classes: int, name: str) -> Tensor:
    if not isinstance(logits, Tensor) or not logits.is_floating_point():
        raise TypeError(f"{name} logits must be floating-point tensors")
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] != classes:
        raise ValueError(f"{name} logits must have shape [B, {classes}] with B > 0")
    if not isinstance(labels, Tensor) or labels.dtype != torch.int64:
        raise TypeError(f"{name} labels must be int64 tensors")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError(f"{name} labels must have shape [B] matching logits")
    if labels.device != logits.device:
        raise ValueError(f"{name} labels and logits must be on the same device")
    if ((labels < 0) | (labels >= classes)).any().item():
        raise ValueError(f"{name} labels must be in [0, {classes})")
    if not torch.isfinite(logits).all().item():
        raise ValueError(f"{name} logits must be finite")
    return F.cross_entropy(logits, labels, reduction="mean")


def compute_multitask_loss(
    outputs: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
    *,
    crop_weight: float = 0.5,
    disease_weight: float = 0.5,
) -> dict[str, Tensor]:
    """Return total/crop/disease losses from raw logits and int64 component IDs.

    Weights need not sum to one, but must be finite, nonnegative and not both
    zero. Extra output/target fields, including composition IDs, are ignored.
    """
    crop_weight = _validate_weight(crop_weight, "crop_weight")
    disease_weight = _validate_weight(disease_weight, "disease_weight")
    if crop_weight == disease_weight == 0:
        raise ValueError("At least one loss weight must be positive")
    if not isinstance(outputs, Mapping) or not isinstance(targets, Mapping):
        raise TypeError("outputs and targets must be mappings")
    losses = {}
    for name, classes in (("crop", 55), ("disease", 175)):
        if f"{name}_logits" not in outputs or name not in targets:
            raise ValueError(f"Missing {name} logits or labels")
        losses[name] = _classification_loss(
            outputs[f"{name}_logits"], targets[name], classes, name
        )
    crop_logits, disease_logits = outputs["crop_logits"], outputs["disease_logits"]
    if (crop_logits.shape[0] != disease_logits.shape[0]
            or crop_logits.device != disease_logits.device):
        raise ValueError("Crop and disease logits require matching batches and devices")
    total = crop_weight * losses["crop"] + disease_weight * losses["disease"]
    if not all(torch.isfinite(value).item() for value in (*losses.values(), total)):
        raise ValueError("Loss calculation produced a non-finite value")
    return {"total": total, **losses}


def _class_mappings(
    mappings: Mapping[str, Any] | None,
) -> dict[str, dict[int, str]] | None:
    """Normalize component ID/name maps from the existing data utilities."""
    if mappings is None:
        return None
    if not isinstance(mappings, Mapping):
        raise TypeError("class_mappings must be a mapping")
    result = {}
    for name, count in (("crop", 55), ("disease", 175)):
        key = f"{name}_id_to_name"
        values = mappings.get(key)
        if not isinstance(values, Mapping):
            raise ValueError(f"class_mappings requires {key}")
        if (any(isinstance(i, bool) or not isinstance(i, Integral) for i in values)
                or set(values) != set(range(count))):
            raise ValueError(f"{key} requires exactly the IDs 0..{count - 1}")
        names = [values[i] for i in range(count)]
        if (any(not isinstance(value, str) or not value.strip() for value in names)
                or len(set(names)) != count):
            raise ValueError(f"{key} requires unique, nonempty string names")
        result[key] = {i: str(value) for i, value in enumerate(names)}
    return result


def _plain_metadata(value: Mapping[str, Any] | None, name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a mapping with string keys")
    try:
        return json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite JSON-compatible values") from exc


def _checkpoint_config(architecture: str) -> dict[str, Any]:
    if not isinstance(architecture, str) or architecture not in MODEL_FEATURE_DIMS:
        raise ValueError(f"Unsupported checkpoint architecture: {architecture!r}")
    return {
        "format_version": 1,
        "architecture": architecture,
        "num_crops": 55,
        "num_diseases": 175,
        "feature_dim": MODEL_FEATURE_DIMS[architecture],
    }


def save_checkpoint(
    model: MultiTaskResNet18 | MultiTaskBackbone,
    path: str | Path,
    *,
    class_mappings: Mapping[str, Any] | None = None,
    preprocessing: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Atomically save CPU state and plain metadata to an existing directory.

    Existing paths are protected unless overwrite=True. Component maps use
    crop_id_to_name and disease_id_to_name; other mapping fields are ignored.
    Optimizer state and exact training resumption are outside this format.
    """
    if isinstance(model, MultiTaskResNet18):
        architecture = "multitask_resnet18"
    elif isinstance(model, MultiTaskBackbone):
        architecture = model.architecture
    else:
        raise TypeError("model must be a supported WP2 multi-task model")
    config = _checkpoint_config(architecture)
    dim = config["feature_dim"]
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a bool")
    if (model.crop_head.in_features, model.crop_head.out_features,
            model.disease_head.in_features, model.disease_head.out_features) != (dim, 55, dim, 175):
        raise ValueError("Model heads are incompatible with the checkpoint architecture")
    payload = {
        **config,
        "class_mappings": _class_mappings(class_mappings),
        "preprocessing": _plain_metadata(preprocessing, "preprocessing"),
        "metadata": _plain_metadata(metadata, "metadata") or {},
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
        },
    }
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    state = model.state_dict()
    dtypes = {str(value.dtype) for value in state.values() if value.is_floating_point()}
    if len(dtypes) != 1 or not dtypes <= _FLOAT_DTYPES.keys():
        raise ValueError("Checkpoint requires one supported floating-point state dtype")
    payload["model_dtype"] = next(iter(dtypes))
    payload["model_state_dict"] = {
        name: value.detach().cpu().clone() for name, value in state.items()
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            # Atomic publication without a check-then-overwrite race on WSL.
            os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    expected_class_mappings: Mapping[str, Any] | None = None,
    expected_preprocessing: Mapping[str, Any] | None = None,
    expected_architecture: str | None = None,
    device: str | torch.device = "cpu",
) -> tuple[MultiTaskResNet18 | MultiTaskBackbone, dict[str, Any]]:
    """Load state strictly, returning an evaluation-mode model and metadata.

    Labelled checkpoints require expected_class_mappings so ID/name changes
    cannot be silently accepted. Preprocessing is compared when supplied.
    Loading is restricted to tensors/plain data; there is no unsafe fallback.
    Version-1 ResNet18 payloads keep their original schema and state keys.
    All architectures reconstruct offline; expected_architecture optionally
    guards the caller against loading another supported model by mistake.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must contain a dictionary")
    architecture = payload.get("architecture")
    config = _checkpoint_config(architecture)
    if expected_architecture is not None:
        _checkpoint_config(expected_architecture)
        if architecture != expected_architecture:
            raise ValueError("Checkpoint architecture does not match expected architecture")
    for key, expected in config.items():
        value = payload.get(key)
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"Incompatible checkpoint {key}: {value!r}")
    required = {"model_state_dict", "model_dtype", "class_mappings",
                "preprocessing", "metadata", "runtime"}
    if not required <= payload.keys():
        raise ValueError("Checkpoint is missing required state or metadata")
    dtype = payload["model_dtype"]
    if not isinstance(dtype, str) or dtype not in _FLOAT_DTYPES:
        raise ValueError("Unsupported checkpoint model_dtype")
    stored_mappings = _class_mappings(payload["class_mappings"])
    expected_mappings = _class_mappings(expected_class_mappings)
    if stored_mappings is not None and expected_mappings is None:
        raise ValueError("Labelled checkpoints require expected_class_mappings")
    if stored_mappings != expected_mappings:
        raise ValueError("Checkpoint class mappings do not match expected class IDs/names")
    preprocessing = _plain_metadata(payload["preprocessing"], "preprocessing")
    if (expected_preprocessing is not None
            and preprocessing != _plain_metadata(expected_preprocessing, "expected_preprocessing")):
        raise ValueError("Checkpoint preprocessing does not match expected preprocessing")
    info = {
        **payload,
        "class_mappings": stored_mappings,
        "preprocessing": preprocessing,
        "metadata": _plain_metadata(payload["metadata"], "metadata"),
        "runtime": _plain_metadata(payload["runtime"], "runtime"),
    }
    state = info.pop("model_state_dict")
    if (not isinstance(state, Mapping)
            or any(not isinstance(value, Tensor) for value in state.values())):
        raise ValueError("Checkpoint model_state_dict must map names to tensors")
    if any(value.is_floating_point() and str(value.dtype) != dtype for value in state.values()):
        raise ValueError("Checkpoint state dtype does not match model_dtype")
    model = create_multitask_model(architecture).to(dtype=_FLOAT_DTYPES[dtype])
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, info



def _epoch_setup(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str | torch.device,
    crop_weight: float,
    disease_weight: float,
) -> tuple[torch.device, float, float]:
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(loader, DataLoader):
        raise TypeError("loader must be a DataLoader")
    crop_weight = _validate_weight(crop_weight, "crop_weight")
    disease_weight = _validate_weight(disease_weight, "disease_weight")
    if crop_weight == disease_weight == 0:
        raise ValueError("At least one loss weight must be positive")
    device = torch.device(device)
    if device.type == "cpu":
        device = torch.device("cpu")
    elif device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    for value in (*model.parameters(), *model.buffers()):
        if value.device != device:
            raise ValueError("Move the model to the requested device before calling the epoch function")
    return device, crop_weight, disease_weight


def _epoch_batch(
    batch: Any, device: torch.device, *, require_composition: bool,
) -> tuple[Tensor, dict[str, Tensor]]:
    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise ValueError("Each batch must contain (inputs, targets)")
    inputs, raw_targets = batch
    if (not isinstance(inputs, Tensor) or not inputs.is_floating_point()
            or inputs.ndim < 2 or inputs.shape[0] == 0):
        raise ValueError("Inputs must be a nonempty floating-point batched tensor")
    if not torch.isfinite(inputs).all().item():
        raise ValueError("Inputs must be finite")
    if isinstance(raw_targets, Mapping):
        targets = {key: raw_targets[key] for key in ("crop", "disease", "composition")
                   if key in raw_targets}
    elif isinstance(raw_targets, (tuple, list)) and len(raw_targets) in (2, 3):
        targets = dict(zip(("crop", "disease", "composition"), raw_targets))
    else:
        raise TypeError("Targets must be a mapping or a crop/disease[/composition] sequence")
    required = ("crop", "disease", "composition") if require_composition else ("crop", "disease")
    if any(key not in targets for key in required):
        raise ValueError(f"Targets require {', '.join(required)}")
    for name, value in targets.items():
        if not isinstance(value, Tensor) or value.dtype != torch.int64:
            raise TypeError(f"{name} targets must be int64 tensors")
        if value.ndim != 1 or value.shape[0] != inputs.shape[0]:
            raise ValueError(f"{name} targets must have shape [B] matching inputs")
        if name == "composition" and (value < 0).any().item():
            raise ValueError("Global composition IDs must be nonnegative")
    return inputs.to(device), {key: value.to(device) for key, value in targets.items()}


def _epoch_summary(
    totals: dict[str, float], sample_count: int, batch_count: int,
) -> dict[str, Any]:
    if sample_count == 0:
        raise ValueError("Loader produced no samples; check empty data and drop_last")
    means = {key: value / sample_count for key, value in totals.items()}
    if not all(math.isfinite(value) for value in means.values()):
        raise ValueError("Epoch aggregation produced non-finite losses")
    return {
        "sample_count": sample_count,
        "batch_count": batch_count,
        "losses": means,
    }


def train_one_epoch(
    model: torch.nn.Module,
    fitting_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
    *,
    crop_weight: float = 0.5,
    disease_weight: float = 0.5,
) -> dict[str, Any]:
    """Run one explicitly requested fitting pass; never called automatically.

    Caller supplies a model already on device and an optimizer bound to its
    parameters. Only inputs/targets are moved. Each successful batch contributes
    its pre-step mean loss times its sample count. Optional features are ignored.
    Failures are not transactional: earlier steps and training-mode buffers may
    already have changed. No scheduler, checkpoint, logging or multi-epoch loop
    is started here.
    """
    device, crop_weight, disease_weight = _epoch_setup(
        model, fitting_loader, device, crop_weight, disease_weight
    )
    if torch.is_inference_mode_enabled():
        raise ValueError("Training cannot run inside torch.inference_mode")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch.optim.Optimizer")
    model_ids = {id(parameter) for parameter in model.parameters()}
    optimizer_ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    if (not optimizer_ids or len(optimizer_ids) != len(set(optimizer_ids))
            or not set(optimizer_ids) <= model_ids):
        raise ValueError("Optimizer parameters must be unique and belong to the supplied model")
    model.train()
    totals = {key: 0.0 for key in ("total", "crop", "disease")}
    sample_count = batch_count = 0
    with torch.enable_grad():
        for batch in fitting_loader:
            inputs, targets = _epoch_batch(batch, device, require_composition=False)
            # Clear all model gradients, including any parameters excluded from
            # the optimizer, to avoid unintended accumulation between batches.
            model.zero_grad(set_to_none=True)
            losses = compute_multitask_loss(
                model(inputs), targets, crop_weight=crop_weight, disease_weight=disease_weight
            )
            if not losses["total"].requires_grad:
                raise ValueError("Training loss has no gradient graph")
            losses["total"].backward()
            for parameter in model.parameters():
                if parameter.grad is not None:
                    gradient = parameter.grad
                    values = gradient.coalesce().values() if gradient.is_sparse else gradient
                    if not torch.isfinite(values).all().item():
                        model.zero_grad(set_to_none=True)
                        raise ValueError("Non-finite gradients; optimizer step was not executed")
            optimizer.step()
            size = inputs.shape[0]
            for key, value in losses.items():
                totals[key] += value.detach().item() * size
            sample_count += size
            batch_count += 1
    return _epoch_summary(totals, sample_count, batch_count)


def validate_one_epoch(
    model: torch.nn.Module,
    validation_loader: DataLoader,
    device: str | torch.device,
    *,
    crop_weight: float = 0.5,
    disease_weight: float = 0.5,
) -> dict[str, Any]:
    """Evaluate once, returning sample-weighted losses and CPU predictions.

    Sets eval mode and uses inference_mode without clearing gradients or
    stepping an optimizer. CPU logits and int64 composition_ids preserve loader
    iteration order. Future WP5 callers supply the shared candidate IDs/pairs;
    no composition scoring, Top-k or harmonic mean is implemented here.
    """
    device, crop_weight, disease_weight = _epoch_setup(
        model, validation_loader, device, crop_weight, disease_weight
    )
    model.eval()
    totals = {key: 0.0 for key in ("total", "crop", "disease")}
    sample_count = batch_count = 0
    crop_parts, disease_parts, composition_parts = [], [], []
    with torch.inference_mode():
        for batch in validation_loader:
            inputs, targets = _epoch_batch(batch, device, require_composition=True)
            outputs = model(inputs)
            losses = compute_multitask_loss(
                outputs, targets, crop_weight=crop_weight, disease_weight=disease_weight
            )
            size = inputs.shape[0]
            for key, value in losses.items():
                totals[key] += value.item() * size
            sample_count += size
            batch_count += 1
            # Snapshot each batch even when a model/loader reuses CPU storage.
            crop_parts.append(outputs["crop_logits"].detach().to("cpu", copy=True))
            disease_parts.append(outputs["disease_logits"].detach().to("cpu", copy=True))
            composition_parts.append(targets["composition"].detach().to("cpu", copy=True))
    result = _epoch_summary(totals, sample_count, batch_count)
    result.update(
        crop_logits=torch.cat(crop_parts),
        disease_logits=torch.cat(disease_parts),
        composition_ids=torch.cat(composition_parts),
    )
    return result
