"""Offline WP2 wrappers for the two selected Torchvision backbones.

Future approved notebooks may pass a verified local *stock Torchvision* state
dictionary as backbone_state_dict (load tensors with torch.load(weights_only=True)).
It is loaded strictly before removing the ImageNet classifier; new task heads
are randomly initialized. No constructor fetches weights or reads files.

Intended initialization: EfficientNet_B0_Weights.IMAGENET1K_V1 and
ConvNeXt_Tiny_Weights.IMAGENET1K_V1. Their installed .transforms() specify RGB
float input in [0,1], mean (0.485,0.456,0.406), std (0.229,0.224,0.225):
B0 resize 256/bicubic then center crop 224; Tiny resize 236/bilinear then
center crop 224. Notebook training augmentation must be agreed separately.
No preprocessing is applied inside these models.
"""

from collections.abc import Mapping

from torch import Tensor, nn
from torchvision.models import convnext_tiny, efficientnet_b0

from multitask_model import MultiTaskResNet18


MODEL_FEATURE_DIMS = {
    "multitask_resnet18": 512,
    "multitask_efficientnet_b0": 1280,
    "multitask_convnext_tiny": 768,
}


class MultiTaskBackbone(nn.Module):
    """One shared feature extractor and two independent raw-logit heads."""

    def __init__(
        self,
        architecture: str,
        *,
        backbone_state_dict: Mapping[str, Tensor] | None = None,
    ) -> None:
        super().__init__()
        if architecture == "multitask_efficientnet_b0":
            backbone = efficientnet_b0(weights=None)
            feature_dim = backbone.classifier[-1].in_features
        elif architecture == "multitask_convnext_tiny":
            backbone = convnext_tiny(weights=None)
            feature_dim = backbone.classifier[-1].in_features
        else:
            raise ValueError("Select multitask_efficientnet_b0 or multitask_convnext_tiny")
        if backbone_state_dict is not None:
            backbone.load_state_dict(backbone_state_dict, strict=True)
        if architecture == "multitask_efficientnet_b0":
            # Torchvision already pools and flattens before this classifier.
            # Remove both classifier dropout and the ImageNet linear layer.
            backbone.classifier = nn.Identity()
        else:
            # Retain final LayerNorm2d and flatten after global pooling.
            backbone.classifier[-1] = nn.Identity()
        self.architecture = architecture
        self.feature_dim = feature_dim
        self.backbone = backbone
        self.crop_head = nn.Linear(feature_dim, 55)
        self.disease_head = nn.Linear(feature_dim, 175)

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        """Compute features once; export them before either prediction head."""
        features = self.backbone(images)
        return {
            "crop_logits": self.crop_head(features),
            "disease_logits": self.disease_head(features),
            "features": features,
        }


def create_multitask_model(
    architecture: str,
    *,
    backbone_state_dict: Mapping[str, Tensor] | None = None,
) -> MultiTaskResNet18 | MultiTaskBackbone:
    """Construct an offline model, retaining the accepted ResNet18 reference."""
    if not isinstance(architecture, str) or architecture not in MODEL_FEATURE_DIMS:
        raise ValueError(f"Unsupported multi-task architecture: {architecture!r}")
    if architecture == "multitask_resnet18":
        if backbone_state_dict is not None:
            raise ValueError("Local stock state initialization is supported only for B0 and Tiny")
        return MultiTaskResNet18()
    return MultiTaskBackbone(architecture, backbone_state_dict=backbone_state_dict)
