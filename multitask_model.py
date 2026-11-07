"""WP2 shared ResNet18 with crop and disease classification heads."""

from torch import Tensor, nn
from torchvision.models import resnet18


class MultiTaskResNet18(nn.Module):
    """Return raw crop/disease logits and the shared 512-dimensional features.

    Construction uses random initialization and never downloads weights.
    Inputs follow the existing float32 RGB [B, 3, 224, 224] data contract.
    """

    def __init__(self) -> None:
        super().__init__()
        self.backbone = resnet18(weights=None)
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.crop_head = nn.Linear(feature_dim, 55)
        self.disease_head = nn.Linear(feature_dim, 175)

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        """Compute the backbone once and pass its features to both heads."""
        features = self.backbone(images)
        return {
            "crop_logits": self.crop_head(features),
            "disease_logits": self.disease_head(features),
            "features": features,
        }
