"""CLIP encoder wrapper for image feature extraction."""

import torch
from transformers import CLIPProcessor, CLIPModel


class CLIPEncoder:
    """Frozen CLIP image encoder.

    Loads CLIP ViT-B/32 and provides a simple interface for extracting
    512-dimensional image embeddings.
    """

    def __init__(self, device: str = "cpu"):
        """Initialize CLIP encoder.

        Args:
            device: Device to run on ("cpu", "cuda", or "mps")
        """
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.to(device)
        self.model.eval()

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def encode_image(self, image) -> torch.Tensor:
        """Extract image features using CLIP.

        Args:
            image: PIL Image or similar format accepted by CLIPProcessor

        Returns:
            512-dimensional image embedding tensor
        """
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        return outputs.squeeze(0)  # (512,)


# Singleton pattern for shared encoder
_clip_encoder = None


def get_clip_encoder(device: str = "cpu") -> CLIPEncoder:
    """Get shared CLIP encoder instance.

    Args:
        device: Device to run on (only used on first call)

    Returns:
        Shared CLIPEncoder instance
    """
    global _clip_encoder
    if _clip_encoder is None:
        _clip_encoder = CLIPEncoder(device)
    return _clip_encoder


def reset_clip_encoder():
    """Reset the shared encoder (useful for testing)."""
    global _clip_encoder
    _clip_encoder = None
