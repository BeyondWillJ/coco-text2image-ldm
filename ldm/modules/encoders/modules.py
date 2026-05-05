"""
Text (and image) encoders for cross-attention conditioning.

Provides:
  - CLIPTextEmbedder  – wraps HuggingFace CLIP text encoder
  - FrozenCLIPEmbedder – alias with frozen weights (used during LDM training)
  - AbstractEncoder   – base class
"""

import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class ClassEmbedder(AbstractEncoder):
    """Embed class labels into a conditioning vector."""

    def __init__(self, embed_dim, n_classes=1000, key="class"):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        c = batch[key or self.key][:, None]
        return self.embedding(c)


class FrozenCLIPEmbedder(AbstractEncoder):
    """
    CLIP text encoder with frozen weights, used to condition the UNet.

    Tokenises input strings and returns the last-hidden-state sequence from the
    CLIP text transformer (shape: B × 77 × 768 for ViT-L/14).
    """

    CLIP_VERSION = "openai/clip-vit-large-patch14"

    def __init__(self, version=None, device="cuda", max_length=77):
        super().__init__()
        version = version or self.CLIP_VERSION
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=input_ids)
        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)
