"""
minVLA: Minimal Vision-Language-Action Model for Computer Use
=============================================================

A minimal, educational implementation of a VLA model for computer control.
Inspired by Andrej Karpathy's nanoGPT/micrograd philosophy: simple, readable, hackable.

The model takes:
    - Screenshot (224x224 RGB image)
    - Text instruction ("click the search button")

And outputs:
    - Action: (x, y, action_type, key_text)

Architecture:
    Screenshot -> ViT Encoder -> Visual Tokens
    Instruction -> Embedding -> Text Tokens
    [Visual + Text] -> Transformer -> Action Head -> Action

Usage:
    model = MiniVLA()
    action = model(screenshot, "click on the file menu")

Reference:
    - RT-2: https://arxiv.org/abs/2307.15818
    - Alpamayo: NVIDIA's VLA for autonomous driving
    - nanoGPT: https://github.com/karpathy/nanoGPT

Author: Karpathy-style refactor
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class VLAConfig:
    """Configuration for MiniVLA model."""
    # Vision
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3

    # Model dimensions
    n_embd: int = 512
    n_head: int = 8
    n_layer: int = 6
    dropout: float = 0.1

    # Vocabulary (for text)
    vocab_size: int = 50257  # GPT-2 vocab size
    max_seq_len: int = 128

    # Action space
    screen_size: Tuple[int, int] = (1920, 1080)
    n_action_types: int = 7  # click, double_click, right_click, drag, scroll, type, hotkey

    # Training
    learning_rate: float = 3e-4
    weight_decay: float = 0.1


# -----------------------------------------------------------------------------
# Vision Encoder (Simple ViT)
# -----------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Split image into patches and embed them."""

    def __init__(self, config: VLAConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.n_patches = (config.image_size // config.patch_size) ** 2
        patch_dim = config.in_channels * config.patch_size * config.patch_size

        self.proj = nn.Linear(patch_dim, config.n_embd)
        self.norm = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        p = self.patch_size

        # Split into patches: (B, C, H, W) -> (B, n_patches, patch_dim)
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, -1, C * p * p)

        # Project to embedding dim
        x = self.proj(x)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, config: VLAConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape

        # QKV projection
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_head, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Output
        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    """Simple MLP with GELU activation."""

    def __init__(self, config: VLAConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block: attention + MLP with residuals."""

    def __init__(self, config: VLAConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = Attention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


# -----------------------------------------------------------------------------
# Action Head
# -----------------------------------------------------------------------------

class ActionHead(nn.Module):
    """Predicts actions from transformer output."""

    def __init__(self, config: VLAConfig):
        super().__init__()
        self.config = config

        # Coordinate prediction (normalized 0-1)
        self.coord_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, 4),  # x, y, end_x, end_y (for drag)
            nn.Sigmoid()
        )

        # Action type classification
        self.type_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, config.n_action_types)
        )

        # Confidence score
        self.conf_head = nn.Sequential(
            nn.Linear(config.n_embd, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: (B, n_embd) - pooled transformer output

        Returns:
            coords: (B, 4) - normalized coordinates
            action_type: (B, n_action_types) - action logits
            confidence: (B, 1) - confidence score
        """
        coords = self.coord_head(x)
        action_type = self.type_head(x)
        confidence = self.conf_head(x)

        return coords, action_type, confidence


# -----------------------------------------------------------------------------
# Main Model
# -----------------------------------------------------------------------------

class MiniVLA(nn.Module):
    """
    Minimal Vision-Language-Action model.

    Takes a screenshot and text instruction, outputs an action.
    """

    def __init__(self, config: Optional[VLAConfig] = None):
        super().__init__()
        self.config = config or VLAConfig()

        # Vision encoder
        self.patch_embed = PatchEmbed(self.config)
        n_patches = self.patch_embed.n_patches
        self.vis_pos_emb = nn.Parameter(torch.zeros(1, n_patches, self.config.n_embd))

        # Text encoder
        self.tok_emb = nn.Embedding(self.config.vocab_size, self.config.n_embd)
        self.txt_pos_emb = nn.Parameter(torch.zeros(1, self.config.max_seq_len, self.config.n_embd))

        # Modality tokens (learnable)
        self.vis_token = nn.Parameter(torch.zeros(1, 1, self.config.n_embd))
        self.txt_token = nn.Parameter(torch.zeros(1, 1, self.config.n_embd))
        self.act_token = nn.Parameter(torch.zeros(1, 1, self.config.n_embd))

        # Transformer
        self.blocks = nn.ModuleList([Block(self.config) for _ in range(self.config.n_layer)])
        self.ln_f = nn.LayerNorm(self.config.n_embd)

        # Action head
        self.action_head = ActionHead(self.config)

        # Initialize weights
        self.apply(self._init_weights)

        # Report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"MiniVLA initialized with {n_params/1e6:.2f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def encode_image(self, img):
        """Encode image to visual tokens."""
        # img: (B, C, H, W)
        x = self.patch_embed(img)  # (B, n_patches, n_embd)
        x = x + self.vis_pos_emb

        # Prepend visual modality token
        vis_tok = self.vis_token.expand(x.size(0), -1, -1)
        x = torch.cat([vis_tok, x], dim=1)
        return x

    def encode_text(self, tokens):
        """Encode text tokens."""
        # tokens: (B, seq_len)
        B, T = tokens.shape
        x = self.tok_emb(tokens)  # (B, T, n_embd)
        x = x + self.txt_pos_emb[:, :T, :]

        # Prepend text modality token
        txt_tok = self.txt_token.expand(B, -1, -1)
        x = torch.cat([txt_tok, x], dim=1)
        return x

    def forward(self, img, tokens):
        """
        Forward pass.

        Args:
            img: (B, C, H, W) - screenshot
            tokens: (B, seq_len) - tokenized instruction

        Returns:
            action: dict with coords, type, confidence
        """
        # Encode modalities
        vis_emb = self.encode_image(img)  # (B, 1+n_patches, n_embd)
        txt_emb = self.encode_text(tokens)  # (B, 1+seq_len, n_embd)

        # Add action query token
        B = img.size(0)
        act_tok = self.act_token.expand(B, -1, -1)

        # Concatenate: [VIS_TOK, vis_patches, TXT_TOK, txt_tokens, ACT_TOK]
        x = torch.cat([vis_emb, txt_emb, act_tok], dim=1)

        # Transformer
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        # Get action token output (last token)
        act_emb = x[:, -1, :]  # (B, n_embd)

        # Predict action
        coords, action_type, confidence = self.action_head(act_emb)

        return {
            'coords': coords,           # (B, 4) - x, y, end_x, end_y normalized
            'action_type': action_type, # (B, 7) - logits
            'confidence': confidence    # (B, 1)
        }

    def predict(self, img, instruction, tokenizer):
        """
        Convenience method for inference.

        Args:
            img: PIL Image or numpy array
            instruction: str
            tokenizer: tokenizer with encode method

        Returns:
            action: dict with screen coordinates
        """
        import numpy as np
        from PIL import Image

        self.eval()
        device = next(self.parameters()).device

        # Preprocess image
        if isinstance(img, Image.Image):
            img = img.resize((self.config.image_size, self.config.image_size))
            img = np.array(img) / 255.0

        img_tensor = torch.from_numpy(img).float()
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # Tokenize text
        tokens = tokenizer.encode(instruction)
        tokens = tokens[:self.config.max_seq_len]
        tokens = torch.tensor(tokens).unsqueeze(0).to(device)

        # Forward
        with torch.no_grad():
            out = self.forward(img_tensor, tokens)

        # Decode action
        coords = out['coords'][0].cpu().numpy()
        action_idx = out['action_type'][0].argmax().item()
        conf = out['confidence'][0].item()

        action_types = ['click', 'double_click', 'right_click', 'drag', 'scroll', 'type', 'hotkey']

        return {
            'x': int(coords[0] * self.config.screen_size[0]),
            'y': int(coords[1] * self.config.screen_size[1]),
            'end_x': int(coords[2] * self.config.screen_size[0]),
            'end_y': int(coords[3] * self.config.screen_size[1]),
            'action': action_types[action_idx],
            'confidence': conf
        }


# -----------------------------------------------------------------------------
# Loss Functions
# -----------------------------------------------------------------------------

def compute_loss(pred, target):
    """
    Compute training loss.

    Args:
        pred: model output dict
        target: dict with 'coords', 'action_type'

    Returns:
        loss: scalar tensor
        metrics: dict with individual losses
    """
    # Coordinate loss (smooth L1)
    coord_loss = F.smooth_l1_loss(pred['coords'], target['coords'])

    # Action type loss (cross entropy)
    type_loss = F.cross_entropy(pred['action_type'], target['action_type'])

    # Combined loss
    loss = coord_loss + type_loss

    return loss, {
        'coord_loss': coord_loss.item(),
        'type_loss': type_loss.item(),
        'total_loss': loss.item()
    }


# -----------------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    print("Testing MiniVLA...")

    # Create model
    config = VLAConfig()
    model = MiniVLA(config)

    # Dummy inputs
    B = 2
    img = torch.randn(B, 3, 224, 224)
    tokens = torch.randint(0, 50257, (B, 32))

    # Forward pass
    out = model(img, tokens)

    print(f"\nInput shapes:")
    print(f"  Image: {img.shape}")
    print(f"  Tokens: {tokens.shape}")
    print(f"\nOutput shapes:")
    print(f"  Coords: {out['coords'].shape}")
    print(f"  Action type: {out['action_type'].shape}")
    print(f"  Confidence: {out['confidence'].shape}")

    # Test loss
    target = {
        'coords': torch.rand(B, 4),
        'action_type': torch.randint(0, 7, (B,))
    }
    loss, metrics = compute_loss(out, target)
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")

    print("\nTest passed!")
