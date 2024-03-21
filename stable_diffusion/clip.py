import glob
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import mlx.core as mx
import mlx.nn as nn

@dataclass
class CLIPTextModelConfig:
    num_layers: int = 23
    model_dims: int = 1024
    num_heads: int = 16
    max_length: int = 77
    vocab_size: int = 49408
    projection_dim: Optional[int] = None
    hidden_act: str = "quick_gelu"

_ACTIVATIONS = {"quick_gelu": nn.gelu_fast_approx, "gelu": nn.gelu}

@dataclass
class CLIPOutput:
    # last_hidden_state indexed at the EOS token, projected if projection layer present
    pooled_output: Optional[mx.array] = None

    # final layernorm sequence output
    last_hidden_state: Optional[mx.array] = None

    # hidden states corresponding to the outputs of the transformer layers
    hidden_states: Optional[List[mx.array]] = None


class CLIPEncoderLayer(nn.Module):
    """The transformer encoder layer from CLIP."""

    def __init__(self, model_dims: int, num_heads: int, activation: str):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(model_dims)
        self.layer_norm2 = nn.LayerNorm(model_dims)

        self.attention = nn.MultiHeadAttention(model_dims, num_heads)
        # Add biases to the attention projections to match CLIP
        self.attention.query_proj.bias = mx.zeros(model_dims)
        self.attention.key_proj.bias = mx.zeros(model_dims)
        self.attention.value_proj.bias = mx.zeros(model_dims)
        self.attention.out_proj.bias = mx.zeros(model_dims)

        self.linear1 = nn.Linear(model_dims, 4 * model_dims)
        self.linear2 = nn.Linear(4 * model_dims, model_dims)

        self.act = _ACTIVATIONS[activation]

    def __call__(self, x, attn_mask=None):
        y = self.layer_norm1(x)
        y = self.attention(y, y, y, attn_mask)
        x = y + x

        y = self.layer_norm2(x)
        y = self.linear1(y)
        y = self.act(y)
        y = self.linear2(y)
        x = y + x

        return x


class CLIPTextModel(nn.Module):
    """Implements the text encoder transformer from CLIP."""

    def __init__(self, config: CLIPTextModelConfig):
        super().__init__()

        self.token_embedding = nn.Embedding(config.vocab_size, config.model_dims)
        self.position_embedding = nn.Embedding(config.max_length, config.model_dims)
        self.layers = [
            CLIPEncoderLayer(config.model_dims, config.num_heads, config.hidden_act)
            for i in range(config.num_layers)
        ]
        self.final_layer_norm = nn.LayerNorm(config.model_dims)

        if config.projection_dim is not None:
            self.text_projection = nn.Linear(
                config.model_dims, config.projection_dim, bias=False
            )

    def _get_mask(self, N, dtype):
        indices = mx.arange(N)
        mask = indices[:, None] < indices[None]
        mask = mask.astype(dtype) * (-6e4 if dtype == mx.float16 else -1e9)
        return mask

    def __call__(self, x):
        # Extract some shapes
        B, N = x.shape
        eos_tokens = x.argmax(-1)

        # Compute the embeddings
        x = self.token_embedding(x)
        x = x + self.position_embedding.weight[:N]

        # Compute the features from the transformer
        mask = self._get_mask(N, x.dtype)
        hidden_states = []
        for l in self.layers:
            x = l(x, mask)
            hidden_states.append(x)

        # Apply the final layernorm and return
        x = self.final_layer_norm(x)
        last_hidden_state = x

        # Select the EOS token
        pooled_output = x[mx.arange(len(x)), eos_tokens]
        if "text_projection" in self:
            pooled_output = self.text_projection(pooled_output)

        return CLIPOutput(
            pooled_output=pooled_output,
            last_hidden_state=last_hidden_state,
            hidden_states=hidden_states,
        )

    @staticmethod
    def from_pretrained(path: str):
        path = Path(path)

        with open(path / "config.json", "r") as fid:
            config = json.load(fid)

        text_config = CLIPTextModelConfig(
            num_hidden_layers=config["num_hidden_layers"],
            hidden_size=config["hidden_size"],
            intermediate_size=config["intermediate_size"],
            num_attention_heads=config["num_attention_heads"],
            max_position_embeddings=config["max_position_embeddings"],
            vocab_size=config["vocab_size"],
            layer_norm_eps=config["layer_norm_eps"],
            projection_dim=config["projection_dim"]
        )

        model = CLIPTextModel(text_config)
        weight_files = glob.glob(str(path / "*.safetensors"))
        if not weight_files:
            logging.error(f"No safetensors found in {path}")
            raise FileNotFoundError(f"No safetensors found in {path}")

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))

        weights = model.sanitize(weights)
        model.load_weights(list(weights.items()))
        return model

    @staticmethod
    def sanitize(weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                # rm unused token position IDs
                continue
            elif "patch_embedding.weight" in k:
                # transpose the embedding tensor from [out_channels, in_channels, kH, KW] (MLX) -> [out_channels, kH, KW, in_channels] (pytorch)
                sanitized_weights[k] = v.transpose(0, 2, 3, 1) # [out_channels, kH, KW, in_channels]
            else:
                sanitized_weights[k] = v

        return sanitized_weights
