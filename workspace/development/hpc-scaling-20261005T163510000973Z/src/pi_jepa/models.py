"""Visual-only encoder, action-conditioned causal predictor, and physical readout.

AdaLN-zero follows the MIT-licensed LeWorldModel reference; see THIRD_PARTY.md.
"""

import torch
from torch import nn
from torch.nn import functional as F


MODEL_SIZES = ("small", "medium", "large")


def model_spec(size="small"):
    """The three explicit capacity choices; latent/readout interfaces stay fixed."""
    choices = {
        "small": ("resnet18", 192, 256, 128, 2, 192, 3, 3),
        "medium": ("resnet34", 256, 384, 256, 3, 256, 4, 4),
        "large": ("resnet50", 384, 512, 512, 4, 384, 6, 6),
    }
    if size not in choices:
        raise ValueError(f"model.size must be one of {MODEL_SIZES}, got {size!r}")
    keys = ("encoder_backbone", "encoder_projection", "projector_hidden", "passive_hidden",
            "passive_hidden_layers", "controlled_width", "controlled_layers", "controlled_heads")
    return {**dict(zip(keys, choices[size])), "latent_dim": 32, "readout_hidden": 64}


def model_size(config):
    size = config.get("model", {}).get("size", "small")
    model_spec(size)
    return size


def model_kwargs(config):
    """Constructor arguments; the default also permits tiny zero-argument test nets."""
    size = model_size(config)
    return {} if size == "small" else {"size": size}


def encoder_architecture(size="small"):
    spec = model_spec(size)
    return (f"{spec['encoder_backbone']}_gn32_project{spec['encoder_projection']}"
            f"_ln{spec['projector_hidden']}_24channel_32latent")


class Encoder(nn.Module):
    """Eight chronological RGB frames stacked as [N,24,96,96], in [-1,1].

    GroupNorm and the hidden projector LayerNorm normalize each clip separately.
    There are no batch statistics or moving averages, and the final 32-dimensional
    linear output is unnormalized for the JEPA/SIGReg objective.
    """

    def __init__(self, size="small"):
        super().__init__()
        from torchvision.models import resnet18, resnet34, resnet50

        spec = model_spec(size)
        self.size = size
        backbone = {"resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50}[spec["encoder_backbone"]]
        self.backbone = backbone(weights=None, norm_layer=lambda channels: nn.GroupNorm(32, channels))
        self.backbone.conv1 = nn.Conv2d(24, 64, 3, stride=2, padding=1, bias=False)
        nn.init.kaiming_normal_(self.backbone.conv1.weight, mode="fan_out", nonlinearity="relu")
        self.backbone.maxpool = nn.Identity()
        # Preserve spatial information needed to estimate cart position.
        self.backbone.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features * 3 * 3, spec["encoder_projection"])
        self.projector = nn.Sequential(
            nn.Linear(spec["encoder_projection"], spec["projector_hidden"]),
            nn.LayerNorm(spec["projector_hidden"]), nn.GELU(), nn.Linear(spec["projector_hidden"], 32)
        )

    def forward(self, clips):
        if clips.ndim != 4 or clips.shape[1:] != (24, 96, 96):
            raise ValueError("Encoder expects [N,24,96,96] chronological RGB clips")
        return self.projector(self.backbone(clips))


def encode_temporal(encoder, clips):
    """Encode [B,T,24,H,W] histories one temporal offset at a time.

    Each encoder call receives B independent episodes at the same offset. Future
    clips never share an encoder sample axis with earlier clips. The graph stays
    attached at every offset so both endpoints of the JEPA loss receive gradients.
    """
    if clips.ndim != 5 or clips.shape[0] < 1 or clips.shape[1] < 1 or clips.shape[2] != 24:
        raise ValueError("Temporal encoder expects nonempty clips [B,T,24,H,W]")
    return torch.stack([encoder(clips[:, offset]) for offset in range(clips.shape[1])], dim=1)


class ConditionalBlock(nn.Module):
    """Token-wise AdaLN-zero with strictly causal attention."""

    def __init__(self, width=192, heads=3):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0.1, batch_first=True)
        self.attention_dropout = nn.Dropout(0.1)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(4 * width, width), nn.Dropout(0.1),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x, conditioning, causal_mask):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.modulation(
            conditioning
        ).chunk(6, dim=-1)
        attention_input = self.norm1(x) * (1 + scale_a) + shift_a
        attended, _ = self.attention(
            attention_input, attention_input, attention_input,
            attn_mask=causal_mask, need_weights=False,
        )
        x = x + gate_a * self.attention_dropout(attended)
        return x + gate_f * self.mlp(self.norm2(x) * (1 + scale_f) + shift_f)


class Predictor(nn.Module):
    """Predict every token's next code within an explicit window of at most three.

    Inputs: z [B,L,32], forces [B,L,10], theta [B,2]. Each token receives its
    own ten *subsequent* recorded-interval forces. Positions always restart at zero per call.
    """

    def __init__(self, size="small"):
        super().__init__()
        spec = model_spec(size)
        self.size = size
        width = spec["controlled_width"]
        self.input_projection = nn.Linear(32, width)
        self.position = nn.Parameter(torch.randn(1, 3, width) * 0.02)
        self.conditioning = nn.Sequential(nn.Linear(12, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([ConditionalBlock(width, spec["controlled_heads"])
                                     for _ in range(spec["controlled_layers"])])
        self.final_norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, 32)

    def forward(self, z, forces, theta):
        if z.ndim != 3 or z.shape[-1] != 32 or not 1 <= z.shape[1] <= 3:
            raise ValueError("Predictor expects z [B,L,32] with 1 <= L <= 3")
        batch, length, _ = z.shape
        if forces.shape != (batch, length, 10) or theta.shape != (batch, 2):
            raise ValueError("Predictor expects forces [B,L,10] and theta [B,2]")
        normalized_theta = (theta - theta.new_tensor([1.0, 0.275])) / theta.new_tensor([0.3, 0.225])
        conditioning = self.conditioning(torch.cat([
            forces / 5.0, normalized_theta[:, None].expand(-1, length, -1)
        ], dim=-1))
        x = self.input_projection(z) + self.position[:, :length]
        # True marks forbidden future positions for MultiheadAttention.
        mask = torch.ones(length, length, device=z.device, dtype=torch.bool).triu(1)
        for block in self.blocks:
            x = block(x, conditioning, mask)
        # Projection after LayerNorm leaves the output latent unnormalized.
        return self.output_projection(self.final_norm(x))


class PassivePredictor(nn.Module):
    """Action-free autonomous dynamics: only the current visual latent is input."""

    action_free = True

    def __init__(self, size="small"):
        super().__init__()
        spec = model_spec(size)
        self.size = size
        hidden = spec["passive_hidden"]
        layers = [nn.Linear(32, hidden), nn.GELU()]
        for _ in range(spec["passive_hidden_layers"] - 1):
            layers.extend((nn.Linear(hidden, hidden), nn.GELU()))
        self.net = nn.Sequential(*layers, nn.Linear(hidden, 32))

    def forward(self, z):
        if z.shape[-1] != 32:
            raise ValueError("PassivePredictor expects last dimension 32")
        return self.net(z)


class PhysicalReadout(nn.Module):
    """Read [p, v, normalized sin(q), normalized cos(q), w] from latent codes."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 5))

    def forward(self, z):
        raw = self.net(z)
        angular_pair = F.normalize(raw[..., 2:4], dim=-1, eps=1e-6)
        return torch.cat([raw[..., :2], angular_pair, raw[..., 4:5]], dim=-1)


def to_state(decoded):
    """Convert readout coordinates to [p,v,q,w], respecting angular periodicity."""
    q = torch.atan2(decoded[..., 2], decoded[..., 3])
    return torch.stack([decoded[..., 0], decoded[..., 1], q, decoded[..., 4]], dim=-1)


def scale_readout(decoded):
    """Fixed units used by the physical residual: [p/2,v/2,sin(q),cos(q),w/5]."""
    return decoded / decoded.new_tensor([2.0, 2.0, 1.0, 1.0, 5.0])
