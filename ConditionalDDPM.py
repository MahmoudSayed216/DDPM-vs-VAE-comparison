import torch
import torch.nn as nn
import math


class Downsample(nn.Module):
    """A DDPM ResNet block that can optionally perform Self-Attention and inject time/class conditioning."""

    def __init__(self, in_channels, out_channels, num_groups, emb_dim, use_attn=False):
        super().__init__()
        self.use_attn = use_attn

        # First convolution block
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

        # Time/Class embedding projection layer
        self.time_emb_proj = nn.Linear(emb_dim, out_channels)

        # Second convolution block
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)

        # Shortcut connection if channels mismatch
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Attention block (if enabled for low-resolution levels)
        if use_attn:
            self.attn_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
            self.attention = nn.MultiheadAttention(embed_dim=out_channels, num_heads=4, batch_first=True)

    def forward(self, x, emb):
        # 1. Main residual processing path
        h = self.conv1(self.act1(self.norm1(x)))

        # 2. Project and inject the time/class conditioning vector
        h = h + self.time_emb_proj(emb)[:, :, None, None]

        # 3. Second convolution stage
        h = self.conv2(self.act2(self.norm2(h)))

        # 4. Apply shortcut identity or channel projection
        x = h + self.shortcut(x)

        # 5. Optional Self-Attention block
        if self.use_attn:
            h = self.attn_norm(x)
            B, C, H, W = h.shape
            h = h.view(B, C, H * W).permute(0, 2, 1)
            attn_out, _ = self.attention(h, h, h)
            attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)
            x = x + attn_out

        return x


class Upsample(nn.Module):
    """An Upsampling block that uses nearest-neighbor interpolation followed by a convolution."""

    def __init__(self, in_channels, out_channels, num_groups, emb_dim, use_attn=False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.block = Downsample(in_channels, out_channels, num_groups, emb_dim, use_attn=use_attn)

    def forward(self, x, emb, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.block(x, emb)


class MLPSinudoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, theta=10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.lntheta = math.log(theta)

    def forward(self, timestep):
        half_dim = self.embedding_dim // 2

        if len(timestep.shape) > 1:
            timestep = timestep.squeeze(-1)

        exponent = torch.arange(half_dim, device=timestep.device) * self.lntheta / (half_dim - 1)
        vec = torch.exp(-exponent)

        scaled_time = timestep[:, None] * vec[None, :]

        sin = torch.sin(scaled_time)
        cos = torch.cos(scaled_time)

        return torch.cat([sin, cos], dim=-1)


class ConditionalDDPM(nn.Module):
    def __init__(self, num_classes, embedding_dim, num_groups, channels_per_level, theta=10000):
        super().__init__()

        self.timestep_embedder = MLPSinudoidalPositionalEmbedding(embedding_dim, theta)
        self.class_embedder = nn.Embedding(num_classes, embedding_dim)

        self.emb_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        c1, c2, c3, c4 = channels_per_level

        self.stem_conv = nn.Conv2d(in_channels=3, out_channels=c1, kernel_size=3, padding=1)

        # Encoder (Downsampling Path)
        self.down1a = Downsample(c1, c1, num_groups, embedding_dim, use_attn=False)
        self.down1b = Downsample(c1, c1, num_groups, embedding_dim, use_attn=False)
        self.pool1 = nn.Conv2d(in_channels=c1, out_channels=c1, kernel_size=2, stride=2)

        self.down2a = Downsample(c1, c2, num_groups, embedding_dim, use_attn=True)
        self.down2b = Downsample(c2, c2, num_groups, embedding_dim, use_attn=True)
        self.pool2 = nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=2, stride=2)

        self.down3a = Downsample(c2, c3, num_groups, embedding_dim, use_attn=True)
        self.down3b = Downsample(c3, c3, num_groups, embedding_dim, use_attn=True)
        self.pool3 = nn.Conv2d(in_channels=c3, out_channels=c3, kernel_size=2, stride=2)

        # Bottleneck Middle Layer
        self.mid1 = Downsample(c3, c4, num_groups, embedding_dim, use_attn=True)
        self.mid2 = Downsample(c4, c4, num_groups, embedding_dim, use_attn=True)

        # Decoder (Upsampling Path + Accounting for concatenated Skip Connections)
        self.up3 = Upsample(c4 + c3, c3, num_groups, embedding_dim, use_attn=True)
        self.up2 = Upsample(c3 + c2, c2, num_groups, embedding_dim, use_attn=True)
        self.up1 = Upsample(c2 + c1, c1, num_groups, embedding_dim, use_attn=False)

        # Output Projection Layer
        self.out_layer = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=c1),
            nn.SiLU(),
            nn.Conv2d(in_channels=c1, out_channels=3, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor, class_id: torch.Tensor, timestep: torch.Tensor):
        # 1. Process Conditionings
        cls_emb = self.class_embedder(class_id).squeeze(1) if len(class_id.shape) > 1 else self.class_embedder(class_id)
        ts_emb = self.timestep_embedder(timestep)
        cond_vector = self.emb_mlp(cls_emb + ts_emb)

        # 2. Encoder Step (Save skip connections *before* pooling down spatial resolution)
        x1_skip = self.stem_conv(x)
        x1_skip = self.down1a(x1_skip, cond_vector)
        x1_skip = self.down1b(x1_skip, cond_vector)

        x2_skip = self.pool1(x1_skip)
        x2_skip = self.down2a(x2_skip, cond_vector)
        x2_skip = self.down2b(x2_skip, cond_vector)

        x3_skip = self.pool2(x2_skip)
        x3_skip = self.down3a(x3_skip, cond_vector)
        x3_skip = self.down3b(x3_skip, cond_vector)

        # 3. Bottleneck Step
        out = self.pool3(x3_skip)
        out = self.mid1(out, cond_vector)
        out = self.mid2(out, cond_vector)

        # 4. Decoder Step with concatenated Skip Connections
        out = self.up3(out, cond_vector, skip=x3_skip)
        out = self.up2(out, cond_vector, skip=x2_skip)
        out = self.up1(out, cond_vector, skip=x1_skip)

        # 5. Map back to 3 RGB channels
        return self.out_layer(out)
