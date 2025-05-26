# models/mae.py

import torch
import torch.nn as nn
from einops import rearrange
from refrakt_core.models.templates.base import BaseModel
from refrakt_core.registry.model_registry import register_model
from refrakt_core.utils.methods import get_2d_sincos_pos_embed, random_masking

@register_model("mae")
class MAE(BaseModel):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, encoder_depth=12, decoder_dim=512,
                 decoder_depth=8, num_heads=12, decoder_num_heads=16,
                 mask_ratio=0.75, **kwargs):
        super().__init__({})
        self.mask_ratio = mask_ratio

        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * in_chans

        # === Patch embedding ===
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed_enc = nn.Parameter(
            get_2d_sincos_pos_embed(embed_dim, int(self.num_patches ** 0.5), cls_token=False),
            requires_grad=False
        )

        # === Encoder ===
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=embed_dim * 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)

        # === Mask token for decoder ===
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        # === Decoder positional embedding ===
        self.decoder_pos_embed = nn.Parameter(
            get_2d_sincos_pos_embed(decoder_dim, int(self.num_patches ** 0.5), cls_token=False),
            requires_grad=False
        )

        # === Decoder ===
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim, bias=True)
        decoder_layer = nn.TransformerEncoderLayer(decoder_dim, decoder_num_heads, dim_feedforward=decoder_dim * 4, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)

        self.decoder_pred = nn.Linear(decoder_dim, self.patch_dim, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.xavier_uniform_(self.decoder_pred.weight)
        nn.init.constant_(self.decoder_pred.bias, 0)

    def patchify(self, imgs):
        """ (B, 3, H, W) -> (B, N, patch_dim) """
        p = self.patch_embed.kernel_size[0]
        x = rearrange(imgs, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        return x

    def unpatchify(self, x):
        """ (B, N, patch_dim) -> (B, 3, H, W) """
        p = self.patch_embed.kernel_size[0]
        h = w = int(x.shape[1] ** 0.5)
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=h, w=w, p1=p, p2=p, c=3)
        return x

    def forward(self, imgs):
        # === Patchify and embed ===
        x = self.patch_embed(imgs)  # [B, C, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]

        x = x + self.pos_embed_enc.unsqueeze(0)  # Positional encoding

        # === Random masking ===
        x_masked, mask, ids_restore, ids_keep = random_masking(x, self.mask_ratio)

        # === Encoder (only visible patches) ===
        encoded = self.encoder(x_masked)

        # === Decoder input: insert mask tokens ===
        dec_tokens = self.decoder_embed(encoded)
        B, N_vis, C = dec_tokens.shape
        mask_tokens = self.mask_token.expand(B, self.num_patches - N_vis, -1)

        x_full = torch.zeros(B, self.num_patches, C, device=imgs.device)
        x_full.scatter_(1, ids_restore.unsqueeze(-1).expand(-1, -1, C), torch.cat([dec_tokens, mask_tokens], dim=1))

        x_full = x_full + self.decoder_pos_embed.unsqueeze(0)
        decoded = self.decoder(x_full)

        # === Pixel prediction ===
        pred = self.decoder_pred(decoded)  # [B, N, patch_dim]

        return {
            "recon_patches": pred,
            "mask": mask,
            "original_patches": self.patchify(imgs)
        }
