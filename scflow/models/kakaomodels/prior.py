# ------------------------------------------------------------------------------------
# Adapted from the repos below:
# (a) Guided-Diffusion (https://github.com/openai/guided-diffusion)
# (b) CLIP ViT (https://github.com/openai/CLIP/)
# ------------------------------------------------------------------------------------

import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period)
        * th.arange(start=0, end=half, dtype=th.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def convert_module_to_f16(param):
    """
    Convert primitive modules to float16.
    """
    if isinstance(param, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        param.weight.data = param.weight.data.half()
        if param.bias is not None:
            param.bias.data = param.bias.data.half()


class LayerNorm(nn.LayerNorm):
    """
    Implementation that supports fp16 inputs but fp32 gains/biases.
    """

    def forward(self, x: th.Tensor):
        return super().forward(x.float()).to(x.dtype)


class MultiheadAttention(nn.Module):
    def __init__(self, n_ctx, width, heads):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadAttention(heads, n_ctx)

    def forward(self, x, mask=None):
        x = self.c_qkv(x)
        x = self.attention(x, mask=mask)
        x = self.c_proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4)
        self.c_proj = nn.Linear(width * 4, width)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class QKVMultiheadAttention(nn.Module):
    def __init__(self, n_heads: int, n_ctx: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_ctx = n_ctx

    def forward(self, qkv, mask=None):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.n_heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.n_heads, -1)
        q, k, v = th.split(qkv, attn_ch, dim=-1)
        weight = th.einsum("bthc,bshc->bhts", q * scale, k * scale)
        wdtype = weight.dtype
        if mask is not None:
            weight = weight + mask[:, None, ...]
        weight = th.softmax(weight, dim=-1).type(wdtype)
        return th.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        n_ctx: int,
        width: int,
        heads: int,
    ):
        super().__init__()

        self.attn = MultiheadAttention(
            n_ctx,
            width,
            heads,
        )
        self.ln_1 = LayerNorm(width)
        self.mlp = MLP(width)
        self.ln_2 = LayerNorm(width)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln_1(x), mask=mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    n_ctx,
                    width,
                    heads,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x, mask=None):
        for block in self.resblocks:
            x = block(x, mask=mask)
        return x


class PriorTransformer(nn.Module):
    """
    A Causal Transformer that conditions on CLIP text embedding, text.

    :param text_ctx: number of text tokens to expect.
    :param xf_width: width of the transformer.
    :param xf_layers: depth of the transformer.
    :param xf_heads: heads in the transformer.
    :param xf_final_ln: use a LayerNorm after the output layer.
    :param clip_dim: dimension of clip feature.
    """

    def __init__(
        self,
        xf_width,
        xf_layers,
        xf_heads,
        xf_final_ln,
        clip_dim
    ):
        super().__init__()

        # self.text_ctx = text_ctx
        text_ctx = 0
        self.xf_width = xf_width
        self.xf_layers = xf_layers
        self.xf_heads = xf_heads
        self.clip_dim = clip_dim
        self.ext_len = 3

        self.time_embed = nn.Sequential(
            nn.Linear(xf_width, xf_width),
            nn.SiLU(),
            nn.Linear(xf_width, xf_width),
        )

        self.clip_img_proj = nn.Linear(self.clip_dim, xf_width)
        self.out_proj = nn.Linear(xf_width, self.clip_dim)
        
        self.transformer = Transformer(
            text_ctx + self.ext_len,
            xf_width,
            xf_layers,
            xf_heads,
        )
        if xf_final_ln:
            self.final_ln = LayerNorm(xf_width)
        else:
            self.final_ln = None

        self.positional_embedding = nn.Parameter(
            th.empty(1, text_ctx + self.ext_len, xf_width)
        )
        self.prd_emb = nn.Parameter(th.randn((1, 1, xf_width)))

        nn.init.normal_(self.prd_emb, std=0.01)
        nn.init.normal_(self.positional_embedding, std=0.01)

    def forward(
        self,
        x,  
        t
    ):
        # squeeze the channel dimension
        x = x.squeeze(1)
        bsz = x.shape[0]
        
        t_emb = self.time_embed(timestep_embedding(t, self.xf_width))   

        x = self.clip_img_proj(x)

        input_seq = [
            # text_enc,
            t_emb[:, None, :],
            x[:, None, :],
            self.prd_emb.to(x.dtype).expand(bsz, -1, -1),
        ]
        input = th.cat(input_seq, dim=1)
        input = input + self.positional_embedding.to(input.dtype)


        out = self.transformer(input) #, mask=mask)
        if self.final_ln is not None:
            out = self.final_ln(out)

        out = self.out_proj(out[:, -1])

        return out.unsqueeze(1)

if __name__ == "__main__":
    model = PriorTransformer(
        # text_ctx=77,
        xf_width=2048,
        xf_layers=12,
        xf_heads=32,
        xf_final_ln=True,
        clip_dim = 1536#1536
        
    )

    x = th.randn(3, 768*2)
    timesteps = th.rand(3, dtype=th.float32)
    
    out = model(x, timesteps)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")

