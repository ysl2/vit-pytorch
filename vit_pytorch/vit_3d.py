import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import pysnooper
import copy

# helpers


def snoop(**kwargs):
    return pysnooper.snoop('/home/yusongli/Documents/vit-pytorch/debug.log', **kwargs)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.fn(self.norm(x), **kwargs)
        return x


class PreNorm2(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, x1, **kwargs):
        x, x1 = self.fn(self.norm(x), self.norm(x1), **kwargs)
        return x, x1


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = heads != 1 or dim_head != dim

        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x, x1):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        qkv1 = self.to_qkv(x1).chunk(3, dim=-1)

        def _fn(t):
            return rearrange(t, 'b n (h d) -> b h n d', h=self.heads)
        q, k, v = map(_fn, qkv)
        q1, k1, v1 = map(_fn, qkv1)

        temp = torch.matmul(q, k1.transpose(-1, -2)) * self.scale
        temp1 = torch.matmul(q1, k.transpose(-1, -2)) * self.scale

        temp = self.softmax(temp)
        temp1 = self.softmax(temp1)

        temp = self.dropout(temp)
        temp1 = self.dropout(temp1)

        temp = torch.matmul(temp, v1)
        temp1 = torch.matmul(temp1, v)

        temp = rearrange(temp, 'b h n d -> b n (h d)')
        temp1 = rearrange(temp1, 'b h n d -> b n (h d)')

        temp = self.to_out(temp)
        temp1 = self.to_out(temp1)

        return temp, temp1


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm2(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x, x1):
        for attn, ff, ff1 in self.layers:
            temp = attn(x, x1)
            x, x1 = temp[0] + x, temp[1] + x1
            x = ff(x) + x
            x1 = ff1(x1) + x1
        return x, x1


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        dim,
        depth,
        heads,
        mlp_dim,
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                'b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)',
                p1=patch_height,
                p2=patch_width,
                pf=frame_patch_size,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_out = Rearrange(
            'b (f h w) (ph pw pf c) -> b c (f pf) (h ph) (w pw)',
            ph=patch_height, pw=patch_width, pf=frame_patch_size,
            h=image_height // patch_height,
            w=image_width // patch_width,
        )

    def forward(self, x, x1):
        # NOTE: The patch embedding procedure need to share weights:
        x = self.to_patch_embedding(x)
        x1 = self.to_patch_embedding(x1)

        x += self.pos_embedding
        x1 += self.pos_embedding

        x = self.dropout(x)
        x1 = self.dropout(x1)

        x, x1 = self.transformer(x, x1)

        x = self.to_out(x)
        x1 = self.to_out(x1)
        return x, x1


if __name__ == '__main__':
    # NOTE: `p` is number of patches
    # b, c, f, h, w, pf, ph, pw = (4, 3, 32, 80, 96, 8, 8, 8)

    # NOTE: YUNet layers:
    # b, c, f, h, w, pf, ph, pw = (2, 32, 32, 80, 96, 8, 8, 8)
    # b, c, f, h, w, pf, ph, pw = (2, 64, 32, 40, 48, 8, 8, 8)
    # b, c, f, h, w, pf, ph, pw = (2, 128, 32, 20, 24, 8, 4, 8)
    b, c, f, h, w, pf, ph, pw = (2, 256, 16, 10, 12, 8, 2, 4)
    # b, c, f, h, w, pf, ph, pw = (2, 320, 8, 5, 6)
    video = torch.randn(b, c, f, h, w)  # (batch, channels, frames, height, width)

    # v = Transformer(dim=1440, depth=1, heads=8, dim_head=64, mlp_dim=2048, dropout=0.1)
    # with snoop(watch=('video.shape', 'preds.shape')):
    #     video = rearrange(video, 'b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1=10, p2=12, pf=4)
    #     preds = v(video)
    #     preds = rearrange(preds, 'b (f h w) (p1 p2 pf c) -> b c (f pf) (h p1) (w p2)', p1=10, p2=12, pf=4, f=8, h=8)

    v = ViT(
        image_size=(h, w),  # image size
        frames=f,  # number of frames
        image_patch_size=(ph, pw),  # image patch size
        frame_patch_size=pf,  # frame patch size
        dim=c * pf * ph * pw,
        depth=1,
        heads=8,
        channels=c,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
        dim_head=64
    )

    # with snoop(watch=('video.shape', 'preds.shape')):
    #     preds = v(video, video.clone())
    #     print(preds[0].shape, preds[1].shape)

    preds = v(video, video.clone())
    print(preds[0].shape, preds[1].shape)
