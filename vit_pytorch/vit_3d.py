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


# @snoop(watch='x.shape')
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.fn(self.norm(x), **kwargs)
        return x


# @snoop(watch='x.shape')
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.net(x)
        return x


# @snoop(watch=('x.shape', 'qkv.shape', 'q.shape', 'k.shape', 'v.shape', 'dots.shape', 'attn.shape' 'out.shape'))
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.attend1 = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_qkv1 = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()
        self.to_out1 = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x, x1):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        qkv1 = self.to_qkv1(x1).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        q1, k1, v1 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv1)

        # TODO: 2. Try to use SwinMM's method to implement the transformer structure.
        dots = torch.matmul(q, k1.transpose(-1, -2)) * self.scale
        dots1 = torch.matmul(q1, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        attn1 = self.attend1(dots1)
        attn1 = self.dropout1(dots1)

        out = torch.matmul(attn, v1)
        out1 = torch.matmul(attn1, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out1 = rearrange(out1, 'b h n d -> b n (h d)')

        out = self.to_out(out)
        out1 = self.to_out1(out1)
        return out, out1


# @snoop(watch='x.shape')
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
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


# @snoop(watch=('video.shape', 'x.shape', 'cls_tokens.shape'))
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
        pool='cls',
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

        self.to_in = Rearrange(
            'b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)',
            p1=patch_height,
            p2=patch_width,
            pf=frame_patch_size,
        ),
        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.to_patch_embedding1 = copy.deepcopy(self.to_patch_embedding)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_out = Rearrange(
            'b (f h w) (ph pw pf c) -> b c (f pf) (h ph) (w pw)',
            ph=patch_height, pw=patch_width, pf=frame_patch_size,
            h=image_height // patch_height,
            w=image_width // patch_width,
        )

    @snoop(watch=('video.shape', 'x.shape', 'cls_tokens.shape'))
    def forward(self, x, x1):
        # NOTE: The patch embedding procedure need to share weights:
        x = self.to_in(x)
        x1 = self.to_in(x1)

        x = self.to_patch_embedding(x)
        x1 = self.to_patch_embedding1(x1)

        x += self.pos_embedding
        x1 += self.pos_embedding

        x = self.dropout(x)
        x1 = self.dropout(x1)

        x, x1 = self.transformer(x, x1)

        x = self.to_out(x)
        x1 = self.to_out(x1)
        return x, x1


if __name__ == '__main__':
    b = 4
    c = 3
    f = 32
    h = 80
    w = 96
    video = torch.randn(b, c, f, h, w)  # (batch, channels, frames, height, width)

    ph = 10
    pw = 12
    pf = 4

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
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
    )
    with snoop(watch=('video.shape', 'preds.shape')):
        preds = v(video, video.clone())
        print(preds[0].shape, preds[1].shape)
        # preds = rearrange(preds, 'b (f h w) (ph pw pf c) -> b c (f pf) (h ph) (w pw)', ph=ph, pw=pw, pf=pf, h=h // ph, w=w // pw)
