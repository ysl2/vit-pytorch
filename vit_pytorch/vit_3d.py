import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import pysnooper
import pathlib

# helpers


def snoop(**kwargs):
    LOG = pathlib.Path('/home/yusongli/Documents/vit-pytorch/debug.log')
    LOG.parent.mkdir(parents=True, exist_ok=True)
    return pysnooper.snoop(LOG, **kwargs)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, *args):
        temp = self.fn(*[self.norm(arg) for arg in args])
        return temp[0] if len(temp) == 1 else temp


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
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, double_outputs=True):
        super().__init__()
        self.double_outputs = double_outputs
        inner_dim = dim_head * heads
        project_out = heads != 1 or dim_head != dim

        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    # @snoop(watch=(
    #     'x.shape',
    #     'x1.shape',
    #     'qkv.shape',
    #     'qkv1.shape',
    #     'q.shape',
    #     'k.shape',
    #     'v.shape',
    #     'q1.shape',
    #     'k1.shape',
    #     'v1.shape',
    #     'temp.shape',
    #     'temp1.shape',
    # ))
    def forward(self, x, x1):
        # NOTE:
        # x provides q
        # x1 provides k1, v1

        x = self.to_qkv(x).chunk(3, dim=-1)
        x1 = self.to_qkv(x1).chunk(3, dim=-1)

        def _fn(_x):
            return rearrange(_x, 'b n (h d) -> b h n d', h=self.heads)
        q, k, v = map(_fn, x)
        q1, k1, v1 = map(_fn, x1)

        def _add_zero_attn(_x):
            return torch.concat([_x, torch.zeros(_x.shape[0], _x.shape[1], 1, _x.shape[3])], dim=2)

        def _cross_attn(_q, _k1, _v1):
            _q = torch.matmul(_q, _k1.transpose(-1, -2)) * self.scale
            _q = self.softmax(_q)
            _q = self.dropout(_q)
            _q = torch.matmul(_q, v1)
            _q = rearrange(_q, 'b h n d -> b n (h d)')
            _q = self.to_out(_q)
            return _q

        x = _cross_attn(
            q,
            _add_zero_attn(k1),
            _add_zero_attn(v1)
        )
        if self.double_outputs:
            x1 = _cross_attn(
                q1,
                _add_zero_attn(k),
                _add_zero_attn(v)
            )

        result = [x, x1] if self.double_outputs else x
        return result


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0, double_outputs=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        layer = [
            PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, double_outputs=double_outputs)),
            PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
        ]
        if double_outputs:
            layer.append(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
        self.double_outputs = double_outputs
        for _ in range(depth):
            self.layers.append(nn.ModuleList(layer))

    def forward(self, *args):
        args = list(args)
        for attn, *ff in self.layers:
            temp = attn(*args)
            if not isinstance(temp, (list, tuple)):
                temp = [temp]
            for i in range(len(temp)):
                args[i] = temp[i] + args[i]
                args[i] = ff[i](args[i]) + args[i]
        return args[0] if len(temp) == 1 else args


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
        emb_dropout=0.0,
        double_outputs=True
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
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, double_outputs=double_outputs)
        self.to_out = Rearrange(
            'b (f h w) (ph pw pf c) -> b c (f pf) (h ph) (w pw)',
            ph=patch_height, pw=patch_width, pf=frame_patch_size,
            h=image_height // patch_height,
            w=image_width // patch_width,
        )

    # @snoop(watch=('x.shape', 'x1.shape', 'len(temp)', 'temp.shape', 'type(temp)'))
    def forward(self, x, x1):
        # NOTE: The patch embedding procedure need to share weights:
        x = self.to_patch_embedding(x)
        x1 = self.to_patch_embedding(x1)

        x += self.pos_embedding
        x1 += self.pos_embedding

        x = self.dropout(x)
        x1 = self.dropout(x1)

        temp = self.transformer(x, x1)
        if not isinstance(temp, (tuple, list)):
            return self.to_out(temp)
        return [self.to_out(t) for t in temp]


if __name__ == '__main__':
    # NOTE:
    # 1. `np` is number of patches
    # 2. `p` is size of patches
    # b, c, f, h, w, npf, nph, npw = (1, 1, 32, 80, 96, 8, 8, 8)

    # NOTE: YUNet layers:
    # b, c, f, h, w, npf, nph, npw = (2, 32, 32, 80, 96, 8, 8, 8)
    # b, c, f, h, w, npf, nph, npw = (2, 64, 32, 40, 48, 8, 8, 8)
    b, c, f, h, w, npf, nph, npw = (2, 128, 32, 20, 24, 8, 4, 8)
    # b, c, f, h, w, npf, nph, npw = (2, 256, 16, 10, 12, 8, 2, 4)
    # b, c, f, h, w = (2, 320, 8, 5, 6)
    video = torch.randn(b, c, f, h, w)  # (batch, channels, frames, height, width)

    pf = f // npf
    ph = h // nph
    pw = w // npw

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
        dim_head=64,
        double_outputs=True
    )

    preds = v(video, video.clone())
    print(type(preds))
    import ipdb; ipdb.set_trace()  # HACK: Songli.Yu: ""

    # v = Transformer(dim=1440, depth=1, heads=8, dim_head=64, mlp_dim=2048, dropout=0.1)
    # with snoop(watch=('video.shape', 'preds.shape')):
    #     video = rearrange(video, 'b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1=10, p2=12, pf=4)
    #     preds = v(video)
    #     preds = rearrange(preds, 'b (f h w) (p1 p2 pf c) -> b c (f pf) (h p1) (w p2)', p1=10, p2=12, pf=4, f=8, h=8)
