from timm.models.layers import Mlp,DropPath
from timm.models.vision_transformer import LayerScale
from model.layers.Attention import BaseAttention, CrossAttention
import torch.nn as nn

class BaseBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            use_local = False
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = BaseAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.use_local = use_local
        if use_local:
            self.local_attn = BaseAttention(dim, num_heads=4, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x,mask=None):
        res = x
        x_ = self.norm1(x)
        x = self.attn(x_)
        if self.use_local:
            x += self.local_attn(x_,mask)
        x = res + self.drop_path1(self.ls1(x))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class CrossBlock(BaseBlock):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            use_local=False
    ):
        super(CrossBlock, self).__init__(dim,num_heads,mlp_ratio,qkv_bias,drop,
                                         attn_drop,init_values,drop_path,act_layer,norm_layer,use_local)
        self.normq = norm_layer(dim)
        self.crossAttn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

    def forward_(self, q, x, mask=None):
        x = self.crossAttn(self.norm1(x),self.normq(q))
        x_ = self.attn(x)
        if self.use_local:
            x_ += self.local_attn(x,mask)
        x = x + self.drop_path1(self.ls1(x_))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x