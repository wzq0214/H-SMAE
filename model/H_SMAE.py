import torch.nn as nn
import torch
from model.layers.PositionEmbedding import PositionalEncoding
from model.layers.Block import BaseBlock, CrossBlock
from timm.models.layers import Mlp

class H_SMAE(nn.Module):
    def __init__(self,
                 embed_dim=1024, depth=2, num_heads=8,
                 decoder_embed_dim=512, decoder_depth=1, decoder_num_heads=8,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()

        # --------------------------------------------------------------------------
        self.pos_embed = PositionalEncoding(embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.blocks = nn.ModuleList([
            BaseBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,use_local=True)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = PositionalEncoding(decoder_embed_dim)

        self.decoder_blocks = nn.ModuleList([
            CrossBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,use_local=True)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, embed_dim, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.shot_block = BaseBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,use_local=False)
        self.shot_norm = norm_layer(embed_dim)
        self.fc = nn.Linear(embed_dim, 1)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=decoder_embed_dim,out_features=embed_dim)
        self.initialize_weights()

    def initialize_weights(self):

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, norm_feat,norm_mask=None):
        for blk in self.blocks:
            norm_feat = blk(norm_feat, norm_mask)
        norm_feat = self.norm(norm_feat)

        return norm_feat

    def forward_decoder(self, del_feat,norm_feat, ids, len_del, mk_mask=None):
        norm_feat = self.decoder_embed(norm_feat)
        del_feat = self.decoder_embed(del_feat)

        mask_tokens = self.mask_token.repeat(del_feat.shape[0],len_del, 1)

        x_ = torch.cat([del_feat, mask_tokens], dim=1)  # no cls token
        x_ = x_[:,ids]

        x = x_ + self.decoder_pos_embed(x_.shape[1])

        for blk in self.decoder_blocks:
            x = blk.forward_(x, norm_feat, mk_mask)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x.squeeze()

    def encoding_complete_frame(self, frame_feat, complete_mask=None):
        pos_embed = self.pos_embed(len(frame_feat))
        frame_feat = frame_feat + pos_embed
        frame_latent = self.forward_encoder(frame_feat, complete_mask)
        return frame_latent

    def forward(self, frame_feat, fuse_feat, ids, ids_restore, len_del,complete_mask=None, mk_mask=None):
        len_frame = len(frame_feat) - len_del
        # 加上位置编码
        pos_embed = self.pos_embed(len(frame_feat))
        frame_feat = frame_feat + pos_embed
        del_feat = frame_feat[:,ids][:,:len_frame]

        del_latent = self.forward_encoder(del_feat,mk_mask)
        feat = self.forward_decoder(del_latent,fuse_feat,ids_restore,len_del,complete_mask)

        return feat

    def shot_forward(self,shot_latent,is_train=False):
        shot_latent = self.shot_block(shot_latent)
        shot_latent = self.shot_norm(shot_latent)

        y = None
        if is_train:
            y = self.mlp(shot_latent).squeeze()
        score = self.fc(shot_latent.squeeze()).squeeze()
        return score, y

