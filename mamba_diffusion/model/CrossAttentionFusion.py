import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from timm.layers import use_fused_attn
from torch.jit import Final

####### SPLIT CHANNELS IN HALF WHILE CONSTRUCTING FORWARD PASS
# HALF FOR SPATIAL HALF FOR FREQUENCY

# TODO: PAPER: CAN WRITE EXPANDING QKV LINEAR N_HIDDEN IN PAPER
# NOT DONE IN DIMSUM, ARCHITECTURE MODIFICATION
class CrossAttentionFusion (nn.Module):
    def __init__ (self, n_embd, n_head=8, qk_norm=False, n_hidden=None, qkv_bias=True, norm_layer:nn.Module=nn.LayerNorm,swap_k=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.n_embd = n_embd
        # self.n_embd_half = n_embd // 2
        self.n_head = n_head
        self.swap_k = swap_k
        if n_hidden is not None:
            self.n_hidden = n_hidden
        else:
            self.n_hidden = self.n_embd

        assert n_hidden % n_head == 0, f"QKV hidden channels:{n_hidden} must be divisible by n_heads:{n_head}"
        
        # spatial, frequency
        self.qkv_spatial = nn.Linear (self.n_embd, 3*n_hidden, bias=qkv_bias)
        self.qkv_frequency = nn.Linear (self.n_embd, 3*n_hidden, bias=qkv_bias)

        # default affine is true
        self.q_ln_freq = norm_layer(n_hidden) if qk_norm else nn.Identity()
        self.k_ln_freq = norm_layer(n_hidden) if qk_norm else nn.Identity()
        self.q_ln_spatial = norm_layer(n_hidden) if qk_norm else nn.Identity()
        self.k_ln_spatial = norm_layer(n_hidden) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        
        # Projection layers project back to n_embd from n_hidden
        self.proj_from_space = nn.Linear (n_hidden, n_embd, bias=qkv_bias)
        self.proj_from_freq = nn.Linear (n_hidden, n_embd, bias=qkv_bias)
        self.proj.NANO_GPT_SCALE_INIT = 1
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward (self, freq, spat):
        B_freq, T_freq, C_freq = freq.shape
        B_spat, T_spat, C_spat = spat.shape

        # extract linear qkv for both freq and spat
        qkv_freq = self.qkv_frequency(freq) # (B, T, 3C')
        qkv_spat = self.qkv_spatial(spat)   # (B, T, 3C')

        qf, kf, vf = qkv_freq.split (self.n_hidden, dim = -1) # 3x (B, T, C')
        qs, ks, vs = qkv_spat.split (self.n_hidden, dim = -1) # 3x (B, T, C')

        # normalize q and k not v for some reason
        qf = self.q_ln_freq(qf)
        kf = self.k_ln_freq(kf)
        qs = self.q_ln_spatial(qs)
        ks = self.k_ln_spatial(ks)

        qf = qf.view (B_freq, T_freq, self.n_head, self.n_hidden // self.n_head).transpose(1,2) # (B, nh, T, hs)
        qs = qs.view (B_spat, T_spat, self.n_head, self.n_hidden // self.n_head).transpose(1,2) # (B, nh, T, hs)
        
        kf = kf.view (B_freq, T_freq, self.n_head, self.n_hidden // self.n_head).transpose(1,2) # (B, nh, T, hs)
        ks = ks.view (B_spat, T_spat, self.n_head, self.n_hidden // self.n_head).transpose(1,2) # (B, nh, T, hs)


        if not self.swap_k:
            from_space = F.scaled_dot_product_attention(qf,ks,vs, dropout_p=self.attn_drop.p if self.training else 0.0) # (B, nh, T, hs)
            from_freq = F.scaled_dot_product_attention(qs, kf,vf, dropout_p=self.attn_drop.p if self.training else 0.0) # (B, nh, T, hs)
        else:
            from_space = F.scaled_dot_product_attention(qs, kf, vs, dropout_p=self.attn_drop.p if self.training else 0.0) # (B, nh, T, hs)
            from_freq = F.scaled_dot_product_attention(qf, ks, vf, dropout_p=self.attn_drop.p if self.training else 0.0) # (B, nh, T, hs)
        
        from_space = from_space.transpose(1, 2).contiguous().view(B_spat, T_spat, C_spat) # (B, T, C')
        from_freq = from_freq.transpose(1, 2).contiguous().view(B_freq, T_freq, C_freq) # (B, T, C')

        from_space = self.proj_from_space (from_space) # (B, T, C)
        from_space = self.proj_drop (from_space) # (B, T, C)

        from_freq = self.proj_from_freq (from_freq) # (B, T, C)
        from_freq = self.proj_drop (from_freq) # (B, T, C)

        spat = spat + from_freq
        freq = freq + from_space

        return spat, freq
