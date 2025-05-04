# Convolutional Positional embedding

import torch.nn as nn


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PosCNN(nn.Module):
    # stride = 1 for maintaining spatial resolution with 3x3 kernel
    # Grouped convolution, num_groups specifies the number of groups in which in and out channels are divided
    # so, in_channels in one particular group are used to compute out channels in corresponding group
    # no channel mixing across groups
    # this particular setup allows for in_channels = k * n_embd or k * n_groups
    # if k = 1 then thats depthwise convolution, gathers spatial information from one in channel to evaluate corresponding out channel
    # Very efficient spatial processing, useful for positional encoding or lightweight feature extraction
    def __init__(self, in_channels, n_embd=768, stride=1):
        super(PosCNN, self).__init__()
        assert in_channels == n_embd, f"in_channels: {in_channels} must be equal to groups=embed_dim, we are doing depthwise convolution\
            \n since the residual without channel up implies there is no change in shape"
        self.proj = nn.Sequential(nn.Conv2d(in_channels, n_embd, 3, stride, padding=1, bias=True, groups=n_embd))
        self.stride = stride

    def forward(self, x, H, W):
        # T = H * W
        B, T, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W) # (B, C, H, W)
        if self.stride == 1:
            # the fact that we are adding residual without channel up implies that they have the same channels
            # so it is indeed depthwise convolution
            x = self.proj(x) + x # (B, C, H, W)
        else:
            # can't add residual in case of spatial contraction since stride is not 1 (although padding is 1)
            # will suck at scaling for deep networks
            x = self.proj(x)
        
        # revert to original shape from (B, C, H, W)
        x = x.flatten(2).transpose(1, 2) # (B, T, C)
        return x

    # hacky: made a change so that it doesn't return hallucinated layer weights like before
    # which was gnarly, but then again assumes every layer in the sequential list has .weight
    # can be improved further but we will stop here for now
    def no_weight_decay(self):
        return ["proj.%d.weight" % i for i in range(len(self.proj))]


class AdaInPosCNN(nn.Module):
    def __init__(self, in_channels, n_embd=768, stride=1):
        super(AdaInPosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_channels, n_embd, 3, stride, padding=1, bias=True, groups=n_embd))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(n_embd, 2 * n_embd, bias=True))
        self.norm = nn.LayerNorm(n_embd)
        self.stride = stride

    def forward(self, x, c, H, W):
        B, N, C = x.shape
        feat_token = x
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = modulate(self.norm(x), shift, scale)
        return x

    def no_weight_decay(self):
        return ["proj.%d.weight" % i for i in range(4)]
