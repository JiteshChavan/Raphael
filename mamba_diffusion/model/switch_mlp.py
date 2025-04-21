import torch
import torch.nn as nn
import torch.nn.functional as F
from mlp import MLP

# EC MOE NOT SINKHORN
# DONT NEED AUXILIARY LOSSES IN LOSS LANDSCAPE

class FeedForwardECMoe (nn.Module):
    """Expert Choice style Mixture of Experts feed forward layer with GELU activation
    
    Args:
        num_experts (int) : number of experts in the layer
        expert_capacity (float): capacity factor determining tokens per expert
        n_embd (int): Input and output dimension
        n_hidden (int): hidden layer channels
        hidden_base_mult (int) : Round hidden dimension upto next nearest multiple of this value
    """

    def __init__(self, num_experts, expert_capacity:float, n_embd, n_hidden, hidden_base_mult):
        super().__init__()
        self.num_experts = num_experts

        # scaling that restricts or boosts capacity of an expert (eg: 0.5x or 1.5x) we default to 1.0 I think
        self.expert_capacity = expert_capacity
        
        self.n_embd = n_embd
        self.hidden_base_mult = hidden_base_mult
        self.n_hidden = hidden_base_mult * ((n_hidden + hidden_base_mult -1) // hidden_base_mult)

        # to get softmax over num_experts for T tokens
        self.gate = nn.Linear (n_embd, num_experts, bias=False) # bias false makes sense in case model wants to 1 hot on experts

        # each expert goes from n_embd to n_hidden
        self.w1 = nn.Parameter (torch.ones (num_experts, n_embd, n_hidden))
        # non linear activation
        self.gelu = nn.GELU()
        # each expert goes from n_hidden to n_embd
        self.w2 = nn.Parameter (torch.ones (num_experts, n_hidden, n_embd))
    
    def forward (self, x:torch.Tensor):
        # extract shapes
        assert x.dim() == 3
        B, T, C = x.shape

        
        tokens_per_expert = int( self.expert_capacity * T / self.num_experts)

        # get scores, softmax for each token over experts, how appealing is an expert to each of the T tokens
        scores = self.gate (x) # (B, T, E) E is number of experts
        probs = F.softmax (scores, dim=-1) # probs for T tokens across experts

        probs_expert_looking_at_all_tokens = probs.permute(0, 2, 1) # (B, E, T)

        # gather top-tokens-per-expert
        # probs, idices
        expert_specific_token_probs, expert_specific_tokens = torch.topk (probs_expert_looking_at_all_tokens, tokens_per_expert, dim=-1)
        # (B, E, l)       (B, E, l) l is tokens per expert
        # create one hot vectors of T size for the selected tokens, so that we can extract from B,T,C
        # to construct xin for moe
        extract_from_x_one_hot = F.one_hot(expert_specific_tokens, num_classes=T).float() # (B, E, l, T)

        # Goal: (B, E, l C) from x
        xin = torch.einsum ('BElT, BTC -> BElC', extract_from_x_one_hot, x)
        
        # forward
        activation = torch.einsum ('BElC, ECH -> BElH', xin, self.w1) # (B, E, l, H)
        activation = self.gelu(activation)
        activation = torch.einsum ('BElH, EHC -> BElC', activation, self.w2) # (B, E, l, C)

        # scale the activation with gating score probs, so that stronger experts have greater influence on the outputs
        activation = activation * expert_specific_token_probs.unsqueeze(dim=-1)

        # use inner product to combine results of T tokens from all the different experts
        out = torch.einsum ('BElC, BElT -> BTC', activation, extract_from_x_one_hot)
        return out
    
    def custom_init (self, init_std:float):
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=init_std)