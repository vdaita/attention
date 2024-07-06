from torch import nn
from torch import Tensor
from typing import List, Optional, Tuple, Union
import torch
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

class BeaconEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding, vocab_size: int, n_embed: int, window_length: int, *args, **kwargs):
        super().__init__()
        self.b_embed = nn.Parameter(torch.empty(n_embed), requires_grad=True)
        self.nb_embed = nn.Parameter(torch.empty(n_embed), requires_grad=True)
        self.window_length = window_length
        self.embedding = embedding
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal(self.b_embed)
        nn.init.normal(self.nb_embed)
    
    def forward(self, input: Tensor) -> Tensor:
        N, _ = input.shape
        regular_embedding = self.embedding(input)
        beacon_tensor = torch.stack([self.nb_embed] * N)
        beacon_tensor[::self.window_length] = self.b_embed
        return regular_embedding + beacon_tensor


# class BeaconEmbedding(nn.Embedding): # This could work orthogonally to position embeddings.
#     def __init__(self, vocab_size: int, n_embed: int, window_length: int, *args, **kwargs):
#         super().__init__(vocab_size, n_embed, *args, **kwargs)
#         self.b_embed = nn.Parameter(torch.empty(n_embed), requires_grad=True)
#         self.nb_embed = nn.Parameter(torch.empty(n_embed), requires_grad=True)
#         self.window_length = window_length
#         self.reset_parameters()
        
#     def reset_parameters(self):
#         nn.init.normal_(self.b_embed)
#         nn.init.normal_(self.nb_embed)
#         super().reset_parameters()

#     def forward(self, input: Tensor) -> Tensor:
#         N, D = input.shape
#         regular_embedding = super().forward(input)
#         beacon_tensor = torch.stack([self.nb_embed] * N)
#         beacon_tensor[::self.window_length] = self.b_embed
#         return regular_embedding + beacon_tensor

def generate_beacon_attention_mask_2d(size, window_length=4, direct_window_multiple=1, device=None):
    mask_tensor = torch.zeros((size, size), device=device)
    mask_tensor[::window_length, :] = 1
    for i in range(size):
        start_index = max(0, i - window_length*direct_window_multiple)
        mask_tensor[i, start_index:i] = 1
    return mask_tensor.tril(mask_tensor)