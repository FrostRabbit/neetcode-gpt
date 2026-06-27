import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.k_w = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.q_w = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.v_w = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        K = self.k_w(embedded)
        Q = self.q_w(embedded)
        V = self.v_w(embedded)
        scores = torch.einsum("btd, bjd -> btj", Q, K) / math.sqrt(Q.shape[-1])
        mask = torch.tril(torch.ones(Q.shape[1],Q.shape[1]))==0
        scores = scores.masked_fill(mask, float('-inf'))
        scores = nn.functional.softmax(scores,dim=2) @ V
        return torch.round(scores, decimals=4)