import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
        for i in range(epochs):
            torch.manual_seed(i)
            start = torch.randint(0, data.shape[0] - context_length, (batch_size,))
            idx = start.unsqueeze(1) + torch.arange(context_length)
            X = data[idx]
            Y = data[idx+1].reshape(batch_size*context_length)
            logits = model(X).reshape(batch_size*context_length, vocab_size)
            loss = F.cross_entropy(logits, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return round(loss.item(), 4)
