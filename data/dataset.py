import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]], List[List[str]]]:
        # 1. Tokenize by splitting on whitespace: raw_dataset.split()
        # 2. Generate batch_size random start indices using torch.randint()
        #    Range: [0, len(tokens) - context_length)
        # 3. For each index i, X = tokens[i:i+context_length], Y = tokens[i+1:i+1+context_length]
        torch.manual_seed(0)
        words = raw_dataset.split()
        start = torch.randint(0, len(words) - context_length, (batch_size,)).tolist()
        X = []
        Y = []
        for s in start:
            X.append(words[s:s+context_length])
            Y.append( words[s+1:s+context_length+1])
        return (X, Y)
