import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        vocab = set()
        T = 0
        N = len(positive)
        pos = []
        neg = []
        
        for p, n in zip(positive,negative):
            p = p.strip().split()
            n = n.strip().split()
            T = max(T, len(p), len(n))
            pos.append(p)
            neg.append(n)
            vocab.update(p, n)
        vocab = sorted(vocab)
        vocab = dict(zip(vocab, [float(x+1) for x in range(len(vocab))]))
        temp = []
        print(vocab)
        for p in pos:
            temp.append([vocab[i] for i in p])
        for n in neg:
            temp.append([vocab[i] for i in n])
        tensor_list = [torch.tensor(i) for i in temp]
        result = nn.utils.rnn.pad_sequence(tensor_list, batch_first=True, padding_value=0)
        return result
