from typing import Dict, List, Tuple
from collections import Counter
class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        stoi = list(sorted(set(text)))
        stoi = {v:i for i,v in enumerate(stoi)}
        itos = {v:k for k,v in stoi.items()}
        return (stoi, itos)

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        text = list(text)
        return [stoi[t] for t in text]

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        return ''.join([itos[i] for i in ids])
