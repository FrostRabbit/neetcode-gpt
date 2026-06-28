from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        return [self.tokenize(str(x), vocab) for x in numbers]

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        return len(self.tokenize(text, vocab))

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        word_count = len(text.split())
        return round(len(self.tokenize(text, vocab))/word_count,4)
    
    def tokenize(self, text:str, vocab: Dict[str, int]) -> List[str]:
        tokens = []
        i = 0
        n = len(text)
        while i < n:
            for j in range(n,i,-1):
                if text[i:j] in vocab:
                    match = text[i:j]
                    i = j
                    break

            if match:
                tokens.append(match)
            else:
                tokens.append(text[i])
                i+=1
        return tokens
