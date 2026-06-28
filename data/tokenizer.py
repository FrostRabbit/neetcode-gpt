from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        chars = " ".join(list(corpus))
        results = []
        for _ in range(num_merges):
            if len(chars) < 2:
                break
            words = chars.split()
            count = Counter([(words[i], words[i+1]) for i in range(len(words)-1)])
            pair = sorted(count.items(), key=lambda item: (-item[1], item[0]))[0][0]
            print(pair)
            results.append(list(pair))
            chars = chars.replace(' '.join(pair), ''.join(pair))
        return results
