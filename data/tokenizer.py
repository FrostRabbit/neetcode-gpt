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
        chars = list(corpus)
        results = []
        for _ in range(num_merges):
            if len(chars) < 2:
                break
        
            count = Counter([(chars[i], chars[i+1]) for i in range(len(chars)-1)])
            count = sorted(count.items(), key=lambda item: (-item[1], item[0]))
            results.append(list(count[0][0]))
            merge = ''.join(count[0][0])

            i=0
            temp = []

            while i < len(chars):
                if i < len(chars) - 1 and chars[i]+chars[i+1] == merge:
                    temp.append(merge)
                    i+=2
                else:
                    temp.append(chars[i])
                    i+=1
            
            chars = temp
        return results
