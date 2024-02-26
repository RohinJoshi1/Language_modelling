from encoder.base import Tokenizer,get_stats,merge
from typing import Dict,List, Tuple

class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def train(self, text,vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        text_bytes = text.encode('utf-8')
        ids = list(map(int, text_bytes))
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(0,255)}
        for i in range(num_merges):
           stats=get_stats(ids)
           pair = max(stats, key= lambda x : stats[x])
           #Replace the pair in list
           idx = 256 + i
           ids = merge(ids, pair,idx)
           merges[pair] = idx
           vocab[pair] = vocab[pair[0]] + vocab[pair[1]]

           if verbose:
               print(f"merge {i+1}/{num_merges}: {pair}-> {idx} ({vocab[idx]})")

        self.merges = merges
        self.vocab = vocab

    def encode(self, text):
        #1.Convert to UTF-8
        text_bytes = text.encode('utf-8')
        # Map to int range 0-255
        ids = list(map(int , text_bytes))
        while len(ids) >= 2:
            #Returns dict of pairs with the count
            stats = get_stats(ids)
            #Fetch the pair with lowest merge index
            pair = min(stats, key = lambda p: self.merges.get(p,float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids


    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8",errors="replace")
        return text
