from abc import abstractmethod
from typing_extensions import Dict,List


def get_stats(ids:List[int],counts = None)->Dict:
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair,0)+1
    return counts


def merge(ids, pair, idx):
    '''
        In the list of integers, replace all occurrences of pair with new token "idx"
    '''
    new_ids = []
    i = 0
    while i<len(ids):
        if i < len(ids)-1 and ids[i]==ids[i+1]:
            new_ids.append(idx)
            i+=2
        else:
            new_ids.append(ids[i])
            i+=1
    return new_ids




class Tokenizer:
    def __init__(self):
        self.merges ={}
        self.pattern = ""
        self.special_tokens = {}
        self.vocab = self._build_vocab

    @abstractmethod
    def encode(self, text)->List:
        pass

    @abstractmethod
    def decode(self, ids)->str:
        pass

    @abstractmethod
    def train(self, text, vocab_size, verbose=False):
        pass


    def _build_vocab(self):
        vocab = {idx : bytes([idx]) for idx in range(256)}
        for (p0,p1), idx in self.merges.items():
            vocab[idx] = vocab[p0]+vocab[p1]
        for special,idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab
