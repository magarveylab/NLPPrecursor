
from typing import Collection, List, Callable
from fastai.core import num_cpus, partition_by_cores
from concurrent.futures import ProcessPoolExecutor
import numpy as np




class BaseProteinTokenizer:
    """Basic class for a tokenizer function."""
    def __init__(self, lang: str):
        self.lang = lang

    def tokenizer(self, t: str) -> List[str]:
        return t.split("-")

    def add_special_cases(self, toks: Collection[str]):
        pass


class ProteinTokenizer:
    """Put together rules, a tokenizer function and a language to tokenize text with multiprocessing."""

    def __init__(self, tok_func: Callable=BaseProteinTokenizer, lang: str='prot', n_cpus: int=None):
        self.tok_func = tok_func
        self.lang = lang
        self.n_cpus = n_cpus or num_cpus()//2

    def __repr__(self) -> str:
        res = f'Tokenizer {self.tok_func.__name__} in {self.lang}'
        return res

    def process_text(self, t: str, tok: BaseProteinTokenizer) -> List[str]:
        """Process one text `t` with tokenizer `tok`."""
        return tok.tokenizer(t)

    def _process_all_1(self, texts: Collection[str]) -> List[List[str]]:
        """Process a list of `texts` in one process."""
        tok = self.tok_func(self.lang)
        return [self.process_text(t, tok) for t in texts]

    def process_all(self, texts: Collection[str]) -> List[List[str]]:
        """Process a list of `texts`."""
        if self.n_cpus <= 1: return self._process_all_1(texts)
        with ProcessPoolExecutor(self.n_cpus) as e:
            return sum(e.map(self._process_all_1, partition_by_cores(texts, self.n_cpus)), [])


class Vocab():
    """Contain the correspondence between numbers and tokens and numericalize."""

    def __init__(self, itos, l_itos=None, max_len=None):
        self.itos = itos
        self.stoi = {value: key for key, value in self.itos.items()}
        try:
            self.pad_idx = self.stoi['pad']
        except KeyError:
            self.pad_idx = None
        self.l_itos = l_itos
        if self.l_itos is not None:
            self.l_stoi = {value: key for key, value in self.l_itos.items()}
            try:
                self.l_pad_idx = self.l_stoi['pad']
            except KeyError:
                self.l_pad_idx = None
        self.pad_char = 'pad'
        self.max_len = max_len

    @property
    def n_labels(self):
        return len(self.l_itos)


    def numericalize(self, t: Collection[str], is_labels: bool=False) -> List[int]:
        """Convert a list of tokens `t` to their ids."""
        stoi = self.stoi if not is_labels else self.l_stoi
        return np.asarray([stoi[w] for w in t])

    def textify(self, nums:Collection[int]) -> str:
        """Convert a list of `nums` to their tokens."""
        return ' '.join([self.itos[i] for i in nums])
