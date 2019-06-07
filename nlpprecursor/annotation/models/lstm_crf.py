

import torch
import torch.nn as nn
from ...classification.transform import Vocab
from .crf import ConditionalRandomField as CRF
from .. import utils


class LSTMCRF(nn.Module):

    def __init__(self,
                 vocab: Vocab,
                 crf: CRF,
                 hidden_dim: int=100,
                 layers: int=1,
                 dropout: float=0.1,
                 bidir: bool=True,
                 embedding_size: int=100,
                 pad_idx=0) -> None:
        super(LSTMCRF, self).__init__()
        self.mode = ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(self.mode)
        assert vocab.n_labels == crf.num_tags
        self.vocab = vocab
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(len(vocab.itos), embedding_size, torch.tensor(vocab.pad_idx, device=self.device))
        self.pad_idx = pad_idx

        self.rnn = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_dim,
            num_layers=layers,
            bidirectional=bidir,
            dropout=dropout if layers > 1 else 0,
            batch_first=True
        )

        self.rnn_output_size = hidden_dim*2 if bidir else hidden_dim
        self.rnn_to_crf = nn.Linear(self.rnn_output_size, self.vocab.n_labels)
        self.crf = crf
        self.to_cuda()

    def get_trainable_params(self):
        out = list()
        out.append({"params": self.embedding.parameters()})
        out.append({"params": self.rnn.parameters()})
        out.append({"params": self.rnn_to_crf.parameters()})
        out.append({"params": self.crf.parameters()})
        return out

    def to_cuda(self):
        models = [self.embedding, self.dropout, self.rnn, self.rnn_to_crf, self.crf]
        for model in models:
            model.to(self.device)


    def predict(self, words: torch.Tensor):

        mask = utils.sequence_mask(words, self.pad_idx)
        embeddings = self.embedding(words)
        out, hidden = self.rnn(embeddings)
        drops = self.dropout(out)
        feats = self.rnn_to_crf(drops)
        preds = self.crf.viterbi_tags(feats, mask)
        return preds

    def forward(self, words: torch.Tensor, labs: torch.Tensor):
        mask = utils.sequence_mask(words, self.pad_idx)
        embeddings = self.embedding(words)
        out, hidden = self.rnn(embeddings)
        drops = self.dropout(out)
        feats = self.rnn_to_crf(drops)
        loglik = self.crf(feats, labs, mask=mask)
        return -1. * loglik

