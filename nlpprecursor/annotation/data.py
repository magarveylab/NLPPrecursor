
import numpy as np
from typing import Tuple
from fastai.core import PathOrStr
from ..classification.transform import Vocab, ProteinTokenizer
from enum import IntEnum
import pickle
import json
import random
import torch
from tqdm import tqdm
import warnings
from .models.lstm_crf import LSTMCRF
from .models.crf import ConditionalRandomField as CRF
from pathlib import Path
warnings.filterwarnings("ignore")



ProtMtd = IntEnum('ProtMtd', 'JSON TOK SPLIT LMTRAIN CLASTRAIN')

class DatasetGenerator:
    '''
    Create the LM dataset, and the classifier dataset (train/validation can be shared)
    Maybe do some pre-processing to make sure train and test don't share too much sequence similarity?
        - This could be slow though
    Do all the tokenization / numericalization here so that there isn't
    # Do all the tokenization / numericalization here, and then just implement

    '''

    def __init__(self, split_perc, json_path: PathOrStr, save_path, seed: int=4, bs: int=64,):
        self.split_perc = split_perc
        self.json_path = json_path
        self.save_path = Path(save_path)
        if not self.save_path.exists():
            self.save_path.mkdir()

        self.seed = seed
        self.bs = bs
        self.mode = ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(self.mode)
        self.tokenizer = ProteinTokenizer()


    def run(self, epochs=100):
        self._read_jsons()
        self.tokenize()
        self.train(epochs)
        self.save_model()

    def _read_jsons(self):
        with open(self.json_path) as fp:
            self.raw_data = json.load(fp)
        self.raw_data = [x for x in self.raw_data if len(x['sequence']) < 200]
        #self.raw_data = [x for x in self.raw_data if "*" not in x['sequence']]

    def tokenize(self):
        seqs = [x['sequence'] for x in self.raw_data]
        labels = [x['labels'] for x in self.raw_data]
        self.seq_names = [x['name'] for x in self.raw_data]


        tokens = self.tokenizer.process_all(seqs)
        label_tokens = self.tokenizer.process_all(labels)

        max_len = max([len(x) for x in tokens])

        stoi = {}
        for seq in tokens:
            for token in seq:
                if token not in stoi:
                    stoi[token] = len(stoi)
        if 'pad' not in stoi:
            stoi['pad'] = len(stoi)
        itos = {v:k for k,v in stoi.items()}
        l_stoi = {}
        for seq in label_tokens:
            for token in seq:
                if token not in l_stoi:
                    l_stoi[token] = len(l_stoi)
        if 'pad' not in l_stoi:
            l_stoi['pad'] = len(l_stoi)
        l_itos = {v:k for k,v in l_stoi.items()}

        print(stoi)

        self.vocab = Vocab(itos, l_itos=l_itos, max_len=max_len)


        self.seqs = tokens
        self.labels = label_tokens
        self.seq_ids, self.label_ids = [], []

        for seq_tokens in tqdm(self.seqs):
            len_diff = max_len - len(seq_tokens)
            tok_ids = self.vocab.numericalize(seq_tokens)
            pad_ids = np.asarray([self.vocab.pad_idx]*len_diff)
            self.seq_ids.append(np.concatenate((tok_ids, pad_ids)))
        for lab_tokens in tqdm(self.labels):
            len_diff = max_len - len(lab_tokens)
            tok_ids = self.vocab.numericalize(lab_tokens, True)
            pad_ids = np.asarray([self.vocab.l_pad_idx]*len_diff)
            self.label_ids.append(np.concatenate((tok_ids, pad_ids)))

        with open('{}/vocab.pkl'.format(self.save_path), 'wb') as fp:
            pickle.dump(self.vocab, fp)




    def train(self, epochs=50):

        random.seed(self.seed)
        matched = list(zip(self.seq_ids, self.label_ids, self.seq_names))
        random.shuffle(matched)
        num_seqs = len(matched)
        split_perc = self.split_perc

        training_data = matched[:int(num_seqs * split_perc)]

        testing_data = matched[int(num_seqs * split_perc):]
        valid_data = training_data[int(len(training_data)*split_perc):]
        training_data = training_data[:int(len(training_data) * split_perc)]


        train_ds = ProteinSequenceDataset(training_data, self.vocab, 'train')
        valid_ds = ProteinSequenceDataset(valid_data, self.vocab, 'valid')
        test_ds = ProteinSequenceDataset(testing_data, self.vocab, 'test')


        name_data_map = {x:'train' for x in train_ds.seq_names}
        name_data_map.update({x: 'valid' for x in valid_ds.seq_names})
        name_data_map.update({x: 'test' for x in test_ds.seq_names})

        with open(self.save_path / 'datasplit.json', "w") as fp:
            json.dump(name_data_map, fp)


        crf = self.create_crf()
        self.model = LSTMCRF(vocab=self.vocab, crf=crf, pad_idx=self.vocab.pad_idx)


        self.fit(train_ds, valid_ds, test_ds, epochs=epochs)

    def create_crf(self):
        allowed_transitions = [('start', 'before'),
                               ('before', 'prop'),
                               ('before', 'before'),
                               ('prop', 'after'),
                               ('prop', 'prop'),
                               ('prop', 'stop'),
                               ('stop', 'pad'),
                               ('after', 'after'),
                               ('after', 'stop'),]
        allowed_token_transitions = []
        l_stoi = self.vocab.l_stoi
        print(l_stoi)
        for trans in allowed_transitions:
            allowed_token_transitions.append((l_stoi[trans[0]], l_stoi[trans[1]]))

        transition_of_interest = (l_stoi['before'], l_stoi['prop'])
        crf = CRF(num_tags=self.vocab.n_labels, constraints=allowed_token_transitions,
                  transition_of_interest=transition_of_interest)
        return crf

    def fit(self, train_ds, valid_ds, test_ds, epochs):

        optimizer = torch.optim.Adam(params=self.model.get_trainable_params())

        for _ in tqdm(range(epochs)):
            self.model.train()
            train_ds.shuffle()
            batches = []
            for i in range(0, len(train_ds), self.bs):
                batch = train_ds[i:i+self.bs]
                batches.append(batch)

            batch_loss: torch.Tensor = 0.

            for batch in tqdm(batches):
                src, tgt = batch[0], batch[1]
                src, tgt = torch.tensor(src, device=self.device, dtype=torch.long), \
                           torch.tensor(tgt, device=self.device, dtype=torch.long)
                loss = self.model(src, tgt)
                batch_loss += loss
            batch_loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 5.)
            optimizer.step()
            perc_accuracy = self.evaluate(valid_ds)

            print("Loss:", batch_loss)
            print("Validation accuracy:", perc_accuracy)



    def evaluate(self, ds):
        self.model.eval()
        batches = []
        for i in range(0, len(ds), self.bs):
            batch = ds[i:i + self.bs]
            batches.append(batch)
        perc_accuracies = []
        for batch in tqdm(batches):
            src, tgt = batch[0], batch[1]
            src = torch.tensor(src, device=self.device, dtype=torch.long)
            preds = self.model.predict(src)
            for batch_idx in range(len(batch)):
                targs, pred = tgt[batch_idx], preds[batch_idx][0]
                total_right = 0
                for i, pred_tag in enumerate(pred):
                    if i==0:
                        continue
                    if i==len(pred)-1:
                        continue
                    real_tag = targs[i]
                    if int(real_tag) == int(pred_tag):
                        total_right += 1
                total_length = len(pred)-2
                perc_right = total_right / float(total_length)
                perc_accuracies.append(perc_right)
        return np.mean(perc_accuracies)

    def save_model(self):
        model_path = self.save_path / "model.p"
        torch.save(self.model, model_path)


    @classmethod
    def predict(cls, model_path, vocab_path, props):
        mode = ("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device(mode)

        model = torch.load(model_path, map_location=mode)
        with open(vocab_path, "rb") as fp:
            vocab = pickle.load(fp)

        outputs = []
        for prop in tqdm(props):
            seq = prop['sequence']
            tokens = ['start'] + [x for x in seq] + ['stop']
            tok_ids = vocab.numericalize(tokens)
            len_diff = vocab.max_len = len(tok_ids)
            pad_ids = np.asarray([vocab.pad_idx] * len_diff)
            tok_ids = np.concatenate((tok_ids, pad_ids))
            tok_ids = torch.tensor([tok_ids], dtype=torch.long, device=device)
            preds, viterbi_score = model.predict(tok_ids)[0]
            preds = [vocab.l_itos[x] for x in preds]

            sublists = []
            for i, pred in enumerate(preds):
                for j in range(i+1, len(preds)):
                    sublist = preds[i:j]
                    if len(set(sublist)) == 1 and sublist[0]=="prop":
                        sublists.append((i,j))

            try:
                longest_stretch = sorted(sublists, key=lambda x:x[1]-x[0])[-1]
            except IndexError:
                outputs.append(dict(name=prop['name'], cleavage_prediction={'status':"failed"}))
                continue
            propeptide_sequece = "".join([x for x in tokens[longest_stretch[0]:longest_stretch[1]]])

            output = {'name':prop['name']}
            cleave_pred = {}
            cleave_pred['sequence'] = propeptide_sequece
            cleave_pred['start'] = seq.find(propeptide_sequece)
            cleave_pred['stop'] = cleave_pred['start'] + len(propeptide_sequece)
            cleave_pred['score'] = viterbi_score
            cleave_pred['name'] = prop['name']
            cleave_pred['status'] = "success"
            output['cleavage_prediction'] = cleave_pred
            outputs.append(output)

        return outputs



    @classmethod
    def evaluate_later(cls, model_path, vocab_path, datasplit_path, data_path):

        mode = ("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device(mode)

        model = torch.load(model_path, map_location=mode)

        with open(vocab_path, "rb") as fp:
            vocab = pickle.load(fp)

        with open(datasplit_path) as fp:
            datamap = json.load(fp)

        with open(data_path) as fp:
            raw_data = json.load(fp)

        raw_data = [x for x in raw_data if len(x['sequence']) < 200]
        testing_seqs = [x for x in raw_data if datamap[x['name']] != 'train']

        for prop in tqdm(testing_seqs):
            seq = prop['sequence']
            tokens = seq.split("-")
            tok_ids = vocab.numericalize(tokens)
            len_diff = vocab.max_len = len(tok_ids)
            pad_ids = np.asarray([vocab.pad_idx] * len_diff)
            tok_ids = np.concatenate((tok_ids, pad_ids))
            tok_ids = torch.tensor([tok_ids], dtype=torch.long, device=device)
            preds = model.predict(tok_ids)[0][0]
            preds = "-".join([vocab.l_itos[x] for x in preds])
            prop['prediction'] = preds

        return testing_seqs




class ProteinSequenceDataset:

    def __getitem__(self, idx: int) -> Tuple[int, int]:
        return self.source[idx], self.target[idx]

    def __len__(self) -> int:
        return len(self.source)

    def __iter__(self):
        for src, tgt in zip(self.source, self.target):
            yield src, tgt

    def __init__(self, raw_data, vocab, name):
        self.source, self.target, self.seq_names = zip(*raw_data)
        self.vocab = vocab
        self.name = name
        self.is_test = name=='test'
        self.classes = vocab.l_stoi

    def shuffle(self) -> None:
        """Shuffle source and targets together."""
        combined = list(zip(self.source, self.target, self.seq_names))
        random.shuffle(combined)
        self.source, self.target, self.seq_names = zip(*combined)
