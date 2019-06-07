

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Callable, Collection, Optional, Tuple
from fastai.core import PathOrStr, Classes, ifnone, maybe_copy, KWArgs, extract_kwargs
from fastai.data import DataBunch, DatasetBase
from fastai.torch_core import fastai_types
from fastai.text.data import LanguageModelLoader, classifier_data
from .transform import Vocab, ProteinTokenizer
from .learner import ProtRNNLearner
from ..utils import softmax
from enum import IntEnum
from pathlib import Path
import pickle
import os
import json
import itertools
import random
import torch
from tqdm import tqdm, trange
from pandas_ml import ConfusionMatrix
import sys
import warnings
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

    def __init__(self, split_perc: float, lm_json_path: PathOrStr, class_json_path: PathOrStr, save_path: PathOrStr,
                 stage: ProtMtd=ProtMtd.JSON, seed: int=4, bs: int=64, tokenizer: ProteinTokenizer=ProteinTokenizer()):
        self.stage = stage
        self.seed = seed
        self.split_perc = split_perc
        self.lm_json_path = Path(lm_json_path)
        self.class_json_path = Path(class_json_path)
        self.save_path = Path(save_path)
        self.lm_path = self.save_path / 'lm'
        if not self.lm_path.exists():
            self.lm_path.mkdir()
        self.class_path = self.save_path / 'class'
        if not self.class_path.exists():
            self.class_path.mkdir()
        self.ltoi = {}
        self.stoi = {}
        self.tokenizer = tokenizer
        self.vocab = None
        self.lm_learner = None
        self.class_learner = None
        self.lm_tokens = None
        self.class_tokens = None
        self.lm_ids = None
        self.class_ids = None
        self.bs = bs
        self.mode = ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(self.mode)

    def _run(self):
        if self.stage == ProtMtd.JSON:
            self._read_jsons()
            self.stage += 1

        if self.stage == ProtMtd.TOK:
            self.tokenize()
            self.stage += 1

        if self.stage == ProtMtd.SPLIT:
            self.split_class_data()
            self.stage += 1

        if self.stage == ProtMtd.LMTRAIN:
            self.train_lm()
            self.stage += 1

        if self.stage == ProtMtd.CLASS_TRAIN:
            self.train_class()


    def _read_jsons(self):
        with open(self.lm_json_path) as fp:
            self.lm_raw_data = json.load(fp)

        with open(self.class_json_path) as fp:
            self.class_raw_data = json.load(fp)

    def tokenize(self, vocab_path=None):
        #pull seqs
        lm_seqs = [x['sequence'] for x in self.lm_raw_data]
        class_seqs = [x['sequence'] for x in self.class_raw_data]

        lm_tokens = self.tokenizer.process_all(lm_seqs)
        class_tokens = self.tokenizer.process_all(class_seqs)
        classlabel_list = list(set([x['label'] for x in self.class_raw_data]))
        l_stoi = {x:i for i,x in enumerate(classlabel_list)}
        l_itos = {v:k for k,v in l_stoi.items()}
        class_dict = {i:classlabel_list.index(i) for i in classlabel_list}

        if vocab_path == None:
            
            stoi = {}
            all_tokens = lm_tokens + class_tokens
            merged = list(itertools.chain(*all_tokens))
            for token in merged:
                if token not in stoi:
                    stoi[token] = len(stoi)

            itos = {v:k for k,v in stoi.items()}#create itos

            self.vocab = Vocab(itos, l_itos=l_itos)

        else:
            with open(vocab_path, 'rb') as pkl_object:
                vocab = pickle.load(pkl_object)

            self.vocab = vocab

        class_labels = np.array([class_dict[x['label']] for x in self.class_raw_data])
        class_names = np.array([x['name'] for x in self.class_raw_data])

        self.lm_tokens, self.lm_ids = lm_tokens, []
        self.class_tokens, self.class_ids, self.class_labels, self.vocab.class_dict, self.class_names = class_tokens, [], class_labels, class_dict, class_names
        self.class_seqs = class_seqs

        for tokens in lm_tokens:
            self.lm_ids.append(self.vocab.numericalize(tokens))
        for tokens in class_tokens:
            self.class_ids.append(self.vocab.numericalize(tokens))

        #dumps extracted model from learner
        with open('{}/vocab.pkl'.format(self.save_path), 'wb') as output:
            pickle.dump(self.vocab, output)

    def split_class_data(self):
        '''
        Splits classifier into training and test data according to split_perc
        Also calculates weights for labels to overrepresent underrepresented classes
        '''
        pass

    def train_lm(self, epochs=50, drop_mult=1):
        '''
        Creates an "LM DataBunch" that is essentially two dummy datasets (training and validation)
        Saves encoder from pre-trained language model
        '''

        random.seed(self.seed)
        random.shuffle(self.lm_ids)
        num_seqs = len(self.lm_ids)
        split_perc = 0.8
        train_ids = self.lm_ids[:int(num_seqs * split_perc)]
        valid_ids = self.lm_ids[int(num_seqs * split_perc):]
        print(num_seqs, len(train_ids), len(valid_ids))

        train_ds = ProteinDatasetLite(train_ids, None, self.vocab, 'train_lm')
        valid_ds = ProteinDatasetLite(valid_ids, None, self.vocab, 'valid_lm')
        datasets = [train_ds, valid_ds]
        dataloaders = [LanguageModelLoader(ds, self.bs) for ds in datasets]
        lm_data = DataBunch(*dataloaders)
        learn = ProtRNNLearner.language_model(lm_data, pad_token=self.vocab.pad_idx, drop_mult=drop_mult, mode=self.mode)
        learn.path = self.lm_path
        os.makedirs(self.lm_path / 'models', exist_ok=True)
        learn.unfreeze()
        learn.fit_one_cycle(epochs, slice(1e-4, 1e-2))
        #learn.fit(epochs, slice(1e-4, 1e-2))
        learn.save_encoder('enc')
        self.lm_learner = learn

    def train_class(self, epochs=50, drop_mult=1, encoder_path=None, complete_train=False):
        '''
        Creates an "Class DataBunch" that is essentially two datasets (training, validation)
        Saves final classifier model
        '''

        #create panda data frame with columns class_labels, class_ids
        df = pd.DataFrame({'class_ids': self.class_ids, 'class_labels': self.class_labels, 'class_names': self.class_names, 'seqeunces': self.class_seqs})

        if complete_train == False:
            train, test = train_test_split(df, test_size=.1, random_state=42, stratify=df[['class_labels']])
            train, valid = train_test_split(train, test_size=.1, random_state=42, stratify=train[['class_labels']])

            self.test_ids = np.array(test['class_ids'])
            self.test_labels = np.array(test['class_labels'])

            datasplit = {'training': train[['class_names', 'seqeunces', 'class_labels']].to_dict('records'),
                         'validating': valid[['class_names', 'seqeunces', 'class_labels']].to_dict('records'),
                         'testing': test[['class_names', 'seqeunces', 'class_labels']].to_dict('records')}

        if complete_train == True:
            train, valid = train_test_split(df, test_size=.1, random_state=42, stratify=df[['class_labels']])

            datasplit = {'training': train[['class_names', 'seqeunces', 'class_labels']].to_dict('records'),
                         'validating': valid[['class_names', 'seqeunces', 'class_labels']].to_dict('records')}

        self.train_ids = np.array(train['class_ids'])
        self.train_labels = np.array(train['class_labels'])

        self.valid_ids = np.array(valid['class_ids'])
        self.valid_labels = np.array(valid['class_labels'])

        train_ds = ProteinDatasetLite(self.train_ids, self.train_labels, self.vocab, 'train_class')
        valid_ds = ProteinDatasetLite(self.valid_ids, self.valid_labels, self.vocab, 'valid_class')

        datasets = [train_ds, valid_ds]
        class_data = classifier_data(datasets,self.save_path)
        datasplit['class_dict'] = self.vocab.class_dict

        with open('{}/data_split.json'.format(self.class_path), 'w') as json_data:
            json.dump(datasplit, json_data)

        print(self.mode)
        learn = ProtRNNLearner.classifier(class_data, pad_token=self.vocab.pad_idx, drop_mult=drop_mult,
                                          labelled_data=list(self.class_labels), mode=self.mode, qrnn=False)
        if encoder_path == None:
            learn.load_encoder_path('{}/models/enc.pth'.format(self.lm_path))
        else:
            learn.load_encoder_path('{}/enc.pth'.format(encoder_path))
        learn.unfreeze()

        learn.freeze_to(-1)
        learn.fit_one_cycle(epochs, slice(5e-3 / 2., 5e-3))
        learn.freeze_to(-2)
        learn.fit_one_cycle(epochs, slice(5e-3 / 2., 5e-3))
        learn.freeze_to(-3)
        learn.fit_one_cycle(epochs, slice(5e-3 / 2., 5e-3))
        learn.unfreeze()
        learn.fit_one_cycle(epochs, slice(2e-3 / 100, 2e-3))

        #dumps extracted model from learner
        torch.save(learn.model, self.save_path / 'models/final_model.p')
        if complete_train == False:
            with open('{}/models/final_model.pkl'.format(self.save_path), 'wb') as output:
                pickle.dump(learn.model, output)
                
        if complete_train == True:
            with open('{}/models/complete_model.pkl'.format(self.save_path), 'wb') as output:
                pickle.dump(learn.model, output)

        self.class_learner = learn
        self.final_model = learn.model


    @classmethod
    def predict(cls, model_path, vocab_path, orfs, bs=64):

        mode = ("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device(mode)
        model = torch.load(model_path, map_location=mode)

        with open(vocab_path, "rb") as fp:
            vocab = pickle.load(fp)

        outputs = []
        orf_lens = {len(x['sequence']) for x in orfs}
        orf_map = {x:[y for y in orfs if len(y['sequence'])==x] for x in orf_lens}

        pbar = tqdm(total=len(orfs))
        for orf_len, batch_orfs in orf_map.items():
            for i in range(0, len(batch_orfs), bs):
                batch_sub_orfs = batch_orfs[i:(i+bs)]
                tokens = []
                for orf in batch_sub_orfs:
                    tok_ids = vocab.numericalize([x for x in orf['sequence']])
                    tokens.append(tok_ids)
                num_in_batch = len(batch_sub_orfs)
                tokens = torch.tensor(np.transpose(np.asarray(tokens)), dtype=torch.long, device=device)
                predictions = model(tokens)[0]
                for j in range(num_in_batch):
                    pred = predictions[j]
                    softmax_preds = list(softmax(pred.detach().cpu().numpy())[0])
                    softmax_preds = [{"class":vocab.l_itos[k], "score":float(score)}
                                     for k, score in enumerate(softmax_preds)]
                    softmax_preds = sorted(softmax_preds, key=lambda x:x['score'])[::-1]
                    softmax_preds = [x for x in softmax_preds if x['score'] > 0.2]
                    orf = batch_sub_orfs[j]
                    outputs.append(dict(class_predictions=softmax_preds, name=orf['name']))
                pbar.update(len(batch_sub_orfs))
        pbar.close()

        return outputs


    def test_class(self, mode: str="GPU", model=None):
        '''
        Tests final model
        '''

        if model is None:
            model = self.final_model
        if mode is None:
            mode = self.mode
            
        y_pred = []
        overall_predictions = []

        for ary in tqdm(self.test_ids):

            ary = np.reshape(ary,(-1,1))

            # turn this array into a tensor
            tensor = torch.tensor(ary, dtype=torch.long, device=self.device)

            # wrap in a torch Variable
            variable = torch.autograd.Variable(tensor)

            # do the predictions
            predictions = model(variable)

            # convert back to numpy
            numpy_preds = predictions[0].data.cpu().numpy()
            softmax_preds = list(softmax(numpy_preds[0])[0])
            y_pred.append(softmax_preds.index(max(softmax_preds)))
            overall_predictions.append(softmax_preds)
        
        with open('{}/data_split.json'.format(self.class_path)) as json_data:
            datasplit = json.load(json_data)
        
        for x, y in zip(datasplit['testing'], overall_predictions):
            x['prediction'] = [round(float(i),2) for i in y]
        
        #store class dict in datasplit json
        datasplit['class_dict'] = self.vocab.class_dict
        
        with open('{}/data_split.json'.format(self.class_path), 'w') as output:
            json.dump(datasplit, output)
        
        #Displaying confusion table, stats

        orig_stdout = sys.stdout
        f = open('{}/stats.txt'.format(self.save_path), 'w')
        sys.stdout = f
        
        print("Model variables")
        print("")
        model_data_param = [['Variable', 'index', 'train_num', 'valid_num', 'test_num']]
        for key, value in self.vocab.class_dict.items():
            model_data_param.append([key, 
                                     value, 
                                     list(self.train_labels).count(value),
                                     list(self.valid_labels).count(value),
                                     list(self.test_labels).count(value),
                                     ])

        s = [[str(e) for e in row] for row in model_data_param]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        print('\n'.join(table))
        print("")

        cm = ConfusionMatrix(list(self.test_labels), y_pred)
        cm.print_stats()
        
        sys.stdout = orig_stdout
        f.close()



    @property
    def itos_file(self):
        return

    @property
    def ltoi_file(self):
        return


class ProteinDatasetLite:

    def __getitem__(self, idx: int) -> Tuple[int, int]:
        return self.ids[idx], self.labels[idx]

    def __len__(self) -> int:
        return len(self.ids)

    def __init__(self, ids, labels, vocab, name):
        self.ids = ids
        self.labels = labels
        self.vocab = vocab
        self.name = name
        self.classes = vocab.class_dict


DataFunc = Callable[[Collection[DatasetBase], PathOrStr, KWArgs], DataBunch]
fastai_types[DataFunc] = 'DataFunc'


def standard_data(datasets: Collection[DatasetBase], path: PathOrStr, **kwargs) -> DataBunch:
    """
    Simply create a `DataBunch` from the `datasets`.
    """
    return DataBunch.create(*datasets, path=path, **kwargs)

