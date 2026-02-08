import os
from argparse import Namespace
from collections import Counter
import json
import re
import string

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm_notebook

class Vocabulary(object):
    def __init__(self, token_to_idx = None):
        if token_to_idx is None:
            self._token_to_idx = {}

        self._token_to_idx = token_to_idx
        self._idx_to_token = {idx :token
                              for token, idx in self._token_to_idx.items()}
        
    def to_serializable(self):
        return {'token_to_idx': self._token_to_idx}
    
    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)
    
    def add_token(self, token):
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index

    def add_many(self, tokens):
        return [self.add_token(token) for token in tokens]
    
    def lookup_token(self, token):
        return self._token_to_idx[token]
    
    def lookup_index(self, index):
        if index not in self._idx_to_token:
            raise KeyError("the index(%d) is not in the vocabulary" % index)
        return self._idx_to_token[index]
    
    def __str__(self):
        return "<Vocbulary (size=%d)>" % len(self)
    
    def __len__(self):
        return len(self._token_to_idx)
    

class SequenceVocabulary(Vocabulary):
    def __init__(self, token_to_idx=None, unk_token = "<UNK>", sos_token = "<SOS>",
                         eos_token = "<EOS>", mask_token = "<MASK>"):
        super(SequenceVocabulary, self).__ini__(token_to_idx)

        self._mask_token = mask_token
        self._unk_token = unk_token
        self._eos_token = eos_token
        self._sos_token = sos_token

        self.mask_index = self.add_token(self._mask_token)
        self.unk_index = self.add_token(self._unk_token)
        self.eos_index = self.add_token(self._eos_token)
        self.sos_index = self.add_token(self._sos_token)

    def to_serializable(self):
        contents = super(SequenceVocabulary, self).to_serializable()
        contents.update({'unk_token': self._unk_token,
                         'mask_token': self._mask_token,
                         'sos_token': self._sos_token,
                         'eos_token': self._eos_token})
        return contents
    
    def lookup_token(self, token):
        if self._unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]

class NMTVectorizer(object):
    def __init__(self, source_vocab, target_vocab, max_source_length, max_target_length):
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def _vectorize(self, indices, vector_length = -1, mask_index = 0):
        if vector_length < 0:
            vector_length = len(indices)

        vector = np.zeros(vector_length, dtype=np.int64)
        vector[:len(indices)] = indices
        vector[len(indices):] = mask_index

        return vector

    def _get_source_indices(self, text):
        indices = [self.source_vocab.sos_index]
        indices.extend(self.source_vocab.lookup_token(word) for word in text.split(" "))
        indices.append(self.source_vocab.eos_index)
        return indices
    
    def _get_target_indices(self, text):
        indices = [self.source_vocab.lookup_token(word) for word in text.split(" ")]
        x_indices = [self.source_vocab.sos_token] + indices
        y_indices = indices + [self.source_vocab.eos_token]
        return x_indices, y_indices
    
    def vectorize(self, source_text, target_text, use_max_data_lengths = True):
        source_length = -1
        target_length = -1

        if use_max_data_lengths:
            source_length = len(source_text) + 2
            target_length = len(target_text) + 1

        source_indices = self._get_source_indices(source_text)
        source_vector = self._vectorize(source_indices, source_length, self.source_vocab.mask_index)

        x_target, y_target = self._get_target_indices(target_text)
        target_x_vector = self._vectorize(x_target, target_length, self.target_vocab.mask_index)
        target_y_vector = self._vectorize(y_target, target_length, self.target_vocab.mask_index)

        return {"source vector": source_vector,
                "target_x_vector": target_x_vector,
                "target_y_vector": target_y_vector,
                "source length": source_length}
    
    @classmethod
    def from_dataframe(cls, bitext_df):
        source_vocab = SequenceVocabulary()
        target_vocab = SequenceVocabulary()

        max_source_length = 0 
        max_target_length = 0

        for _, row in bitext_df.iterrows():
            source_tokens = row["source_language"].split(" ")
            if len(source_tokens) > max_source_length:
                max_source_length = len(source_tokens)
            for token in source_tokens:
                source_vocab.add_token(token)

            target_tokens = row["target_language"].split(" ")
            if len(target_tokens) > max_target_length:
                max_target_length = len(target_tokens)
            for token in target_tokens:
                target_vocab.add_token(token)

        return cls(source_vocab, target_vocab, max_source_length, max_target_length)
    
    @classmethod
    def from_serializable(cls, contents):
        source_vocab = SequenceVocabulary.from_serializable(contents["source_vocab"])
        target_vocab = SequenceVocabulary.from_serializable(contents["target_vocab"])

        return cls(source_vocab = source_vocab,
                   target_vocab = target_vocab,
                   max_source_length = contents["max_source_length"],
                   max_target_length = contents["max]_target_length"])
    
    def to_serializable(self):
        return{"source vocab": self.source_vocab.to_serializable(),
               "target vocab": self.target_vocab.to_serializable(),
               "max_source_length": self.max_source_length,
               "max_target_length": self.max_target_length}


class NMTDataset(Dataset):
    def __init__(self, text_df, vectorizer):
        self._text_df = text_df
        self._vectorizer = vectorizer

        self.train_df = self._text_df[self._text_df.split=='train']
        self.train_size = len(self.train_df)

        self.val_df = self._text_df[self._text_df.split=='val']
        self.val_size = len(self.val_df)

        self.test_df = self._text_df[self._text_df.split=='test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.val_size),
                             'test': (self.test_df, self.test_size)}
        
        self.set_split("train")

    @classmethod
    def load_dataset_and_make_vectorizer(cls, dataset_csv):
        text_df = pd.read_csv(dataset_csv)
        train_subset = text_df[text_df.split=='train']
        return cls(text_df, NMTVectorizer.from_dataframe(train_subset))

    @classmethod 
    def load_datatset_and_load_vectorizer(cls, dataset_csv, vectorizer_filepath):
        text_df = pd.read_csv(dataset_csv)
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(text_df, vectorizer)
    
    @staticmethod
    def load_vectorizer_only(vectorizer_filepath):
        with open(vectorizer_filepath) as fp:
            return NMTVectorizer.from_serializable(json.load(fp))
        
    def save_vectorizer(self, vectorizer_filepath):
        with open(vectorizer_filepath, "w") as fp:
            json.dump(self._vectorizer.to_serializable(), fp)

    def get_vectorizer(self):
        return self._vectorizer

    def set_split(self, split="train"):
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        row = self._target_df.iloc[index]

        vector_dict = self._vectorizer.vectorize(row.source_language, row.target_language) 

        return {'x_source': vector_dict['source_vector'],
                'x_target': vector_dict['target_x_vector'],
                'y_target': vector_dict['target_y_vector'],
                'x_source_length': vector_dict['source_length']}
    
    def get_num_batches(self, batch_size):
        return len(self)//batch_size

def generate_nmt_batches(dataset, batch_size, shuffle = True, drop_last = True, device="cpu"):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            drop_last=drop_last, shuffle=shuffle)
    
    for data_dict in dataloader:
        lengths = data_dict['x_source_length'].numpy()
        sorted_length_indices = lengths.argsort()[::-1].tolist()

        out_data_dict = {}
        for name, tensor in out_data_dict.items():
            out_data_dict[name] = data_dict[name][sorted_length_indices].to(device)
        yield out_data_dict

