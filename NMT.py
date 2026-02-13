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

class NMTEncoder(nn.Module):
    def __init__(self, num_embeddings, embedding_size, rnn_hidden_size):
        super(NMTEncoder, self).__init__()
        self.source_embedding = nn.Embedding(num_embeddings, embedding_size, padding_idx = 0)

        self.birnn = nn.GRU(embedding_size, rnn_hidden_size, bidirectional=True, batch_first=True)

    def forward(self, x_source, x_lengths):
        x_embedded = self.source_embedding(x_source)

        x_packed = pack_padded_sequence(x_embedded, x_lengths.detach().cpu().numpy(), batch_first=True)

        x_birnn_out, x_birnn_h = self.birnn(x_packed)
        x_birnn_h = x_birnn_h.permute(1, 0, 2)

        x_birnn_h = x_birnn_h.contiguous().view(x_birnn_h.size(0), -1)
        x_unpacked, _ = pad_packed_sequence(x_birnn_out, batch_first=True)

        return x_unpacked, x_birnn_h

def verbose_attention(encoder_state_vectors, query_vector):
    batch_size, num_vectors, vector_size = encoder_state_vectors.size()
    vector_scores = torch.sum(encoder_state_vectors * query_vector.view(batch_size, 1, vector_size),
                              dim=2)
    vector_probabilities = F.softmax(vector_scores, dim=1)
    weighted_vectors = encoder_state_vectors * vector_probabilities.view(batch_size, num_vectors, 1)
    context_vectors = torch.sum(weighted_vectors, dim=1)
    return context_vectors, vector_probabilities, vector_scores

def terse_attention(encoder_state_vectors, query_vector):
    vector_scores = torch.matmul(encoder_state_vectors, query_vector.unsqueeze(dim=2)).squeeze()
    vector_probabilities = F.softmax(vector_scores, dim=2)
    context_vector = torch.matmul(encoder_state_vectors.transpose(-2,-1), vector_probabilities.unsqueeze(dim=2)).squeeze
    return context_vector, vector_probabilities

class NMTDecoder(nn.Module):
    def __init__(self, num_embeddings, embedding_size, rnn_hidden_size, sos_index):
        super(NMTDecoder, self).__init__
        self._rnn_hidden_size = rnn_hidden_size
        self.target_embedding = nn.Embedding(num_embeddings, embedding_size, padding_idx=0)
        self.gru_cell = nn.GRUCell(embedding_size + rnn_hidden_size,
                                rnn_hidden_size)
        self.hidden_map = nn.Linear(rnn_hidden_size, rnn_hidden_size)
        self.classifier = nn.Linear(rnn_hidden_size*2, num_embeddings)
        self.sos_index = sos_index
        self._sampling_temparture = 3

    def _init_indices(self, batch_size):
        return torch.ones(batch_size, dtype=torch.int64) * self.sos_index
    
    def _init_context_vector(self, batch_size):
        return torch.zeros(batch_size, self._rnn_hidden_size) 
    
    def forward(self, encoder_state, initial_hidden_state, target_sequence, sample_probability =0.0):
        if target_sequence is None:
            sample_probability = 1.0
        else:
            #input = (Batch, sequence)
            target_sequence = target_sequence.permute(1,0)
            output_sequence_size = target_sequence.size(0)

        h_t = self.hidden_map(initial_hidden_state)
        batch_size = encoder_state.size(0)
        context_vectors = self._init_context_vector(batch_size)
        y_t_index = self._init_indices(batch_size)

        h_t = h_t.to(encoder_state.device)
        context_vectors = context_vectors.to(encoder_state.device)
        y_t_index = y_t_index.to(encoder_state.deivce)

        output_vectors = []
        self._cached_ht = []
        self._cached_p_attn = []
        self._cached_decoder_state = encoder_state.cpu().detach().numpy()

        for i in range(output_sequence_size):
            use_sample = np.random.random() < sample_probability
            if not use_sample:
                y_t_index = target_sequence[i]

            y_input_vector = self.target_embedding(y_t_index)
            rnn_input = torch.cat([y_input_vector, context_vectors], dim=1)

            h_t = self.gru_cell(rnn_input, h_t)
            self._cached_ht.append(h_t.cpu().detach().numpy())

            context_vectors, p_attn, _ = verbose_attention(encoder_state_vectors=encoder_state,
                                                           query_vector=h_t)
            self._cached_p_attn.append(p_attn.cpu().detach().numpy())

            prediction_vector = torch.cat([context_vectors, h_t], dim=1)
            score_for_y_t_index = self.classifier(F.dropout(prediction_vector,0.3))

            if use_sample:
                p_y_t_index = F.softmax(score_for_y_t_index*self._sampling_temparture, dim=1)
                _, y_t_index = torch.max(p_y_t_index, 1)

            output_vectors.append(score_for_y_t_index)

        output_vectors = torch.stack(output_vectors).permute(1,0,2)

        return output_vectors
    

class NMTModel(nn.Module):
    def __init__(self, source_vocab_size, source_embedding_size, target_vocab_size,
                 target_embedding_size, encoding_size, target_sos_index):
        super(NMTModel, self).__init__()
        self.encoder = NMTEncoder(num_embeddings=source_vocab_size,
                                  embedding_size=source_embedding_size,
                                  rnn_hidden_size=encoding_size)
        decoding_size = 2*encoding_size
        self.decoder = NMTDecoder(num_embeddings=target_vocab_size,
                                  embedding_size=target_embedding_size,
                                  rnn_hidden_size=decoding_size,
                                  sos_index=target_sos_index)
        
    def forward(self, x_source, x_source_lengths, target_sequence, sample_probability=0.0):
        encoder_state, final_hidden_state = self.encoder(x_source, x_source_lengths)
        decoder_states = self.decoder(encoder_state=encoder_state, 
                                      rnn_hidden_state=final_hidden_state, 
                                      target_sequence=target_sequence, 
                                      sample_probability=sample_probability)
        return decoder_states

def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def make_train_state(args):
    return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'learning_rate': args.learning_rate,
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1,
            'model_filename': args.model_state_file}

def update_train_state(args, model, train_state):
    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(), train_state['model_filename'])
        train_state['stop_early'] = False

    elif train_state['epoch_index'] > 0:
        loss_tm1, loss_t = train_state['val_loss'][-2:]
        if loss_t >= loss_tm1:
            train_state['early_stopping'] += 1
        else:
            if loss_t < train_state['early_stopping_best_val']:
                torch.save(model.state_dict(), train_state['model_filename'])
                train_state['early_stopping_best_val'] = loss_t

            train_state['early_stopping_step'] = 0
        
            train_state['stop_early'] = \
            train_state['early_stopping_step'] >= args.early_stopping_criteria

    return train_state

def normalize_sizes(y_pred, y_true):
    if len(y_pred.size()) == 3:
        y_pred = y_pred.contigous(-1, y_pred.size(2))
    if len(y_true.size()) == 2:
        y_true = y_true.contigous().view(-1)
    return y_pred, y_true

def compute_accuracy(y_pred, y_true, mask_index):
    y_pred, y_true = normalize_sizes(y_pred, y_true)
    _, y_pred_indices = y_pred.max(dim=1)

    correct_indices = torch.eq(y_pred_indices, y_true).float()
    valid_indices = torch.ne(y_pred_indices, mask_index).float()

    n_correct = (correct_indices * valid_indices).sum().item()
    n_valid = valid_indices.sum().item()

    return n_correct/n_valid*100

def sequence_loss(y_pred, y_true, mask_index):
    y_pred, y_true = normalize_sizes(y_pred, y_true)
    return F.cross_entropy(y_pred, y_true, ignore_index=mask_index)

args = Namespace(dataset_csv="simplest_eng_fra.csv",
                 vectorizer_file="vectorizer.json",
                 model_state_file="model.pth",
                 save_dir="model_storage",
                 reload_from_files=False,
                 expand_filepaths_to_save_dir=True,
                 cuda=True,
                 seed=9248,
                 learning_rate=5e-4,
                 batch_size=32,
                 num_epochs=100,
                 early_stopping_criteria=5,              
                 source_embedding_size=24, 
                 target_embedding_size=24,
                 encoding_size=32,
                 catch_keyboard_interrupt=True)

if args.expand_filepaths_to_save_dir:
    args.vectorizer_file = os.path.join(args.save_dir,
                                        args.vectorizer_file)

    args.model_state_file = os.path.join(args.save_dir,
                                         args.model_state_file)
    
    print("Expanded filepaths: ")
    print("\t{}".format(args.vectorizer_file))
    print("\t{}".format(args.model_state_file))


if not torch.cuda.is_available():
    args.cuda = False

args.device = torch.device("cuda" if args.cuda else "cpu")
    
print("Using CUDA: {}".format(args.cuda))

set_seed_everywhere(args.seed, args.cuda)

handle_dirs(args.save_dir)

if args.reload_from_files and os.path.exists(args.vectorizer_file):
        dataset = NMTDataset.load_dataset_and_load_vectorizer(args.dataset_csv,
                                                          args.vectorizer_file)
else:
    dataset = NMTDataset.load_dataset_and_make_vectorizer(args.dataset_csv)
    dataset.save_vectorizer(args.vectorizer_file)

vectorizer = dataset.get_vectorizer()

model = NMTModel(source_vocab_size=len(vectorizer.source_vocab), 
                 source_embedding_size=args.source_embedding_size, 
                 target_vocab_size=len(vectorizer.target_vocab),
                 target_embedding_size=args.target_embedding_size, 
                 encoding_size=args.encoding_size,
                 target_bos_index=vectorizer.target_vocab.begin_seq_index)

if args.reload_from_files and os.path.exists(args.model_state_file):
    model.load_state_dict(torch.load(args.model_state_file))
    print("Reloaded model")
else:
    print("New model")


model = model.to(args.device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                           mode='min', factor=0.5,
                                           patience=1)

mask_index = vectorizer.target_vocab.mask_index
train_state = make_train_state(args)

epoch_bar = tqdm_notebook(desc='training routine', 
                          total=args.num_epochs,
                          position=0)

dataset.set_split('train')
train_bar = tqdm_notebook(desc='split=train',
                          total=dataset.get_num_batches(args.batch_size), 
                          position=1, 
                          leave=True)
dataset.set_split('val')
val_bar = tqdm_notebook(desc='split=val',
                        total=dataset.get_num_batches(args.batch_size), 
                        position=1, 
                        leave=True)

try:
    for epoch_index in range(args.num_epochs):
        sample_probability = (20 + epoch_index) / args.num_epochs
        
        train_state['epoch_index'] = epoch_index

        # Iterate over training dataset

        # setup: batch generator, set loss and acc to 0, set train mode on
        dataset.set_split('train')
        batch_generator = generate_nmt_batches(dataset, 
                                               batch_size=args.batch_size, 
                                               device=args.device)
        running_loss = 0.0
        running_acc = 0.0
        model.train()
        
        for batch_index, batch_dict in enumerate(batch_generator):
            optimizer.zero_grad()

            y_pred = model(batch_dict['x_source'], 
                           batch_dict['x_source_length'], 
                           batch_dict['x_target'],
                           sample_probability=sample_probability)

            loss = sequence_loss(y_pred, batch_dict['y_target'], mask_index)

            loss.backward()


            optimizer.step()

            running_loss += (loss.item() - running_loss) / (batch_index + 1)

            acc_t = compute_accuracy(y_pred, batch_dict['y_target'], mask_index)
            running_acc += (acc_t - running_acc) / (batch_index + 1)

            train_bar.set_postfix(loss=running_loss, acc=running_acc, 
                                  epoch=epoch_index)
            train_bar.update()

        train_state['train_loss'].append(running_loss)
        train_state['train_acc'].append(running_acc)

        dataset.set_split('val')
        batch_generator = generate_nmt_batches(dataset, 
                                               batch_size=args.batch_size, 
                                               device=args.device)
        running_loss = 0.
        running_acc = 0.
        model.eval()

        for batch_index, batch_dict in enumerate(batch_generator):
            y_pred = model(batch_dict['x_source'], 
                           batch_dict['x_source_length'], 
                           batch_dict['x_target'],
                           sample_probability=sample_probability)

            loss = sequence_loss(y_pred, batch_dict['y_target'], mask_index)

            running_loss += (loss.item() - running_loss) / (batch_index + 1)
            
            acc_t = compute_accuracy(y_pred, batch_dict['y_target'], mask_index)
            running_acc += (acc_t - running_acc) / (batch_index + 1)

            val_bar.set_postfix(loss=running_loss, acc=running_acc, 
                            epoch=epoch_index)
            val_bar.update()

        train_state['val_loss'].append(running_loss)
        train_state['val_acc'].append(running_acc)

        train_state = update_train_state(args=args, model=model, 
                                         train_state=train_state)

        scheduler.step(train_state['val_loss'][-1])

        if train_state['stop_early']:
            break
        
        train_bar.n = 0
        val_bar.n = 0
        epoch_bar.set_postfix(best_val=train_state['early_stopping_best_val'])
        epoch_bar.update()
        
except KeyboardInterrupt:
    print("Exiting loop")