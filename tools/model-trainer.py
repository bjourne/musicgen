# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
"""
Generative MIDI models trained on MOD files
===========================================

Usage:
    model-trainer.py [options] <code-type> lstm ( train | generate )
        <corpus-path> --emb-size=<i> --batch-size=<i>
        --dropout=<f> --rec-dropout=<f>
        --lstm1-units=<i> --lstm2-units=<i>
    model-trainer.py [options]
        <code-type> transformer ( train | generate ) <corpus-path>
        --dropout=<f> --batch-size=<i>

Options:
    -h --help              show this screen
    -v --verbose           print more output
    --lr=<f>               learning rate
    --epochs=<i>           epochs to train for
    --seq-len=<i>          training sequence length
"""
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from collections import Counter, namedtuple
from docopt import docopt
from musicgen.code_utils import CODE_MIDI_MAPPING
from musicgen.corpus import load_index
from musicgen.generation import (notes_to_audio_file,
                                 notes_to_midi_file)
from musicgen.parser import UnsupportedModule, load_file
from musicgen.pcode import (is_pcode_learnable,
                            mod_to_pcode,
                            pcode_to_notes,
                            pcode_short_pause,
                            pcode_long_pause)
from musicgen.scode import (mod_file_to_scode,
                            scode_to_midi_file,
                            scode_short_pause,
                            scode_long_pause)
from musicgen.tensorflow import (lstm_model,
                                 select_strategy,
                                 sequence_to_samples,
                                 transformer_model)
from musicgen.utils import (SP, CharEncoder,
                            file_name_for_params,
                            find_subseq, flatten,
                            load_pickle_cache,
                            save_pickle)
from pathlib import Path
from random import randrange, shuffle
from tensorflow.data import Dataset
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import *
from tensorflow.nn import softmax
from time import time
from tqdm import trange

import numpy as np
import tensorflow as tf

CodeInfo = namedtuple('CodeInfo', ['to_code_fn',
                                   'short_pause', 'long_pause',
                                   'is_learnable_fn'])

# TODO: Fix scode
CODE_TYPES = {
    'pcode_abs' : CodeInfo(lambda m: mod_to_pcode(m, False),
                           pcode_short_pause(),
                           pcode_long_pause(),
                           is_pcode_learnable),
    'pcode_rel' : CodeInfo(lambda m: mod_to_pcode(m, True),
                           pcode_short_pause(),
                           pcode_long_pause(),
                           is_pcode_learnable),
    'scode_abs' : CodeInfo(lambda m: mod_file_to_scode(m, False),
                           scode_short_pause(),
                           scode_long_pause(),
                           None),
    'scode_rel' : CodeInfo(lambda m: mod_file_to_scode(m, True),
                           scode_short_pause(),
                           scode_long_pause(),
                           None)
}

def mod_file_to_code_w_progress(i, n, file_path, info):
    SP.header('[ %4d / %4d ] PARSING %s' % (i, n, file_path))
    try:
        mod = load_file(file_path)
    except UnsupportedModule:
        SP.print('Unsupported module format.')
        SP.leave()
        return None
    code = list(info.to_code_fn(mod))
    if not info.is_learnable_fn(code):
        code = None
    SP.leave()
    return code

def build_cache(path, shuffle_file, mods, info):
    mod_files = [path / mod.genre / mod.fname for mod in mods]
    n = len(mod_files)

    # Cache the shuffle to make trained models more comparable.
    shuffle_path = path / shuffle_file
    def rebuild_fn():
        indices = list(range(n))
        shuffle(indices)
        return indices
    indices = load_pickle_cache(shuffle_path, rebuild_fn)

    encoder = CharEncoder()
    arrs = []
    for i, p, in enumerate(sorted(mod_files)):
        code = mod_file_to_code_w_progress(i + 1, n, p, info)
        if not code:
            continue
        arrs.append((p.name, encoder.encode_chars(code, True)))

    # Shuffle according to indices.
    tmp = [(i, e) for (i, e) in zip(indices, arrs)]
    arrs = [e for (_, e) in sorted(tmp)]

    return encoder, arrs

class LSTMParams:
    def __init__(self, docopt_args):
        self.rec_dropout = float(docopt_args['--rec-dropout'])
        self.emb_size = int(docopt_args['--emb-size'])
        self.lstm1_units = int(docopt_args['--lstm1-units'])
        self.lstm2_units = int(docopt_args['--lstm2-units'])

    def as_file_name_part(self):
        fmt = '%.2f-%03d-%04d-%04d'
        args = (self.rec_dropout, self.emb_size,
                self.lstm1_units, self.lstm2_units)
        return fmt % args

class TransformerParams:
    def __init__(self, docopt_args):
        pass

    def as_file_name_part(self):
        return ''

class ModelParams:
    @classmethod
    def from_docopt_args(cls, args):
        code_type = args['<code-type>']
        model_type = 'lstm' if args['lstm'] else 'transformer'

        if model_type == 'lstm':
            type_params = LSTMParams(args)
        else:
            type_params = TransformerParams(args)

        dropout = float(args['--dropout'])
        batch_size = int(args['--batch-size'])
        lr = float(args['--lr'])
        seq_len = int(args['--seq-len'])
        epochs = int(args['--epochs'])
        return cls(code_type, model_type,
                   dropout, batch_size, lr, seq_len, epochs,
                   type_params)

    def __init__(self,
                 code_type, model_type,
                 dropout, batch_size, lr, seq_len, epochs,
                 type_params):
        self.code_type = code_type
        self.model_type = model_type

        self.dropout = dropout
        self.batch_size = batch_size
        self.lr = lr
        self.seq_len = seq_len
        self.epochs = epochs
        self.type_params = type_params

    def weights_file(self):
        fmt = 'weights_%s_%s-%.2f-%03d-%.5f-%03d-%03d-%s.h5'
        args = (self.code_type, self.model_type,
                self.dropout, self.batch_size,
                self.lr, self.seq_len, self.epochs,
                self.type_params.as_file_name_part())
        return fmt % args

    def model(self, vocab_size, batch_size, stateful):
        if self.model_type == 'transformer':
            return transformer_model(vocab_size, 128, 2048, 0.2, 8, 16)
        return lstm_model(vocab_size, self.emb_size,
                          self.lstm1_units, self.lstm2_units,
                          self.dropout, self.rec_dropout,
                          stateful, batch_size)

class TrainingData:
    def __init__(self, code_type):
        if code_type not in CODE_TYPES:
            s = ', '.join(CODE_TYPES)
            raise ValueError('<code-type> must be one of %s' % s)
        self.code_type = code_type
        self.info = CODE_TYPES[code_type]

    def load_disk_cache(self, path, kb_limit):
        index = load_index(path)
        mods = [mod for mod in index.values()
                if (mod.n_channels == 4
                    and mod.format == 'MOD'
                    and mod.kb_size <= kb_limit)]
        size_sum = sum(mod.kb_size for mod in mods)
        real_prefix = 'cache_%s' % self.code_type
        params = size_sum, kb_limit
        cache_file = file_name_for_params(real_prefix, 'pickle', params)
        cache_path = path / cache_file
        shuffle_file = file_name_for_params('shuffle', 'pickle', params)
        def rebuild_fn():
            return build_cache(path, shuffle_file, mods, self.info)
        data = load_pickle_cache(cache_path, rebuild_fn)
        self.encoder, self.arrs = data

    def load_mod_file(self, p):
        code = mod_file_to_code_w_progress(1, 1, p, self.info.to_code_fn)
        self.encoder = CharEncoder()
        self.arrs = [(p.name, self.encoder.encode_chars(code, True))]

    def print_historgram(self):
        codes = [arr for (name, arr) in self.arrs]
        seq = np.concatenate(codes)
        ix_counts = Counter(seq)
        ch_counts = {self.encoder.decode_char(ix) : cnt
                     for (ix, cnt) in ix_counts.items()}
        total = sum(ch_counts.values())
        SP.header('%d TOKENS %d TYPES' % (total, len(ch_counts)))
        for (cmd, arg), cnt in sorted(ch_counts.items()):
            SP.print('%s %3d %10d' % (cmd, arg, cnt))
        SP.leave()

    def split_3way(self, train_frac, valid_frac):
        n_mods = len(self.arrs)
        n_train = int(n_mods * train_frac)
        n_valid = int(n_mods * valid_frac)
        n_test = n_mods - n_train - n_valid

        parts = (self.arrs[:n_train],
                 self.arrs[n_train:n_train + n_valid],
                 self.arrs[n_train + n_valid:])
        tds = [TrainingData(self.code_type) for _ in range(3)]
        for td, part in zip(tds, parts):
            td.arrs = part
            td.encoder = self.encoder
        return tds

    def code_to_audio_file(self, seq, file_path):
        code = self.encoder.decode_chars(seq)
        notes = self.info.to_notes_fn(code)
        notes_to_audio_file(notes, file_path, CODE_MIDI_MAPPING, True)

    def code_to_pickle_file(self, seq, file_path):
        code = self.encoder.decode_chars(seq)
        save_pickle(file_path, code)

    def flatten(self):
        pause = self.encoder.encode_chars(self.info.long_pause, False)
        padded = []
        for name, arr in self.arrs:
            if len(arr) > 0:
                padded.append(arr)
                padded.append(pause)
        return np.concatenate(padded)

    def to_samples(self, length):
        seq = self.flatten()
        return sequence_to_samples(seq, length)

def generate_sequences(model, temps, top_ps, seed, n_samples):
    SP.print('Priming the model with %d tokens.' % seed.shape[1])
    for i in range(seed.shape[1] - 1):
        model.predict(seed[:, i:i + 1])

    preds = [seed[:, -1:]]
    n_temps = len(temps)
    n_top_ps = len(top_ps)

    SP.print('Predicting %d tokens.' % n_samples)
    for _ in trange(n_samples, unit = 'preds', mininterval = 0.5):
        last_word = preds[-1]
        Ps = model.predict(last_word)[:, 0, :]
        Ps = softmax(Ps).numpy()

        # First temperature weighing
        ixs = []
        for i in range(n_temps):
            P = np.exp(np.log(Ps[i]) / temps[i])
            P = P / P.sum()
            ixs.append(np.random.choice(len(P), p = P))

        # Then top-p sampling
        for i in range(n_top_ps):
            P = Ps[i + n_temps]
            prob_ixs = np.argsort(-P)

            PC = np.cumsum(P[prob_ixs])
            top_n = len(PC[PC <= top_ps[i]]) + 1

            # Clear the prob of those who didn't make it.
            P[prob_ixs[top_n:]] = 0.0

            P = P / P.sum()

            ixs.append(np.random.choice(len(P), p = P))

        preds.append(np.array(ixs, dtype = np.int32))

    SP.leave()
    return [[int(preds[j][i]) for j in range(n_samples)]
            for i in range(n_temps + n_top_ps)]

def generate_sequences_transformer(model, temps, top_ps, seed, n_samples):
    seed = np.array(seed, dtype = np.int32)
    n_temps = len(temps)
    n_top_ps = len(top_ps)
    preds = []

    SP.print('Predicting %d tokens.' % n_samples)
    for _ in trange(n_samples, unit = 'preds', mininterval = 0.5):
        y = model.predict(seed)
        Ps = y[:, -1, :]
        Ps = softmax(Ps).numpy()

        ixs = []
        for i in range(n_temps):
            P = np.exp(np.log(Ps[i]) / temps[i])
            P = P / P.sum()
            ixs.append(np.random.choice(len(P), p = P))

        # Then top-p sampling
        for i in range(n_top_ps):
            P = Ps[i + n_temps]
            prob_ixs = np.argsort(-P)

            PC = np.cumsum(P[prob_ixs])
            top_n = len(PC[PC <= top_ps[i]]) + 1

            # Clear the prob of those who didn't make it.
            P[prob_ixs[top_n:]] = 0.0
            P = P / P.sum()
            ixs.append(np.random.choice(len(P), p = P))
        ixs = np.array(ixs, dtype = np.int32)
        preds.append(ixs)

        seed = np.hstack((seed, np.expand_dims(ixs, 1)))
        seed = np.roll(seed, -1, axis = 1)
        seed[:,-1] = ixs
    return [[int(preds[j][i]) for j in range(n_samples)]
            for i in range(n_temps + n_top_ps)]

def generate_music(temps, top_ps, data, path, params):
    n_temps = len(temps)
    n_top_ps = len(top_ps)
    n_preds = n_temps + n_top_ps

    # Often more than one full song - not great.
    n_samples = 100
    n_seed = 256

    SP.header('%d PREDICTIONS' % n_preds)
    with select_strategy().scope():
        model = params.model(len(data.encoder.ix2ch), n_preds, True)
        model.load_weights(str(path / params.weights_file()))
        model.reset_states()
    seq = data.flatten()
    long_pause = data.info.long_pause
    long_pause = data.encoder.encode_chars(long_pause, False)
    long_pause = long_pause.tolist()

    while True:
        idx = randrange(len(seq) - n_seed - n_samples)
        seed = seq[idx:idx + n_seed]
        seed_seq = seed.tolist()
        n_unique = len(set(seed_seq))
        if list(find_subseq(seed_seq, long_pause)):
            SP.print('Long pause in seed, regenerating.')
            continue
        if n_unique < 5:
            SP.print('To few different tokens, regenerating.')
            continue
        break
    SP.print('Seed %d+%d (%d unique tokens).' % (idx, n_seed, n_unique))

    seed = np.repeat(np.expand_dims(seed, 0), n_preds, axis = 0)
    if params.model_type == 'transformer':
        seqs = generate_sequences_transformer(model, temps, top_ps,
                                              seed, n_samples)
    else:
        seqs = generate_sequences(model, temps, top_ps, seed, n_samples)

    # Add the original
    seqs.append(seq[idx + n_seed:idx + n_seed + n_samples])
    seqs = np.array(seqs)
    seed = np.vstack((seed, seed[0]))

    # Cut half the seed because it is too long to listen to it all.
    # seed = seed[:,n_seed // 2:]

    join = np.repeat(np.expand_dims(long_pause, 0), len(seqs), axis = 0)
    seqs = np.hstack((seed, join, seqs))

    prefix = '%s-%07d' % (data.code_type, idx)

    file_names = ['%s-t%.3f.pickle' % (prefix, t) for t in temps]
    file_names += ['%s-p%.3f.pickle' % (prefix, p) for p in top_ps]
    file_names.append('%s-orig.pickle' % prefix)
    for file_name, seq in zip(file_names, seqs):
        data.code_to_pickle_file(seq, path / file_name)
    SP.leave()

def train_model(train, valid, path, params):
    vocab_size = len(train.encoder.ix2ch)
    with select_strategy().scope():
        model = params.model(vocab_size, None, False)
        optimizer = RMSprop(learning_rate = params.lr)
        loss_fn = SparseCategoricalCrossentropy(from_logits = True)
        model.compile(
            optimizer = optimizer,
            loss = loss_fn,
            metrics = ['sparse_categorical_accuracy'])
    model.summary()

    weights_path = path / params.weights_file()
    if weights_path.exists():
        SP.print('Loading weights from %s...' % weights_path)
        model.load_weights(str(weights_path))
    else:
        SP.print('Weights file %s not found.' % weights_path)

    cb_best = ModelCheckpoint(
        str(weights_path),
        monitor = 'val_loss',
        verbose = 1,
        save_weights_only = True,
        save_best_only = True,
        mode = 'min')
    reduce_lr = ReduceLROnPlateau(
        factor = 0.2, patience = 8, min_lr = params.lr / 100)
    stopping = EarlyStopping(patience = 30)
    callbacks = [cb_best, reduce_lr, stopping]
    SP.print('Batching samples...')
    train_ds = train.to_samples(params.seq_len) \
        .batch(params.batch_size, drop_remainder = True)
    valid_ds = valid.to_samples(params.seq_len) \
        .batch(params.batch_size, drop_remainder = True)

    model.fit(x = train_ds,
              validation_data = valid_ds,
              epochs = params.epochs, callbacks = callbacks,
              verbose = 1)

def main():
    # Prologue
    args = docopt(__doc__, version = 'Train MOD model 1.0')
    SP.enabled = args['--verbose']
    path = Path(args['<corpus-path>'])
    np.set_printoptions(linewidth = 160)
    do_generate = args['generate']

    # Hyperparameters
    params = ModelParams.from_docopt_args(args)

    # Load data and split it
    data = TrainingData(params.code_type)
    if path.is_dir():
        data.load_disk_cache(path, 150)
        train, valid, test = data.split_3way(0.8, 0.1)
    else:
        data.load_mod_file(path)
        train = valid = test = data
    data.print_historgram()

    args = len(train.arrs), len(valid.arrs), len(test.arrs)
    SP.print('Train/valid/test split %d/%d/%d' % args)

    weights_file = params.weights_file()
    if do_generate:
        temps = [0.9, 1.0, 1.05, 1.10, 1.15]
        top_ps = [0.9, 0.94, 0.98, 0.99, 0.999]
        generate_music(temps, top_ps, valid, path, params)
    else:
        train_model(train, valid, path, params)

if __name__ == '__main__':
    main()
