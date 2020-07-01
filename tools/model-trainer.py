# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
"""
Generative MIDI models trained on MOD files
===========================================

Usage:
    model-trainer.py [options] <code-type> ( train | generate )
        <corpus-path> --emb-size=<i> --batch-size=<i>
        --dropout=<f> --rec-dropout=<f>
        --lstm1-units=<i> --lstm2-units=<i> --lr=<f>
        --seq-len=<i> --epochs=<i>

Options:
    -h --help              show this screen
    -v --verbose           print more output
"""
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from collections import Counter, namedtuple
from docopt import docopt
from musicgen.code_utils import CODE_MIDI_MAPPING
from musicgen.corpus import load_index
from musicgen.generation import notes_to_audio_file
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
from musicgen.tensorflow import select_strategy, sequence_to_samples
from musicgen.utils import (SP, CharEncoder,
                            file_name_for_params,
                            find_subseq, flatten, load_pickle_cache)
from pathlib import Path
from random import randrange, shuffle
from tensorflow.data import Dataset
from tensorflow.keras import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import *
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import *
from tensorflow.nn import softmax
from time import time
from tqdm import trange

import numpy as np
import tensorflow as tf

CodeInfo = namedtuple('CodeInfo', ['to_code_fn', 'to_notes_fn',
                                   'short_pause', 'long_pause',
                                   'is_learnable_fn'])
# TODO: Fix scode
CODE_TYPES = {
    'pcode_abs' : CodeInfo(lambda m: mod_to_pcode(m, False),
                           lambda c: pcode_to_notes(c, False),
                           pcode_short_pause(),
                           pcode_long_pause(),
                           is_pcode_learnable),
    'pcode_rel' : CodeInfo(lambda m: mod_to_pcode(m, True),
                           lambda c: pcode_to_notes(c, False),
                           pcode_short_pause(),
                           pcode_long_pause(),
                           is_pcode_learnable),
    'scode_abs' : CodeInfo(lambda m: mod_file_to_scode(m, False),
                           lambda c, fp: scode_to_midi_file(c, fp, False),
                           scode_short_pause(),
                           scode_long_pause(),
                           None),
    'scode_rel' : CodeInfo(lambda m: mod_file_to_scode(m, True),
                           lambda c, fp: scode_to_midi_file(c, fp, True),
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

class ModelParams:
    @classmethod
    def from_docopt_args(cls, args):
        code_type = args['<code-type>']
        dropout = float(args['--dropout'])
        rec_dropout = float(args['--rec-dropout'])
        emb_size = int(args['--emb-size'])
        batch_size = int(args['--batch-size'])
        lstm1_units = int(args['--lstm1-units'])
        lstm2_units = int(args['--lstm2-units'])
        lr = float(args['--lr'])
        seq_len = int(args['--seq-len'])
        epochs = int(args['--epochs'])
        return cls(code_type,
                   dropout, rec_dropout,
                   lstm1_units, lstm2_units,
                   emb_size, batch_size,
                   seq_len, epochs, lr)

    def __init__(self, code_type,
                 dropout, rec_dropout,
                 lstm1_units, lstm2_units,
                 emb_size, batch_size,
                 seq_len, epochs, lr):
        self.code_type = code_type
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.lstm1_units = lstm1_units
        self.lstm2_units = lstm2_units
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.epochs = epochs
        self.lr = lr

    def weights_file(self):
        args = (self.code_type, self.epochs,
                self.emb_size, self.batch_size,
                self.dropout, self.rec_dropout,
                self.lstm1_units, self.lstm2_units,
                self.lr, self.seq_len)
        fmt = 'weights_%s-%03d-%03d-%03d-%.2f-%.2f-%03d-%03d-%.5f-%03d.h5'
        return fmt % args

    def model(self, vocab_size, batch_size, stateful):
        inp = Input(
            shape = (None,),
            batch_size = batch_size,
            dtype = tf.int32)
        embedding = Embedding(
            input_dim = vocab_size,
            output_dim = self.emb_size)
        lstm1 = LSTM(
            self.lstm1_units,
            stateful = stateful,
            return_sequences = True,
            dropout = self.dropout,
            recurrent_dropout = self.rec_dropout)
        lstm2 = LSTM(
            self.lstm2_units,
            stateful = stateful,
            return_sequences = True,
            dropout = self.dropout,
            recurrent_dropout = self.rec_dropout)
        time_dist = TimeDistributed(Dense(vocab_size))
        out = time_dist(lstm2(lstm1(embedding(inp))))
        return MyModel(inputs = [inp], outputs = [out])

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

def compute_and_apply_gradients(model, x, y):
    with tf.GradientTape() as tape:
        y_hat = model(x, training = True)
        loss = model.compiled_loss(y, y_hat,
                                   regularization_losses = model.losses)
    vars = model.trainable_variables
    grads = tape.gradient(loss, vars)
    grads, _ = tf.clip_by_global_norm(grads, 15)
    model.optimizer.apply_gradients(zip(grads, vars))
    return y_hat

class MyModel(Model):
    def train_step(self, data):
        x, y = data
        y_hat = compute_and_apply_gradients(self, x, y)
        self.compiled_metrics.update_state(y, y_hat)
        return {m.name: m.result() for m in self.metrics}

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

def generate_music(temps, top_ps, data, path, params):
    n_temps = len(temps)
    n_top_ps = len(top_ps)
    n_preds = n_temps + n_top_ps

    # Often more than one full song - not great.
    n_samples = 1200
    n_seed = 256

    SP.header('%d PREDICTIONS' % n_preds)
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
    seqs = generate_sequences(model, temps, top_ps, seed, n_samples)

    # Add the original
    seqs.append(seq[idx + n_seed:idx + n_seed + n_samples])
    seqs = np.array(seqs)
    seed = np.vstack((seed, seed[0]))

    # Cut half the seed because it is too long to listen to it all.
    seed = seed[:,n_seed // 2:]

    join = np.repeat(np.expand_dims(long_pause, 0), len(seqs), axis = 0)
    print(seed.shape, join.shape, seqs.shape)
    seqs = np.hstack((seed, join, seqs))

    for i in range(n_temps):
        file_name = '%s-t%.2f.mp3' % (data.code_type, temps[i])
        file_path = path / file_name
        data.code_to_audio_file(seqs[i], file_path)

    for i in range(n_top_ps):
        file_name = '%s-p%.2f.mp3' % (data.code_type, top_ps[i])
        file_path = path / file_name
        data.code_to_audio_file(seqs[n_temps + i], file_path)

    file_path = path / ('%s-orig.mp3' % data.code_type)
    data.code_to_audio_file(seqs[-1], file_path)
    SP.leave()

def train_model(train, valid, path, params):
    vocab_size = len(train.encoder.ix2ch)
    strategy = select_strategy()
    with strategy.scope():

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
    args = docopt(__doc__, version = 'Train LSTM 1.0')
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
        temps = [0.8, 0.9, 1.0, 1.05, 1.15]
        top_ps = [0.75, 0.85, 0.9, 0.94, 0.98]
        generate_music(temps, top_ps, valid, path, params)
    else:
        train_model(train, valid, path, params)

if __name__ == '__main__':
    main()
