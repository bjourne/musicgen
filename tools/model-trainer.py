# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
"""Train ANY of the models.

Usage:
    model-trainer.py [options] ( train | generate )
        <code-type> <corpus-path>
        --emb-size=<i> --batch-size=<i> --dropout=<f> --rec-dropout=<f>
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
from musicgen.corpus import load_index
from musicgen.pcode import (mod_file_to_pcode,
                            pcode_to_midi_file,
                            pcode_short_pause,
                            pcode_long_pause)
from musicgen.utils import (SP, file_name_for_params,
                            find_subseq, load_pickle_cache)
from musicgen.tf_utils import select_strategy, sequence_to_samples
from pathlib import Path
from random import randrange, shuffle
from tensorflow.data import Dataset
from tensorflow.keras import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import *
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import *
from tensorflow.nn import softmax
import numpy as np
import tensorflow as tf

CodeInfo = namedtuple('CodeInfo', ['to_code_fn', 'to_midi_fn',
                                    'short_pause', 'long_pause'])

CODE_TYPES = {
    'pcode_abs' : CodeInfo(lambda m: mod_file_to_pcode(m, False),
                           lambda c, fp: pcode_to_midi_file(c, fp, False),
                           pcode_short_pause(),
                           pcode_long_pause())
}

def encode_code(code, ch2ix, ix2ch, ix):
    for ch in set(code):
        if ch not in ch2ix:
            ch2ix[ch] = ix
            ix2ch[ix] = ch
            ix += 1
    barr = np.array([ch2ix[ch] for ch in code], dtype = np.uint8)
    return ix, barr

def mod_file_to_code_w_progress(i, n, mod_file, to_code_fn):
    SP.header('[ %4d / %4d ] PARSING %s' % (i, n, mod_file))
    code = list(to_code_fn(mod_file))
    SP.leave()
    return code

def build_cache(path, mods, to_code_fn):
    mod_files = [path / mod.genre / mod.fname for mod in mods]
    n = len(mod_files)
    ch2ix, ix2ch = {}, {}
    ix = 0
    arrs = []
    for i, p, in enumerate(sorted(mod_files)):
        code = mod_file_to_code_w_progress(i + 1, n, p, to_code_fn)
        ix, arr = encode_code(code, ch2ix, ix2ch, ix)
        arrs.append((p.name, arr))
    shuffle(arrs)
    return ix2ch, ch2ix, arrs

# def sequence_to_samples(seq, length):
#     stride = length - 1
#     def split_input_target(chunk):
#         return chunk[:-1], chunk[1:]
#     def flatten_window(win):
#         return win.batch(length + 1, drop_remainder = True)
#     source = tf.constant(seq, dtype = tf.int32)
#     return Dataset    \
#         .from_tensor_slices(source) \
#         .window(length + 1, stride, drop_remainder = True) \
#         .flat_map(flatten_window) \
#         .map(split_input_target) \
#         .shuffle(10000)

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
        def rebuild_fn():
            return build_cache(path, mods, self.info.to_code_fn)
        self.ix2ch, self.ch2ix, self.arrs =  \
            load_pickle_cache(cache_path, rebuild_fn)

    def load_mod_file(self, p):
        self.ch2ix, self.ix2ch = {}, {}
        code = mod_file_to_code_w_progress(1, 1, p, self.info.to_code_fn)
        _, arr = encode_code(code, self.ch2ix, self.ix2ch, 0)
        self.arrs = [(p.name, arr)]

    def print_historgram(self):
        codes = [arr for (name, arr) in self.arrs]
        seq = np.concatenate(codes)
        ix_counts = Counter(seq)
        ch_counts = {self.ix2ch[ix] : cnt
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
            td.ch2ix = self.ch2ix
            td.ix2ch = self.ix2ch
        return tds

    def code_to_midi_file(self, seq, file_path):
        code = [self.ix2ch[ix] for ix in seq]
        self.info.to_midi_fn(code, file_path)

    def flatten(self):
        long_pause = np.array([self.ch2ix[ch]
                               for ch in self.info.long_pause])
        codes = [code for (_, code) in self.arrs]
        padded_codes = []
        for c in codes:
            padded_codes.append(c)
            padded_codes.append(long_pause)
        return np.concatenate(padded_codes)

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
    grads, _ = tf.clip_by_global_norm(grads, 5)
    model.optimizer.apply_gradients(zip(grads, vars))
    return y_hat

class MyModel(Model):
    def train_step(self, data):
        x, y = data
        y_hat = compute_and_apply_gradients(self, x, y)
        self.compiled_metrics.update_state(y, y_hat)
        return {m.name: m.result() for m in self.metrics}

def create_model(vocab_size, emb_size, batch_size,
                 dropout, rec_dropout,
                 lstm1_units, lstm2_units,
                 stateful):
    inp = Input(
        shape = (None,),
        batch_size = batch_size,
        dtype = tf.int32)
    embedding = Embedding(
        input_dim = vocab_size,
        output_dim = emb_size)
    lstm1 = LSTM(
        lstm1_units,
        stateful = stateful,
        return_sequences = True,
        dropout = dropout,
        recurrent_dropout = rec_dropout)
    lstm2 = LSTM(
        lstm2_units,
        stateful = stateful,
        return_sequences = True,
        dropout = dropout,
        recurrent_dropout = rec_dropout)
    time_dist = TimeDistributed(
        Dense(vocab_size))
    out = time_dist(lstm2(lstm1(embedding(inp))))
    return MyModel(inputs = [inp], outputs = [out])

def get_weights_file(code_type, epochs,
                     emb_size, batch_size,
                     dropout, rec_dropout, lstm1_units, lstm2_units,
                     lr, seq_len):
    args = (code_type, epochs,
            emb_size, batch_size,
            dropout, rec_dropout, lstm1_units, lstm2_units,
            lr, seq_len)
    fmt = 'weights_%s-%03d-%03d-%03d-%.2f-%.2f-%03d-%03d-%.5f-%03d.h5'
    return fmt % args

def generate_sequences(model, temps, seed, length):
    batch_size = len(temps)

    # Make temps into a row vector
    temps = np.array(temps)[:,None]

    # Priming the model
    for i in range(seed.shape[1] - 1):
        model.predict(seed[:, i:i + 1])

    preds = [seed[:, -1:]]
    for _ in range(length):
        last_word = preds[-1]
        Ps = model.predict(last_word)[:, 0, :]
        Ps = softmax(Ps).numpy()

        # Weigh probs according to temps
        Ps = np.exp(np.log(Ps) / temps)

        # Normalize
        Ps = (Ps.T / Ps.sum(axis = 1)).T

        next_ixs = []
        for P in Ps:
            # Magic nucleus sampling.
            ixs = np.argsort(-P)
            PC = np.cumsum(P[ixs])
            top_n = len(PC[PC <= 0.9]) + 1

            # Clear the prob of those who didn't make it.
            P[ixs[top_n:]] = 0.0

            # Rescale.
            P = P / P.sum()

            next_ixs.append(np.random.choice(len(P), p = P))

        preds.append(np.asarray(next_ixs, dtype = np.int32))

    SP.leave()
    return [[int(preds[j][i]) for j in range(length)]
            for i in range(batch_size)]

def generate_music(temps, data, path, weights_path,
                   emb_size,
                   dropout, rec_dropout,
                   lstm1_units, lstm2_units):

    batch_size = len(temps)
    SP.header('%d PREDICTIONS' % batch_size)
    model = create_model(len(data.ix2ch), emb_size, batch_size,
                         dropout, rec_dropout,
                         lstm1_units, lstm2_units,
                         True)
    model.load_weights(str(weights_path))
    model.reset_states()

    seq = data.flatten()
    seed_len = 128
    while True:
        idx = randrange(len(seq) - seed_len)
        seed = seq[idx:idx + seed_len]
        if not list(find_subseq(seed.tolist(), data.info.long_pause)):
            break
        SP.print('Pause in seed, regenerating.')
    SP.print('Seed index %d.' % idx)

    seed = np.repeat(np.expand_dims(seed, 0), batch_size, axis = 0)
    seqs = generate_sequences(model, temps, seed, 600)

    join = np.array([data.ch2ix[ch] for ch in data.info.short_pause])
    join = np.repeat(np.expand_dims(join, 0), batch_size, axis = 0)
    seqs = np.hstack((seed, join, seqs))

    fmt = '%s_out-%.2f.mid'
    for temp, seq in zip(temps, seqs):
        file_name = fmt % (data.code_type, temp)
        file_path = path / file_name
        data.code_to_midi_file(seq, file_path)
    SP.leave()

def train_model(weights_path, epochs,
                train, valid,
                emb_size, batch_size,
                dropout, rec_dropout,
                lstm1_units, lstm2_units, lr, seq_len):
    strategy = select_strategy()
    with strategy.scope():
        vocab_size = len(train.ix2ch)
        model = create_model(vocab_size, emb_size, None,
                             dropout, rec_dropout,
                             lstm1_units, lstm2_units, False)
        optimizer = RMSprop(learning_rate = lr)
        loss_fn = SparseCategoricalCrossentropy(from_logits = True)
        model.compile(
            optimizer = optimizer,
            loss = loss_fn,
            metrics = ['sparse_categorical_accuracy'])
    model.summary()

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
    reduce_lr = ReduceLROnPlateau('val_loss', 0.1, 8, 0.00005)
    callbacks = [cb_best, reduce_lr]
    SP.print('Batching samples...')
    train_ds = train.to_samples(seq_len) \
        .batch(batch_size, drop_remainder = True)
    valid_ds = valid.to_samples(seq_len) \
        .batch(batch_size, drop_remainder = True)

    SP.print('Counting samples...')
    n_train = sum(1 for _ in train_ds)
    n_valid = sum(1 for _ in valid_ds)
    fmt = '%d training and %d validation batches.'
    SP.print(fmt % (n_train, n_valid))

    model.fit(x = train_ds,
              validation_data = valid_ds,
              epochs = epochs, callbacks = callbacks,
              verbose = 1)

def main():
    # Prologue
    args = docopt(__doc__, version = 'Train LSTM 1.0')
    SP.enabled = args['--verbose']
    path = Path(args['<corpus-path>'])
    code_type = args['<code-type>']
    np.set_printoptions(linewidth = 160)
    is_generate = args['generate']

    # Hyperparameters
    dropout = float(args['--dropout'])
    rec_dropout = float(args['--rec-dropout'])
    emb_size = int(args['--emb-size'])
    batch_size = int(args['--batch-size'])
    lstm1_units = int(args['--lstm1-units'])
    lstm2_units = int(args['--lstm2-units'])
    lr = float(args['--lr'])
    seq_len = int(args['--seq-len'])
    epochs = int(args['--epochs'])

    # Load data and split it
    data = TrainingData(code_type)
    if path.is_dir():
        data.load_disk_cache(path, 150)
        train, valid, test = data.split_3way(0.8, 0.1)
    else:
        data.load_mod_file(path)
        train = valid = test = data
    data.print_historgram()

    args = len(train.arrs), len(valid.arrs), len(test.arrs)
    SP.print('Train/valid/test split %d/%d/%d' % args)

    weights_file = get_weights_file(code_type, epochs,
                                    emb_size, batch_size,
                                    dropout, rec_dropout,
                                    lstm1_units, lstm2_units,
                                    lr, seq_len)
    weights_path = path / weights_file
    if is_generate:
        temps = [0.5, 0.8, 1.0, 1.2, 1.5]
        generate_music(temps, test, path, weights_path,
                       emb_size,
                       dropout, rec_dropout,
                       lstm1_units, lstm2_units)
    else:
        train_model(weights_path, epochs,
                    train, valid,
                    emb_size, batch_size,
                    dropout, rec_dropout,
                    lstm1_units, lstm2_units, lr, seq_len)

if __name__ == '__main__':
    main()
