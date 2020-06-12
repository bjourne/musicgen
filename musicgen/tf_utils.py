# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
from musicgen.utils import SP
from os import environ
from tensorflow.data import Dataset
from tensorflow.config import (experimental_connect_to_cluster,
                               list_logical_devices)
from tensorflow.distribute.cluster_resolver import TPUClusterResolver
from tensorflow.distribute.experimental import TPUStrategy
from tensorflow.keras.utils import Sequence
from tensorflow.tpu.experimental import initialize_tpu_system
import numpy as np
import tensorflow as tf

def initialize_tpus():
    SP.header('INITIALIZING TPUS')
    tpu_addr = environ.get('COLAB_TPU_ADDR')
    if not tpu_addr:
        SP.print('TPU not found.')
        SP.leave()
        return None
    SP.print('TPU address: %s' % tpu_addr)
    resolver = TPUClusterResolver('grpc://' + tpu_addr)
    experimental_connect_to_cluster(resolver)
    initialize_tpu_system(resolver)
    devs = list_logical_devices('TPU')
    assert len(devs) > 0
    SP.header('%d TPUS' % len(devs))
    for dev in devs:
        SP.print(dev)
    SP.leave()
    strategy = TPUStrategy(resolver)
    SP.print('%d synced replicas.' % strategy.num_replicas_in_sync)
    SP.leave()
    return strategy

def sequence_to_batched_dataset(seq, seq_len, batch_size):
    stride = seq_len - 1
    def split_input_target(chunk):
        return chunk[:-1], chunk[1:]
    def flatten_window(win):
        return win.batch(seq_len + 1, drop_remainder = True)
    SP.print('Length %d, seq_len %d, batch_size %d.'
             % (len(seq), seq_len, batch_size))
    source = tf.constant(seq, dtype = tf.int32)
    return Dataset    \
        .from_tensor_slices(source) \
        .window(seq_len + 1, stride, drop_remainder = True) \
        .flat_map(flatten_window) \
        .map(split_input_target) \
        .shuffle(10000) \
        .batch(batch_size, drop_remainder = True)

def generate_sequences(model, temps, seed, length,
                       excluded):
    SP.header('TEMPERATURES %s' % temps)
    batch_size = len(temps)

    # Make temps into a row vector
    temps = np.array(temps)[:,None]

    # Priming the model
    for i in range(seed.shape[1] - 1):
        model.predict(seed[:, i:i + 1])

    preds = [seed[:, -1:]]
    eps = np.finfo('float').eps
    for _ in range(length):
        last_word = preds[-1]
        Ps = model.predict(last_word)[:, 0, :]

        # Assign a very low probability to tokens to be avoided.
        Ps[:,excluded] = eps

        # Weigh probs according to temps
        Ps = np.exp(np.log(Ps) / temps)

        # Normalize
        Ps = (Ps.T / Ps.sum(axis = 1)).T

        next_idx = [np.random.choice(len(P), p = P) for P in Ps]
        preds.append(np.asarray(next_idx, dtype = np.int32))

    SP.leave()
    return [[int(preds[j][i]) for j in range(length)]
            for i in range(batch_size)]

if __name__ == '__main__':
    a = np.arange(100)
    for e in sequence_to_batched_dataset(a, 5, 2):
        print(e)
