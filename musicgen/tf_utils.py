# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
from musicgen.utils import SP
from os import environ
from tensorflow.data import Dataset
from tensorflow.config import (experimental_connect_to_cluster,
                               list_logical_devices,
                               list_physical_devices)
from tensorflow.distribute import OneDeviceStrategy
from tensorflow.distribute.cluster_resolver import TPUClusterResolver
from tensorflow.distribute.experimental import TPUStrategy
from tensorflow.keras.utils import Sequence
from tensorflow.tpu.experimental import initialize_tpu_system
import numpy as np
import tensorflow as tf

def select_strategy():
    SP.header('SELECTING STRATEGY')
    gpus = list_physical_devices('GPU')
    SP.header('%d GPU(s)' % len(gpus))
    for gpu in gpus:
        SP.print(gpu)
    SP.leave()
    tpu_addr = environ.get('COLAB_TPU_ADDR')
    if not tpu_addr:
        dev = '/GPU:0' if gpus else '/CPU:0'
        SP.print('No TPU, using %s instead.' % dev)
        SP.leave()
        return OneDeviceStrategy(device = dev)
    SP.print('TPU address: %s' % tpu_addr)
    resolver = TPUClusterResolver('grpc://' + tpu_addr)
    experimental_connect_to_cluster(resolver)
    initialize_tpu_system(resolver)
    tpus = list_logical_devices('TPU')
    SP.header('%d TPU(s)' % len(tpus))
    for tpu in tpus:
        SP.print(tpu)
    SP.leave()
    strategy = TPUStrategy(resolver)
    SP.print('%d synced replicas.' % strategy.num_replicas_in_sync)
    SP.leave()
    return strategy

def sequence_to_samples(seq, length):
    stride = length - 1
    def split_input_target(chunk):
        return chunk[:-1], chunk[1:]
    def flatten_window(win):
        return win.batch(length + 1, drop_remainder = True)
    source = tf.constant(seq, dtype = tf.int32)
    return Dataset    \
        .from_tensor_slices(source) \
        .window(length + 1, stride, drop_remainder = True) \
        .flat_map(flatten_window) \
        .map(split_input_target) \
        .shuffle(10000)

def generate_sequences(model, temps, seed, length, excluded):
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
