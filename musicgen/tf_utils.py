# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
from musicgen.utils import SP
from os import environ
from tensorflow import constant, get_logger, int32
from tensorflow.data import Dataset
from tensorflow.config import (experimental_connect_to_cluster,
                               list_logical_devices)
from tensorflow.distribute.cluster_resolver import TPUClusterResolver
from tensorflow.distribute.experimental import TPUStrategy
from tensorflow.keras.utils import Sequence
from tensorflow.tpu.experimental import initialize_tpu_system
import numpy as np

def initialize_tpus():
    tpu_addr = environ.get('COLAB_TPU_ADDR')
    if not tpu_addr:
        SP.print('TPU not configured.')
        return None
    SP.print('Connecting to TPU at %s.' % tpu_addr)
    resolver = TPUClusterResolver('grpc://' + tpu_addr)
    experimental_connect_to_cluster(resolver)
    initialize_tpu_system(resolver)
    devs = list_logical_devices('TPU')
    assert len(devs) > 0
    SP.header('%d TPU DEVICES' % len(devs))
    for dev in devs:
        SP.print(dev)
    SP.leave()
    strategy = TPUStrategy(resolver)
    return strategy

def sequence_to_batched_dataset(seq, seq_len, batch_size):
    stride = seq_len - 1
    def split_input_target(chunk):
        return chunk[:-1], chunk[1:]
    def flatten_window(win):
        return win.batch(seq_len + 1, drop_remainder = True)
    SP.print('Length %d, seq_len %d, batch_size %d.'
             % (len(seq), seq_len, batch_size))
    source = constant(seq, dtype = int32)
    return Dataset    \
        .from_tensor_slices(source) \
        .window(seq_len + 1, stride, drop_remainder = True) \
        .flat_map(flatten_window) \
        .map(split_input_target) \
        .shuffle(10000) \
        .batch(batch_size, drop_remainder = True)
