########################################################################
# === RESULTS ===
#
# ED LSTM1 LSTM2  DO  LR    N-PARAMS BS   STRIDE N  VAL-LOSS
# 64   256   256  .2  0.01           1024     -1 20  1.77803
# 64   128   128  .2  0.01   239 278 1024     -1 20  1.72698
# 32   128   128  .2  0.01   221 422 1024     -1 40  1.71708
# 32   128   128  .2  0.01   221 422  512     -1 40  1.63385
# 32   128   128  .2  0.01   221 422  512     -4 50  1.62806+
# 16   128   128  .2  0.01   221 422  512     -1 50  1.63923
# 16   128   128  .2  0.005  221 422  512     -1 60  1.68280
# 16   128   128  .2  0.02   221 422  512     -1 60  1.67745
#
# *** MORE DATA ***
# ED LSTM1 LSTM2  DO  LR    N-PARAMS  BS  STRIDE  N  VAL-LOSS
# 16   128   128  .2  0.01    212494  512     -1  60  1.59445
# 16   128   128  .2  0.01    212494  256     -1  60  1.57344
# 16   128   128  .2  0.01    212494  256     -1  80  1.59031
# 16   128   128  .2  0.01    212494  128     -1  80  1.65849
# 16   128   128  .2  0.005   212494  128     -1  80  1.53826
#  8   128   128  .2  0.005   208030  128     -1 100  1.60921
# 32   128   128  .2  0.005   221422  128     -1 100  1.52926
# 32   128   128  .3  0.005   221422  128     -1 100  1.54948
# 32   128   128  .1  0.005   221422  128     -1 100  1.52442+
# 32   128   128  .0  0.005   221422  128     -1 120  1.60271 (1.4057)
#
# *** EVEN MORE DATA *** (45 vocab)
# ED LSTM1 LSTM2  DO  LR    N-PARAMS  BS  STRIDE  N  VAL-LOSS
# 32   128   128  .1  0.005   221261 128     -1 120   1.52852 (1.4217)
# 32   128   128  .1  0.005   221261 128     -1 120   1.51869 (1.4141)

# Hyperparameters here
ROOT_PATH = '/content/drive/My Drive/musicgen'
BATCH_SIZE = 128
EPOCHS = 120
LEARNING_RATE = 0.005
EMBEDDING_DIM = 32
LSTM1_UNITS = 128
LSTM2_UNITS = 128
DROPOUT = 0.1
SEQ_LEN = 128

from pathlib import Path
from sys import path
path.append(ROOT_PATH)

from logging import ERROR
from musicgen.pcode import INSN_SILENCE, load_data, pcode_to_midi_file
from musicgen.utils import SP, file_name_for_params
from os import environ, listdir

from random import randrange, shuffle
from tensorflow import constant, get_logger, int32
from tensorflow.config import (experimental_connect_to_cluster,
                               list_logical_devices)
from tensorflow.data import Dataset
from tensorflow.distribute.cluster_resolver import TPUClusterResolver
from tensorflow.distribute.experimental import TPUStrategy
from tensorflow.keras import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.tpu.experimental import initialize_tpu_system
import numpy as np

# get_logger().setLevel(ERROR)
# environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def lstm_model(seq_len, vocab_size, batch_size, stateful):
    source = Input(
        name = 'seed',
        shape = (seq_len,),
        batch_size = batch_size,
        dtype = int32)
    embedding = Embedding(input_dim = vocab_size,
                          output_dim = EMBEDDING_DIM)(source)
    lstm_1 = LSTM(
        LSTM1_UNITS,
        stateful = stateful,
        return_sequences = True,
        dropout = DROPOUT)(embedding)
    lstm_2 = LSTM(
        LSTM2_UNITS,
        stateful = stateful,
        return_sequences = True,
        dropout = DROPOUT)(lstm_1)
    predicted_char = TimeDistributed(
        Dense(vocab_size, activation = 'softmax'))(lstm_2)
    return Model(inputs = [source], outputs = [predicted_char])

def initialize_tpus():
    tpu = 'grpc://' + environ['COLAB_TPU_ADDR']
    resolver = TPUClusterResolver(tpu)
    experimental_connect_to_cluster(resolver)
    initialize_tpu_system(resolver)
    devs = list_logical_devices('TPU')
    assert len(devs) > 0
    SP.print('Our devices %s', devs)
    return TPUStrategy(resolver)

def create_tf_dataset(dataset, seq_len, batch_size, stride):
    def split_input_target(chunk):
        return chunk[:-1], chunk[1:]
    def flatten_window(win):
        return win.batch(seq_len + 1, drop_remainder = True)
    SP.print('Length %d, seq_len %d, batch_size %d.'
             % (len(dataset), seq_len, batch_size))
    source = constant(dataset, dtype = int32)
    return Dataset    \
        .from_tensor_slices(source) \
        .window(seq_len + 1, stride, drop_remainder = True) \
        .flat_map(flatten_window) \
        .map(split_input_target) \
        .shuffle(10000) \
        .batch(batch_size, drop_remainder = True)

def do_train(train, validate, seq_len, vocab_size, batch_size):
    # Must be done before creating the datasets.
    strategy = initialize_tpus()

    # Reshape the raw data into tensorflow Datasets
    ds_train = create_tf_dataset(train, seq_len, batch_size,
                                 seq_len - 1)
    ds_validate = create_tf_dataset(validate, seq_len, batch_size,
                                    seq_len - 1)

    with strategy.scope():
        model = lstm_model(seq_len, vocab_size, None, False)
        opt1 = RMSprop(learning_rate = LEARNING_RATE)
        model.compile(
            optimizer = opt1,
            loss = 'sparse_categorical_crossentropy',
            metrics = ['sparse_categorical_accuracy'])
        SP.print(model.summary())

    cb_best = ModelCheckpoint(
        str(Path(ROOT_PATH) / 'best.h5'),
        monitor = 'val_loss',
        verbose = 1,
        save_best_only = True,
        mode = 'min')

    model.fit(x = ds_train,
              validation_data = ds_validate,
              epochs = EPOCHS,
              callbacks = [cb_best],
              verbose = 2)
    model.save_weights(
        str(Path(ROOT_PATH) / 'tpu-weights.h5'),
        overwrite = True)

def generate_sequences(model, dataset, temperatures, seed, length):
    SP.header('TEMPERATURES %s' % temperatures)
    batch_size = len(temperatures)

    for i in range(seed.shape[1] - 1):
        model.predict(seed[:, i:i + 1])
    SP.print('Consumed seed %s.' % (seed.shape,))

    preds = [seed[:, -1:]]
    for _ in range(length):
        last_word = preds[-1]
        P = model.predict(last_word)[:, 0, :]

        next_idx = [np.random.choice(P.shape[1], p = P[i])
                    for i in range(batch_size)]
        preds.append(np.asarray(next_idx, dtype = np.int32))

    seqs = []
    for i in range(batch_size):
        seq = [int(preds[j][i]) for j in range(length)]
        seqs.append(seq)
    SP.leave()
    return seqs

def do_predict(test, ix2ch, ch2ix, temperatures, seq_len):
    batch_size = len(temperatures)
    model = lstm_model(1, len(ix2ch), batch_size, True)
    model.load_weights(str(Path(ROOT_PATH) / 'best.h5'))
    model.reset_states()

    idx = randrange(len(test) - seq_len)
    seed = np.array(test[idx:idx + seq_len])

    seed = np.repeat(np.expand_dims(seed, 0), batch_size, axis = 0)
    seqs = generate_sequences(model, test, temperatures, seed, 500)
    # Two bars of silence
    join = np.array([ch2ix[(INSN_SILENCE, 16)]] * 2)
    join = np.repeat(np.expand_dims(join, 0), batch_size, axis = 0)
    seqs = np.hstack((seed, join, seqs))

    seqs = [[ix2ch[ix] for ix in seq] for seq in seqs]
    for i, seq in enumerate(seqs):
        file_path = Path(ROOT_PATH) / ('tpu-test-%02d.mid' % i)
        pcode_to_midi_file(seq, file_path, False)

def main():
    SP.enabled = True
    ix2ch, ch2ix, seq = load_data(Path(ROOT_PATH), 150, False)

    # Convert to integer sequence
    n_seq = len(seq)
    vocab_size = len(ix2ch)

    # Split data
    n_train = int(n_seq * 0.8)
    n_validate = int(n_seq * 0.1)
    n_test = n_seq - n_train - n_validate
    train = seq[:n_train]
    validate = seq[n_train:n_train + n_validate]
    test = seq[n_train + n_validate:]
    fmt = '%d, %d, and %d tokens in train, validate, and test sequences.'
    SP.print(fmt % (n_train, n_validate, n_test))

    # Run training and prediction.
    do_train(train, validate, SEQ_LEN, vocab_size, BATCH_SIZE)
    do_predict(test, ix2ch, ch2ix, [0.5, 0.8, 1.0, 1.2, 1.5], SEQ_LEN)

if __name__ == '__main__':
    main()
