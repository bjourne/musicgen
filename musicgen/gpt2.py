# Copyright (C) 2021 Björn Lindqvist <bjourne@gmail.com>
#
# GPT-2 model. Code adapted from Huggingfaces.
from musicgen.utils import SP
from tensorflow.keras import Model
from tensorflow.keras.activations import gelu
from tensorflow.keras.initializers import *
from tensorflow.keras.layers import *

import tensorflow as tf

# GPT2 hyper params. Parameters are passed by setting these globals.
# HIDDEN_SIZE = 768
HIDDEN_SIZE = None
INITIALIZER_RANGE = 0.02
N_POSITIONS = 1024
N_LAYER = 12
L_NORM_EPS = 1e-5
N_HEAD = 12
VOCAB_SIZE = None

# Dropout rates
EMBD_PDROP = 0.15
ATTN_PDROP = 0.15
RESID_PDROP = 0.15

def casual_attn_mask(nd, ns, dtype):
    i = tf.range(nd)[:, None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)

class Conv1D(Layer):
    def __init__(self, nf, nx, **kwargs):
        super().__init__(**kwargs)
        self.nf = nf
        self.nx = nx

    def build(self, input_shape):
        self.weight = self.add_weight(
            "weight",
            shape=[self.nx, self.nf],
            initializer = TruncatedNormal(stddev = INITIALIZER_RANGE)
        )
        self.bias = self.add_weight(
            "bias",
            shape=[1, self.nf],
            initializer=tf.zeros_initializer()
        )

    def call(self, x):
        bz, sl = x.shape[:2]
        x = tf.reshape(x, [-1, self.nx])
        x = tf.matmul(x, self.weight) + self.bias
        x = tf.reshape(x, [bz, sl, self.nf])
        return x

class Attn(Layer):
    def __init__(self):
        super().__init__()
        self.c_attn = Conv1D(3 * HIDDEN_SIZE, HIDDEN_SIZE,
                             name = 'c_attn')
        self.c_proj = Conv1D(HIDDEN_SIZE, HIDDEN_SIZE,
                             name = 'c_proj')
        self.attn_dropout = Dropout(ATTN_PDROP)
        self.resid_dropout = Dropout(RESID_PDROP)

    def split_heads(self, x):
        new_x_shape = x.shape[:-1] + [N_HEAD, x.shape[-1] // N_HEAD]
        x = tf.reshape(x, new_x_shape)
        return tf.transpose(x, (0, 2, 1, 3))

    def merge_heads(self, x):
        x = tf.transpose(x, (0, 2, 1, 3))
        new_x_shape = x.shape[:-2] + [x.shape[-2] * x.shape[-1]]
        return tf.reshape(x, new_x_shape)

    def attn(self, q, k, v, training):
        w = tf.matmul(q, k, transpose_b = True)

        # Always scale
        dk = tf.cast(k.shape[-1], dtype = w.dtype)
        w = w / tf.math.sqrt(dk)

        _, _, nd, ns = w.shape
        b = casual_attn_mask(nd, ns, dtype = w.dtype)
        b = tf.reshape(b, (1, 1, nd, ns))
        w = w * b - 1e4 * (1 - b)

        w = tf.nn.softmax(w, axis = -1)
        # Dropout follows softmax?
        w = self.attn_dropout(w, training = training)

        return tf.matmul(w, v)

    def call(self, x, training):
        x = self.c_attn(x)
        q, k, v = tf.split(x, 3, axis = 2)
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        a = self.attn(q, k, v, training)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a, training = training)
        return a

class Block(Layer):
    def __init__(self):
        super().__init__()
        self.ln_1 = LayerNormalization(epsilon = L_NORM_EPS,
                                       name = 'ln_1')
        self.attn = Attn()
        self.ln_2 = LayerNormalization(epsilon = L_NORM_EPS,
                                       name = 'ln_2')

        n_state = 4 * HIDDEN_SIZE
        self.c_fc = Conv1D(n_state, HIDDEN_SIZE, name = 'c_fc')
        self.c_proj = Conv1D(HIDDEN_SIZE, n_state, name = 'c_proj')
        self.dropout = Dropout(RESID_PDROP)

    def call(self, x, training):
        a = self.ln_1(x)
        a = self.attn(a, training)
        x = x + a

        m = self.ln_2(x)
        m = gelu(self.c_fc(m))
        m = self.c_proj(m)
        m = self.dropout(m, training = training)

        x = x + m
        return x

class SharedEmbeddings(Layer):
    def __init__(self):
        super().__init__(name = 'wte')

    def build(self, input_shape):
        initializer = TruncatedNormal(stddev = INITIALIZER_RANGE)
        self.weight = self.add_weight(
            "weight",
            shape = [VOCAB_SIZE, HIDDEN_SIZE],
            initializer = initializer
        )
        super().build(input_shape)

    def call(self, inputs, mode):
        if mode == 'embedding':
            return tf.gather(self.weight, tf.cast(inputs, tf.int32))
        elif mode == 'linear':
            first_dims = inputs.shape[:-1]
            x = tf.reshape(inputs, [-1, HIDDEN_SIZE])
            logits = tf.matmul(x, self.weight, transpose_b=True)
            return tf.reshape(logits, first_dims + [VOCAB_SIZE])
        else:
            assert False

def print_hyper_params():
    SP.header('INITIALIZING GPT2 MODEL')
    SP.print('Vocab size : %5d' % VOCAB_SIZE)
    SP.print('Hidden size: %5d' % HIDDEN_SIZE)

class GPT2(Model):
    def __init__(self):
        super(GPT2, self).__init__()
        self.wte = SharedEmbeddings()
        initializer = TruncatedNormal(stddev = INITIALIZER_RANGE)
        self.wpe = Embedding(
            N_POSITIONS,
            HIDDEN_SIZE,
            embeddings_initializer = initializer,
            name = 'wpe')
        self.h = [Block() for _ in range(N_LAYER)]
        self.ln_f = LayerNormalization(epsilon = L_NORM_EPS)
        self.drop = Dropout(EMBD_PDROP)
        print_hyper_params()

    def call(self, inputs, training = False):
        input_shape = inputs.shape
        seq_len = input_shape[-1]
        inputs = tf.reshape(inputs, [-1, seq_len])

        position_ids = tf.range(0, seq_len,
                                dtype = tf.int32)[tf.newaxis, :]

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        position_ids = tf.reshape(position_ids, [-1, seq_len])

        inputs_embeds = self.wte(inputs, 'embedding')
        position_embeds = self.wpe(position_ids)

        hs = inputs_embeds + position_embeds
        hs = self.drop(hs, training = training)
        output_shape = input_shape + [hs.shape[-1]]

        for block in self.h:
            hs = block(hs, training)
        hs = self.ln_f(hs)
        hs = tf.reshape(hs, output_shape)
        return self.wte(hs, 'linear')
