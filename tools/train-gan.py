# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
#
# Good range: 20 - 80
"""MOD pattern generator

Usage:
    train-gan.py [-v] [--kb-limit=<int>] <corpus>

Options:
    -h --help                   show this screen
    -v --verbose                print more output
    --kb-limit=<int>            kb limit [default: 150]
"""
from docopt import docopt
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape
from keras.layers import LeakyReLU
from keras.layers import Dropout
from mido import Message, MidiFile, MidiTrack
from musicgen.analyze import sample_props
from musicgen.corpus import load_index
from musicgen.defs import period_to_idx
from musicgen.parser import PowerPackerModule, load_file
from musicgen.rows import linearize_rows, rows_to_mod_notes
from musicgen.utils import SP, flatten
from pathlib import Path
from numpy.random import randint, randn
import numpy as np
import matplotlib.pyplot as plt

########################################################################
# Data conversion
########################################################################
def pattern_to_matrix(pattern, percussive):
    mat = np.zeros((4, 64), dtype = int)
    for i, row in enumerate(pattern.rows):
        current_sample = None
        for j, cell in enumerate(row):
            if cell.sample_idx:
                current_sample = cell.sample_idx
            if cell.period and current_sample not in percussive:
                mat[j, i] = period_to_idx(cell.period)

    neg_counts = -np.count_nonzero(mat, axis = 1)
    order = np.argsort(neg_counts)
    mat = mat[order]

    min_note = np.min(mat[mat > 0], initial = 100)
    mat[mat > 0] -= (min_note - 1)
    return mat.T

MIDI_BASE = 48
ROW_TIME = 125

def stop_note(pitch, time):
    return Message('note_off', note = pitch,
                   velocity = 0, time = time * ROW_TIME)

def start_note(pitch, time):
    return Message('note_on', note = pitch,
                   velocity = 127, time = time * ROW_TIME)

def column_to_track(col):
    notes = []
    for i, pitch in enumerate(col):
        if pitch > 0:
            notes.append((i, pitch))
    dur_notes = [(i1, p1, min(i2 - i1, 4))
                 for (i1, p1), (i2, p2)
                 in zip(notes, notes[1:] + [(len(col) * 2, 0)])]
    prev = 0
    for ofs, pitch, dur in dur_notes:
        rel_ofs = ofs - prev
        yield start_note(pitch + MIDI_BASE, rel_ofs)
        yield stop_note(pitch + MIDI_BASE, dur)
        prev = ofs + dur

def matrix_to_midi(matrix, fname):
    matrix = matrix.reshape(64, 4)
    matrix = np.vstack((matrix,)*4)
    midi = MidiFile(type = 1)
    for col in matrix.T:
        midi.tracks.append(MidiTrack(column_to_track(col)))
    midi.save(fname)

########################################################################
# Real and fake data
########################################################################
def mod_file_to_patterns(mod_file):
    SP.print(str(mod_file))
    try:
        mod = load_file(mod_file)
    except PowerPackerModule:
        return []
    rows = linearize_rows(mod)
    volumes = [header.volume for header in mod.sample_headers]
    notes = rows_to_mod_notes(rows, volumes)
    percussive = {s for (s, p) in sample_props(mod, notes)
                  if p.is_percussive}
    return [pattern_to_matrix(pat, percussive) for pat in mod.patterns]

def good_pattern(pat):
    return 30 <= len(pat[pat > 0]) <= 175 and len(np.unique(pat)) > 3

def load_data_from_disk(corpus_path, kb_limit):
    index = load_index(corpus_path)
    mods = [mod for mod in index.values()
            if (mod.n_channels == 4
                and mod.format == 'MOD'
            and mod.kb_size <= kb_limit)]
    file_paths = [corpus_path / mod.genre / mod.fname for mod in mods]
    patterns = [mod_file_to_patterns(path) for path in file_paths]
    patterns = flatten(patterns)
    patterns = [p for p in patterns if good_pattern(p)]
    return np.array(patterns, dtype = np.int8)

def load_data(corpus_path, kb_limit):
    cache_path = corpus_path / Path('patterns.npy')
    if not cache_path.exists():
        images = load_data_from_disk(corpus_path, kb_limit)
        with open(cache_path, 'wb') as f:
            np.save(f, images)
    with open(cache_path, 'rb') as f:
        return np.load(f)

########################################################################
# Sampling
########################################################################
def generate_latent_points(latent_dim, n_samples):
    x_input = np.random.randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

def generate_fake_samples(g_model, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X = g_model.predict(x_input)
    y = np.zeros((n_samples, 1))
    return X, y

def generate_real_samples(dataset, n_samples):
    ix = np.random.randint(0, len(dataset), n_samples)
    X_real = dataset[ix]
    y_real = np.ones((n_samples, 1))
    return X_real, y_real

def generate_real_and_fake_samples(dataset, g_model, latent_dim,
                                   n_samples):
    half = n_samples // 2
    X_real, y_real = generate_real_samples(dataset, half)
    X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half)
    return np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))

########################################################################
# Model def
########################################################################
def define_discriminator(in_shape):
    model = Sequential()
    model.add(Conv2D(64, (3,3),
                     strides=(2, 2), padding='same',
                     input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt, metrics=['accuracy'])
    return model

def define_generator(latent_dim):
    model = Sequential()
    # foundation for 16x1 image
    n_nodes = 128 * 16 * 1
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((16, 1, 128)))
    # upsample to 32x2
    model.add(Conv2DTranspose(128, (4,4),
                              strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 64x4
    model.add(Conv2DTranspose(128, (4,4),
                              strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(1, (16,1), activation='sigmoid', padding='same'))
    return model

def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(g_model)
    # add the discriminator
    model.add(d_model)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

def colorize_image(img):
    mid = 127
    hi = 255

    cols = [(hi, mid, 0), (hi, 0, mid), (mid, hi, 0),
            (mid, 0, hi), (0, mid, hi), (0, hi, mid),
            (hi, hi, 0), (hi, 0, hi), (0, hi, hi),
            (mid, mid, 0), (mid, 0, mid), (0, mid, mid)]
    color_img = np.zeros((64, 4, 3), dtype = int)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pitch = img[i, j]
            if pitch == 0:
                color_img[i, j] = (220, 220, 220)
            else:
                octave = ((pitch - 1) // 12) + 1
                octave = min(octave, 3)
                note = (pitch - 1) % 12
                col = cols[note]
                col = tuple(int(c * 0.33 * octave)
                            for c in col)
                color_img[i, j] = col
    return color_img

def plot_image(img, ax):
    #img = np.round(img.reshape(64, 4) * SCALE)
    img = img.astype(int)
    color_img = colorize_image(img)
    ax.imshow(color_img, interpolation = 'none', aspect = 'equal')
    ax.axis('off')

N_ROWS = 5
N_COLS = 5
def save_plot(X_fake, epoch):
    assert len(X_fake) == N_ROWS * N_COLS
    fig, axs = plt.subplots(N_ROWS, N_COLS)
    for i in range(N_ROWS):
        for j in range(N_COLS):
            plot_image(X_fake[i * N_COLS + j], axs[i, j])
    if type(epoch) == int:
        fig.savefig('gen-%03d.png' % epoch)
    else:
        fig.savefig('gen-%s.png' % epoch)
    plt.close()

def summarize_performance(epoch, g_model, d_model,
                          dataset, latent_dim):
    n_samples = N_ROWS * N_COLS
    X_real, y_real = generate_real_samples(dataset, n_samples)
    _, acc_real = d_model.evaluate(X_real, y_real, verbose = 0)

    X_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    _, acc_fake = d_model.evaluate(X_fake, y_fake, verbose = 0)

    X_fake = X_fake.reshape(len(X_fake), 64, 4)
    X_fake = np.round(X_fake * SCALE)
    X_fake = X_fake.astype(int)

    print('>Accuracy real: %.0f%%, fake: %.0f%%'
          % (acc_real*100, acc_fake*100))
    save_plot(X_fake, epoch)
    for i, X in enumerate(X_fake):
        matrix_to_midi(X, 'test-gan-%03d.mid' % i)

SCALE = None

def main():
    global SCALE
    args = docopt(__doc__, version = 'GAN Model 1.0')
    SP.enabled = args['--verbose']
    kb_limit = int(args['--kb-limit'])
    corpus_path = Path(args['<corpus>'])

    dataset = load_data(corpus_path, kb_limit)
    dataset = dataset.reshape(len(dataset), 64, 4, 1)

    # Scale to [0,1]
    SCALE = np.max(dataset)
    dataset = dataset / SCALE

    n_patterns = len(dataset)
    n_train = int(n_patterns * 0.8)
    train, test = dataset[:n_train], dataset[n_train:]
    SP.print('%d train and %d test patterns.', (len(train), len(test)))

    latent_dim = 100
    d_model = define_discriminator((64, 4, 1))
    g_model = define_generator(latent_dim)
    gan_model = define_gan(g_model, d_model)

    n_batch = 256
    n_epochs = 500
    batches_per_epoch = n_patterns // n_batch
    for i in range(n_epochs):
        if i  % 50 == 0:
            summarize_performance(i, g_model, d_model, test, latent_dim)
        for j in range(batches_per_epoch):
            X, y = generate_real_and_fake_samples(train, g_model,
                                                  latent_dim, n_batch)
            d_loss, _ = d_model.train_on_batch(X, y)

            X_gan = generate_latent_points(latent_dim, n_batch)
            y_gan = np.ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(X_gan, y_gan)

            if j == 0:
                fmt = '>%d, %d/%d, d=%.3f, g=%.3f'
                print(fmt % (i + 1, j + 1, batches_per_epoch,
                             d_loss, g_loss))
    summarize_performance('final', g_model, d_model, test, latent_dim)


if __name__ == '__main__':
    main()
