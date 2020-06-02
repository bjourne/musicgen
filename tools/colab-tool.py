# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
#
# Interact with Google Colab. There are maybe easier ways of doing
# this.
"""Colab Tool

Usage:
    colab-tool.py [-v] --port=<i> --password=<s> --root-path=<s>
        get-data
    colab-tool.py [-v] --port=<i> --password=<s> --root-path=<s>
        upload-code
    colab-tool.py [-v] --port=<i> --password=<s> --root-path=<s>
        upload-caches <corpus-path>
    colab-tool.py [-v] --port=<i> --password=<s> --root-path=<s>
        upload-file <local-file>
    colab-tool.py [-v] --port=<i> --password=<s> --root-path=<s>
        upload-and-run-file <local-file>
    colab-tool.py [-v] --port=<i> --password=<s> --root-path=<s>
        run-training
    colab-tool.py [-v] --port=<i> --password=<s> --root-path=<s>
        train-lstm-poly

Options:
    -h --help                   show this screen
    -v --verbose                print more output
    --port=<i>                  port number
    --password=<s>              password
    --root-path=<s>             path to code and data on colab
"""
from docopt import docopt
from fabric import Connection
from musicgen.utils import SP, flatten
from pathlib import Path

def get_data(connection, sftp):
    paths = [Path(p) for p in sftp.listdir()]
    paths = [p for p in paths if p.suffix in ('.mid', '.png')]

    SP.header('DOWNLOADING %d FILES' % len(paths))
    for path in paths:
        SP.print(path)
        connection.get(path)
    SP.leave()

def upload_files(connection, files):
    SP.header('UPLOADING %d FILES' % len(files))
    for src, dst in sorted(files):
        SP.print('%-30s => %s' % (src, dst))
        connection.put(str(src), str(dst))
    SP.leave()

def remote_mkdir_safe(sftp, path):
    try:
        sftp.mkdir(str(path))
    except OSError:
        pass

def upload_code(connection, sftp):
    dirs = [Path(d) for d in ['musicgen', 'tools']]
    for dir in dirs:
        remote_mkdir_safe(sftp, dir)
    files = flatten([[(src, d) for src in d.glob('*.py')] for d in dirs])
    upload_files(connection, files)

def upload_caches(connection, corpus_path):
    caches = [corpus_path.glob(f'*.{ext}') for ext in ['pickle', 'npy']]
    caches = flatten(caches) + [corpus_path / 'index']
    files = [(c, c.name) for c in caches]
    upload_files(connection, files)

def upload_file(connection, local_path):
    files = [(local_path, local_path.name)]
    upload_files(connection, files)

def run_python_file(connection, root_path, file_name):
    cmds = prepare_commands(root_path) \
        + ['python3 %s' % file_name]
    script = ' && '.join(cmds)
    connection.run(script, pty = True)

def prepare_commands(root_path):
    return ['pip3 install mido construct',
            f'cd "{root_path}"',
            'export PYTHONPATH="."']

def run_training(connection, root_path):
    cmds = prepare_commands(root_path) \
        + ['python3 tools/train-lstm.py -v . --pack-mycode']
    script = ' && '.join(cmds)
    connection.run(script, pty = True)

def train_lstm_poly(connection, root_path):
    cmds = prepare_commands(root_path) \
        + ['python3 tools/train-lstm-poly.py -v .']
    script = ' && '.join(cmds)
    connection.run(script, pty = True)

def main():
    args = docopt(__doc__, version = 'Colab Tool 1.0')
    SP.enabled = args['--verbose']
    port = int(args['--port'])
    password = args['--password']
    root_path = Path(args['--root-path'])

    connect_kwargs = {'password' : password}
    connection = Connection('0.ssh.ngrok.io', 'root', port,
                   connect_kwargs = connect_kwargs)
    sftp = connection.sftp()
    remote_mkdir_safe(sftp, root_path)
    sftp.chdir(str(root_path))
    if args['get-data']:
        get_data(connection, sftp)
    elif args['upload-code']:
        upload_code(connection, sftp)
    elif args['run-training']:
        run_training(connection, root_path)
    elif args['upload-caches']:
        corpus_path = Path(args['<corpus-path>'])
        upload_caches(connection, corpus_path)
    elif args['upload-file']:
        local_path = Path(args['<local-file>'])
        upload_file(connection, local_path)
    elif args['train-lstm-poly']:
        train_lstm_poly(connection, root_path)
    elif args['upload-and-run-file']:
        local_path = Path(args['<local-file>'])
        upload_file(connection, local_path)
        run_python_file(connection, root_path, local_path.name)
    else:
        assert False

if __name__ == '__main__':
    main()
