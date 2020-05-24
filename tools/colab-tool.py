# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
#
# Interact with Google Colab. There are maybe easier ways of doing
# this.
"""Colab Tool

Usage:
    colab-tool.py [-v] --port=<str> --password=<int> --root-path=<str>
        get-data
    colab-tool.py [-v] --port=<str> --password=<int> --root-path=<str>
        upload-code
    colab-tool.py [-v] --port=<str> --password=<int> --root-path=<str>
        run-training

Options:
    -h --help                   show this screen
    -v --verbose                print more output
    --port=<int>                port number
    --password=<str>            password
    --root-path=<str>           path to code and data on colab
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

def upload_code(connection):
    dirs = [Path(d) for d in ['musicgen', 'tools']]
    files = flatten([[(src, d) for src in d.glob('*.py')] for d in dirs])
    SP.header('UPLOADING %d FILES' % len(files))
    for src, dst in sorted(files):
        SP.print('%-30s => %s' % (src, dst))
        connection.put(str(src), str(dst))
    SP.leave()

def run_training(connection, root_path):
    cmds = ['pip3 install mido construct',
            f'cd "{root_path}"',
            'export PYTHONPATH="."',
            'python3 tools/train-model.py -v .'
            ]
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
    sftp.chdir(str(root_path))
    if args['get-data']:
        get_data(connection, sftp)
    elif args['upload-code']:
        upload_code(connection)
    elif args['run-training']:
        run_training(connection, root_path)

if __name__ == '__main__':
    main()
