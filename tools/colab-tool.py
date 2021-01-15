# Copyright (C) 2020 Björn Lindqvist <bjourne@gmail.com>
#
# Interact with Google Colab. There are maybe easier ways of doing
# this.
"""Colab Tool

Usage:
    colab-tool.py [-v] --authority=<s> --root-path=<s> get-data
    colab-tool.py [-v] --authority=<s> --root-path=<s>
        upload-code
    colab-tool.py [-v] --authority=<s> --root-path=<s>
        upload-caches <corpus-path>
    colab-tool.py [-v] --authority=<s> --root-path=<s>
        upload-file <local-file>
    colab-tool.py [-v] --authority=<s> --root-path=<s>
        run-file -- <file> <args>...
    colab-tool.py [-v] --authority=<s> --root-path=<s>
        upload-and-run-file [--drop-path] -- <file> <args>...

Options:
    -h --help                   show this screen
    -v --verbose                print more output
    --authority=<s>             user:pwd@host:port
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
    SP.header('UPLOADING %d FILE(S)' % len(files))
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

def run_python_file(connection, root_path, file_name, args):
    cmds = [f'cd "{root_path}"',
            'export PYTHONPATH="."',
            'python3 %s %s' % (file_name, ' '.join(args))]
    script = ' && '.join(cmds)
    SP.print('Running %s...' % file_name)
    connection.run(script, pty = True)

def main():
    args = docopt(__doc__, version = 'Colab Tool 1.0')
    SP.enabled = args['--verbose']
    root_path = Path(args['--root-path'])
    auth = args['--authority']
    userinfo, netloc = auth.split('@')
    _, password = userinfo.split(':')
    host, port = netloc.split(':')
    port = int(port)

    connect_kwargs = {'password' : password}
    conn = Connection(host, 'root', port, connect_kwargs = connect_kwargs)
    sftp = conn.sftp()
    remote_mkdir_safe(sftp, root_path)
    sftp.chdir(str(root_path))
    if args['get-data']:
        get_data(conn, sftp)
    elif args['upload-code']:
        upload_code(conn, sftp)
    elif args['upload-caches']:
        corpus_path = Path(args['<corpus-path>'])
        upload_caches(conn, corpus_path)
    elif args['upload-file']:
        local_path = Path(args['<local-file>'])
        upload_file(conn, local_path)
    elif args['upload-and-run-file']:
        src = Path(args['<file>'])
        dst = src.parent
        if args['--drop-path']:
            dst = Path('.')
        upload_files(conn, [(src, dst)])
        if args['--drop-path']:
            src = src.name

        run_python_file(conn, root_path, str(src), args['<args>'])
    elif args['run-file']:
        run_python_file(conn, root_path, args['<file>'], args['<args>'])
    else:
        assert False

if __name__ == '__main__':
    main()
