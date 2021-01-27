class LSTMParams:
    def __init__(self, docopt_args):
        self.rec_dropout = float(docopt_args['--rec-dropout'])
        self.emb_size = int(docopt_args['--emb-size'])
        self.lstm1_units = int(docopt_args['--lstm1-units'])
        self.lstm2_units = int(docopt_args['--lstm2-units'])

    def as_file_name_part(self):
        fmt = '%.2f-%03d-%04d-%04d'
        args = (self.rec_dropout, self.emb_size,
                self.lstm1_units, self.lstm2_units)
        return fmt % args

class TransformerParams:
    def __init__(self, docopt_args):
        pass

    def as_file_name_part(self):
        return ''

class ModelParams:
    @classmethod
    def from_docopt_args(cls, args):
        code_type = args['<code-type>']
        model_type = 'lstm' if args['lstm'] else 'transformer'

        if model_type == 'lstm':
            type_params = LSTMParams(args)
        else:
            type_params = TransformerParams(args)

        dropout = float(args['--dropout'])
        batch_size = int(args['--batch-size'])
        lr = float(args['--lr'])
        seq_len = int(args['--seq-len'])
        epochs = int(args['--epochs'])
        return cls(code_type, model_type,
                   dropout, batch_size, lr, seq_len, epochs,
                   type_params)

    def __init__(self,
                 code_type, model_type,
                 dropout, batch_size, lr, seq_len, epochs,
                 type_params):
        self.code_type = code_type
        self.model_type = model_type

        self.dropout = dropout
        self.batch_size = batch_size
        self.lr = lr
        self.seq_len = seq_len
        self.epochs = epochs
        self.type_params = type_params

    def to_string(self):
        fmt = '%s_%s-%.2f-%03d-%.5f-%03d-%03d-%s'
        args = (self.code_type, self.model_type,
                self.dropout, self.batch_size,
                self.lr, self.seq_len, self.epochs,
                self.type_params.as_file_name_part())
        return fmt % args

    def weights_file(self):
        return 'weights_%s.h5' % self.to_string()

    def log_file(self):
        return 'log_%s.log' % self.to_string()
