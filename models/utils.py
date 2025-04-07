import os
import sys
import numpy as np
import torch


# clone stdout to file. Optionally process carriage 'proc_cr' return to handle tqdm bars as displayed in stdout.
class Tee(object):
    def __init__(self, *args, proc_cr=False, **kwargs):
        self.file = open(*args, **kwargs)
        self.stdout = sys.stdout
        self.proc_cr = proc_cr
        #self.dbg_print = lambda msg: print(msg, file=self.stdout)
        self.dbg_print = lambda msg: None
        self.nl_pos = 0
        sys.stdout = self

    def __del__(self):
        self.dbg_print(f'del {self.file.name}')
        self.free()

    def free(self):
        if self.file.closed:
            self.dbg_print(f'  !!no action since resources already freed')
            return
        # self.flush()
        sys.stdout = self.stdout
        self.file.close()

    def __enter__(self):
        self.dbg_print(f'enter {self.file.name}')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dbg_print(f'exit {self.file.name}')
        self.free()

    def write(self, data):
        self.stdout.write(data)
        if self.proc_cr:
            # split lines such that last line never ends with a linesep
            lines = (data + 'X').splitlines(keepends=True)
            lines[-1] = lines[-1][:-1]
            # write full lines
            for line in lines[:-1]:
                self.file.write(line)
                # convert CR
                if line.endswith('\r'):
                    self.file.seek(self.nl_pos)
                else:
                    self.nl_pos = self.file.tell()
            # write incomplete (last) line
            self.file.write(lines[-1])
        else:
            self.file.write(data)

    def flush(self):
        # print('flush', file=self.stdout)
        self.file.flush()


# Early stops the training if validation score doesn't improve after a given patience. Optionally saves model params if
# 'checkpoint_file' is specified.
class EarlyStopping:
    def __init__(self, patience, verbose=False, higher_better=False, delta=0.0, checkpoint_file=None,
                 print_file=None, float_fmt='.6f'):
        self.patience = patience
        self.verbose = verbose
        self.higher_better = higher_better
        self.checkpoint_file = checkpoint_file
        assert delta >= 0.0, "negative 'delta' not allowed"
        self.delta = delta if higher_better else -delta
        self.print_file = print_file
        self.float_fmt = float_fmt
        self.counter = 0
        self.early_stop = False
        # initial value is worst possible value
        self.best_score = -np.inf if higher_better else np.inf

    def __call__(self, score, model, epoch):
        threshold = self.best_score + self.delta
        if self.higher_better:
            improved = score > threshold
        else:
            improved = score < threshold
        if improved:
            if self.verbose:
                #msg = f'Score improved: {self.best_score:{self.float_fmt}} --> {score:{self.float_fmt}} - saving model'
                msg = f'{">" if self.higher_better else "<"} {self.best_score:{self.float_fmt}} --> checkpoint'
                print(msg, file=self.print_file)
            if self.checkpoint_file:
                self.save_checkpoint(score, model, epoch)
            self.best_score = score
            self.counter = 0
        if self.counter >= self.patience:
            self.early_stop = True
        self.counter += 1
        return self.early_stop, improved

    def save_checkpoint(self, score, model, epoch):
        torch.save(model.state_dict(), self.checkpoint_file)
