#!/usr/bin/env python
# -*- coding: utf-8 -*-
from models import CBOW, Vocab
import codecs
import numpy as np
import argparse
def load_data(data_file, data_type = 'cbow'):
    x_full = []
    y_full = []
    with codecs.open(data_file, 'r', 'utf-8') as _file:
        for line in _file:
            if data_type == 'cbow':
                int_line = [int(i) for i in line.strip().split()]
                x_full.append(int_line[:-1])
                y_full.append(int_line[-1])
            elif data_type == 'skipgram':
                int_line = [int(i) for i in line.strip().split()]
                x_full.append(int_line[0])
                y_full.append(int_line[1])
            else:
                raise BaseException("unknown data_type" + data_type)
    return np.asarray(x_full), np.asarray(y_full)


if __name__ == '__main__':
    opt= argparse.ArgumentParser(description="write program description here")
    #insert options here
    opt.add_argument('-v', action='store', dest='vocab_file', required = True)
    opt.add_argument('-s', action='store', dest='save_path', required = True)
    opt.add_argument('-t', action='store', dest='training_data', required = True)
    opt.add_argument('-d', action='store', dest='dev_data', required = True)
    opt.add_argument('-e', action='store', dest='embed_size', type = int, required = True)
    opt.add_argument('--bs', action='store', dest='batch_size', default = 128)
    opt.add_argument('-m', action='store', dest='model', help='cbow or skipgram', required = True)
    options = opt.parse_args()
    vocab = Vocab(options.vocab_file)
    X_full, Y_full = load_data(options.training_data, options.model)
    X_dev, Y_dev = load_data(options.dev_data, options.model)
    print vocab.voc_dist.shape
    print X_full.shape
    print Y_full.shape
    cbow = CBOW(vocab_model = vocab, batch_size = options.batch_size, context_size = X_full.shape[1], embed_size = options.embed_size, reg=0.0)
    t_idx = np.arange(X_full.shape[0])
    epochs = 5
    mask_diag = np.ones((vocab.size, vocab.size), dtype=bool)
    np.fill_diagonal(mask_diag, 0)
    for e_idx in xrange(epochs):
        np.random.shuffle(t_idx)
        batches = np.array_split(t_idx, X_full.shape[0] / options.batch_size)
        for b_idx, batch_idxs in enumerate(batches):
            _batch_loss  = cbow.do_update(0.001, X_full[batch_idxs,:], Y_full[batch_idxs])
            print e_idx, b_idx, 'of', len(batches), _batch_loss
        cosine_sims = cbow.cosine_similarity()
        sorted_cosine_sims = np.argpartition(-cosine_sims, 10)
        sorted_cosine_sims = sorted_cosine_sims[:, :10]
        for row_idx in xrange(sorted_cosine_sims.shape[0]):
            print cbow.vocab_model.id2voc[row_idx], '\t:', ', '.join([cbow.vocab_model.id2voc[i] for i in sorted_cosine_sims[row_idx,:]])
            pass
        cbow.save_model(options.save_path + str(e_idx) + '.json')
