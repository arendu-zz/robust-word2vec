#!/usr/bin/env python
# -*- coding: utf-8 -*-
from models import CBOW, Vocab
import pdb
import codecs
import numpy as np
from numpy import unravel_index
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
    opt.add_argument('-t', action='store', dest='training_data', required = True)
    opt.add_argument('-d', action='store', dest='dev_data', required = True)
    opt.add_argument('--bs', action='store', dest='batch_size', default = 128)
    opt.add_argument('-m', action='store', dest='model', help='cbow or skipgram', required = True)
    options = opt.parse_args()
    vocab = Vocab(options.vocab_file)
    X_full, Y_full = load_data(options.training_data, options.model)
    X_dev, Y_dev = load_data(options.dev_data, options.model)
    print vocab.voc_dist.shape
    print X_full.shape
    print Y_full.shape
    cbow = CBOW(vocab_model = vocab, batch_size = options.batch_size, context_size = X_full.shape[1], embed_size = 300)
    t_idx = np.arange(X_full.shape[0])
    epochs = 2
    mask_diag = np.ones((vocab.size, vocab.size), dtype=bool)
    np.fill_diagonal(mask_diag, 0)
    for e_idx in xrange(epochs):
        np.random.shuffle(t_idx)
        batches = np.array_split(t_idx, X_full.shape[0] / options.batch_size)
        for b_idx, batch_idxs in enumerate(batches):
            _batch_loss  = cbow.do_update(0.1, X_full[batch_idxs,:], Y_full[batch_idxs])
            print e_idx, b_idx, 'of', len(batches), _batch_loss
            if b_idx % 1000 == 0:
                cosine_sims = cbow.cosine_similarity()
                max_cs = cosine_sims[mask_diag].max()
                argmax_cs = unravel_index(cosine_sims[mask_diag].argmax(), cosine_sims.shape)
                min_cs = cosine_sims[mask_diag].min()
                argmin_cs = unravel_index(cosine_sims[mask_diag].argmin(), cosine_sims.shape)
                print 'max:', vocab.id2voc[argmax_cs[0]], vocab.id2voc[argmax_cs[1]], max_cs
                print 'min:', vocab.id2voc[argmin_cs[0]], vocab.id2voc[argmin_cs[1]], min_cs
                sim1 = cbow.cosine_similarity_cell(cbow.vocab_model.voc2id['person'], cbow.vocab_model.voc2id['person'])
                sim2 = cbow.cosine_similarity_cell(cbow.vocab_model.voc2id['person'], cbow.vocab_model.voc2id['people'])
                sim3 = cbow.cosine_similarity_cell(cbow.vocab_model.voc2id['person'], cbow.vocab_model.voc2id['english'])
                sim4 = cbow.cosine_similarity_cell(cbow.vocab_model.voc2id['person'], cbow.vocab_model.voc2id['the'])
                print 'person-person person-people person-english person-the', '%.4f'% sim1, '%.4f'% sim2, '%.4f'% sim3, '%.4f'% sim4
                #_dev_losses = cbow.get_loss_t(X_dev[:30000], Y_dev[:30000])
        save_path = './data/model.' + str(e_idx) + '.json'
        cbow.save_model(save_path)
