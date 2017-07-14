#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import codecs
import argparse
from gensim.models import Word2Vec as W2V
from gensim.models.word2vec import LineSentence
__author__ = 'arenduchintala'
"""
The following sys setup fixes a lot of issues with writing and reading in utf-8 chars.
WARNING: pdb does not seem to work if you do reload(sys)
"""
reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)


if __name__ == '__main__':
    opt= argparse.ArgumentParser(description="write program description here")

    #insert options here
    opt.add_argument('-c', action='store', dest='corpus_file', required = True)
    opt.add_argument('--min-count', action='store', dest='min_count', type = int, required = True)
    opt.add_argument('-e', action='store', dest='embed_size', type = int, required = True)
    opt.add_argument('-w', action='store', dest='window_size', type = int, default = 3)
    opt.add_argument('-m', action='store', dest='model', choices = ['cbow', 'sg'])
    opt.add_argument('-s', action='store', dest='save_path', required = False, default = None)
    options = opt.parse_args()
    #sentences = [s.split() for s in codecs.open(options.corpus_file, 'r', 'utf-8').readlines()]
    sentences = LineSentence(options.corpus_file) #[s.split() for s in codecs.open(options.corpus_file, 'r', 'utf-8').readlines()]
    model = W2V(sentences = sentences, size = options.embed_size, min_count = options.min_count, window = options.window_size)
    model.init_sims(replace=True)
    model.save(options.save_path + 'gensim.model')
    model.wv.save_word2vec_format(options.save_path + '.gensim.vecs', binary=False)
