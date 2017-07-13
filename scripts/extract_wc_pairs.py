#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import numpy as np
import codecs
import argparse
__author__ = 'arenduchintala'
"""
The following sys setup fixes a lot of issues with writing and reading in utf-8 chars.
WARNING: pdb does not seem to work if you do reload(sys)
"""
reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

START_SYM = '<START>'
END_SYM = '<END>'

if __name__ == '__main__':
    opt= argparse.ArgumentParser(description="write program description here")

    #insert options here
    opt.add_argument('-c', action='store', dest='corpus_file', default='', required = True)
    opt.add_argument('-o', action='store', dest='out_corpus_file', default='', required = True)
    opt.add_argument('-w', action='store', dest='window_size', default=3, type=int)
    opt.add_argument('-f', action='store', dest='freq_cutoff', default=5000, type=int)
    opt.add_argument('--subsample', action='store', dest='token_subsample', default=1e-3, type=float)
    opt.add_argument('-i', action='store_true', dest='int_data', default=False, help='add this flag if you want to save just the word ids')
    options = opt.parse_args()
    word2count = {}
    with codecs.open(options.corpus_file, 'r', 'utf-8') as _file:
        for line in _file:
            line = line.lower()
            for word in line.split():
                word2count[word] = word2count.get(word, 0)  + 1
    top_wordcounts = sorted(word2count.items(), key=lambda x: -x[1])
    top_wordcounts = top_wordcounts[:options.freq_cutoff]
    word_count = sum([c for w,c in top_wordcounts])
    word2wordid = {START_SYM: 0, END_SYM: 1}
    word2keepprob = {START_SYM: 1., END_SYM: 1.}
    for top_w, top_c in top_wordcounts[:options.freq_cutoff]:
        word2wordid[top_w] = word2wordid.get(top_w, len(word2wordid))
        top_z = float(top_c) / float(word_count)
        word2keepprob[top_w] = (np.sqrt(top_z / options.token_subsample)  + 1.) * (options.token_subsample/ top_z)

    line_idx = -1
    corpus_file_prefix = options.out_corpus_file 
    skipgram_data = codecs.open(corpus_file_prefix + '.sg.txt', 'w', 'utf-8')
    cbow_data = codecs.open(corpus_file_prefix + '.cbow.txt', 'w', 'utf-8')
    vocab_data = codecs.open(corpus_file_prefix + '.vocab.txt', 'w', 'utf-8')
    vocab_data.write('\n'.join([str(v_id) + ' ' + v + ' '  + str(word2count.get(v,0)) for v,v_id in sorted(word2wordid.items(), key=lambda x: x[1])]))
    vocab_data.flush()
    vocab_data.close()
    with codecs.open(options.corpus_file, 'r', 'utf-8') as _file:
        for line in _file:
            line = line.lower()
            line_idx += 1
            words = line.split()
            words = [w for w in words if (w in word2wordid and word2keepprob[w] > np.random.rand())]
            if len(words) <= 2:
                continue
            for w_idx, w in enumerate(words):
                if word2wordid.get(w, None) is not None:
                    context_left = words[max(0, w_idx - options.window_size): w_idx] 
                    context_left = [cl for cl in context_left]
                    context_left = context_left if len(context_left) == options.window_size else ([START_SYM] * (options.window_size - len(context_left))) + context_left
                    context_right = words[w_idx + 1: w_idx  + options.window_size + 1] 
                    context_right = [cr for cr in context_right]
                    context_right = context_right if len(context_right) == options.window_size else context_right + ([END_SYM] * (options.window_size - len(context_right)))
                    int_context_left = [str(word2wordid[cl]) for cl in context_left]
                    int_context_right = [str(word2wordid[cl]) for cl in context_right]
                    if options.int_data:
                        cbow_data.write(' '.join(int_context_left + int_context_right) + ' ' + str(word2wordid[w]) + '\n')
                    else:
                        cbow_data.write(' '.join(context_left + context_right) + ' ' + w.strip().lower() + '\n')
                    for c_idx, c in enumerate(context_left, -options.window_size): #context_right:
                        if options.int_data:
                            skipgram_data.write(str(word2wordid[w]) + ' ' + str(word2wordid[c]) + ' '+ str(c_idx) + '\n')
                        else:
                            skipgram_data.write(w + ' ' + c + ' '+ str(c_idx) + '\n')
                    for c_idx, c in enumerate(context_right, 1): 
                        if options.int_data:
                            skipgram_data.write(str(word2wordid[w]) + ' ' + str(word2wordid[c]) + ' '+ str(c_idx) + '\n')
                        else:
                            skipgram_data.write(w + ' ' + c + ' '+ str(c_idx) + '\n')
    cbow_data.flush()
    skipgram_data.flush()
    cbow_data.close()
    skipgram_data.close()
