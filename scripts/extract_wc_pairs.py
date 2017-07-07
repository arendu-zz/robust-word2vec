#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
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

START_SYM = '<SRT>'
OOV = '<OOV>'
END_SYM = '<END>'

if __name__ == '__main__':
    opt= argparse.ArgumentParser(description="write program description here")

    #insert options here
    opt.add_argument('-c', action='store', dest='corpus_file', default='', required = True)
    opt.add_argument('-w', action='store', dest='window_size', default=3, type=int)
    opt.add_argument('-f', action='store', dest='freq_cutoff', default=5000, type=int)
    options = opt.parse_args()
    word2count = {}
    with codecs.open(options.corpus_file, 'r', 'utf-8') as _file:
        for line in _file:
            line = line.lower()
            for word in line.split():
                word2count[word] = word2count.get(word, 0)  + 1
    top_wordcounts = sorted(word2count.items(), key=lambda x: -x[1])
    top_wordcounts = top_wordcounts[:options.freq_cutoff]
    word2wordid = {START_SYM: 0, END_SYM: 1, OOV: 2}
    for top_w, top_c in top_wordcounts[:options.freq_cutoff]:
        word2wordid[top_w] = word2wordid.get(top_w, len(word2wordid))

    line_idx = -1
    skipgram_data = codecs.open(options.corpus_file + '.skipgram.data.txt', 'w', 'utf-8')
    cbow_data = codecs.open(options.corpus_file + '.cbow.data.txt', 'w', 'utf-8')
    vocab_data = codecs.open(options.corpus_file + '.vocab.txt', 'w', 'utf-8')
    with codecs.open(options.corpus_file, 'r', 'utf-8') as _file:
        for line in _file:
            line = line.lower()
            line_idx += 1
            words = line.split()
            for w_idx, w in enumerate(words):
                context_left = words[w_idx - options.window_size: w_idx] 
                context_left = [(cl if cl in word2wordid else OOV) for cl in context_left]
                context_left = context_left if len(context_left) == options.window_size else ([START_SYM] * (options.window_size - len(context_left))) + context_left
                context_right = words[w_idx + 1: w_idx  + options.window_size] 
                context_right = [(cr if cr in word2wordid else OOV) for cr in context_right]
                context_right = context_right if len(context_right) == options.window_size else context_right + ([END_SYM] * (options.window_size - len(context_right)))
                cbow_data.write(' '.join(context_left + context_right) + ' ' + w.strip().lower() + '\n')
                _w = w if w in word2wordid else OOV
                for c in context_left: #context_right:
                    _c = c if c in word2wordid else OOV
                    skipgram_data.write(_w + ' ' + _c + ' l\n')
                for c in context_right: 
                    _c = c if c in word2wordid else OOV
                    skipgram_data.write(_w + ' ' + _c + ' r\n')
    cbow_data.flush()
    skipgram_data.flush()
    cbow_data.close()
    skipgram_data.close()
    vocab_data.write('\n'.join([str(v_id) + ' ' + v + ' '  + str(word2count.get(v,0)) for v,v_id in sorted(word2wordid.items(), key=lambda x: x[1])]))
    vocab_data.flush()
    vocab_data.close()
