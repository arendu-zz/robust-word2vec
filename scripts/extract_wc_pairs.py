#!/usr/bin/env python
__author__ = 'arenduchintala'
import sys
import logging
import codecs
import argparse
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

if __name__ == '__main__':
    opt= argparse.ArgumentParser(description="write program description here")

    #insert options here
    opt.add_argument('-c', action='store', dest='corpus_file', default='', required = True)
    opt.add_argument('-w', action='store', dest='window_size', default=5, type=int, required = True)
    opt.add_argument('-f', action='store', dest='freq_cutoff', default=10000, type=int, required = False)
    options = opt.parse_args()
    logging.debug(str(options))
    word2count = {}
    with codecs.open(options.corpus_file, 'r', 'utf-8') as _file:
        for line in _file:
            for word in line.split():
                word2count[word] = word2count.get(word, 0)  + 1
    top_wordcounts = sorted(word2count.items(), key=lambda x: -x[1])
    top_wordcounts = top_wordcounts[:options.freq_cutoff]
    top_wordcounts = dict((w,c) for w,c in top_wordcounts)
    line_idx = -1
    skipgram_data = codecs.open(options.corpus_file + 'skipgram.data.txt', 'w', 'utf-8')
    cbow_data = codecs.open(options.corpus_file + 'cbow.data.txt', 'w', 'utf-8')
    with codecs.open(options.corpus_file, 'r', 'utf-8') as _file:
        for line in _file:
            line_idx += 1
            words = line.split()
            for w_idx, w in enumerate(words):
                cs = words[w_idx - options.window_size: w_idx] + words[w_idx + 1: w_idx  + options.window_size] 
                cbow_data.write(' '.join(cs) + ' ' + w.strip().lower() + '\n')
                for c in cs:
                    if w in top_wordcounts and c in top_wordcounts:
                        skipgram_data.write(w.strip().lower() + ' ' + c.strip().lower() + '\n')
                    else:
                        pass
    cbow_data.flush()
    skipgram_data.flush()
    cbow_data.close()
    skipgram_data.close()
