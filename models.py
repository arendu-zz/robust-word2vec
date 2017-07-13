#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import codecs
import numpy as np
import theano
import theano.tensor as T
from optimizers import adam, sgd, sgd_clipped, rmsprop, momentum, adagrad, rmsprop_clipped
from abc import ABCMeta, abstractmethod
import json
__author__ = 'arenduchintala'
"""
The following sys setup fixes a lot of issues with writing and reading in utf-8 chars.
WARNING: pdb does not seem to work if you do reload(sys)
"""
reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

START_SYM = '<SRT>'
END_SYM = '<END>'
OOV = '<OOV>'

if theano.config.floatX == 'float32':
    intX = np.int32
    floatX = np.float32
else:
    intX = np.int64
    floatX = np.float64

def _get_weights(name, shape1, shape2, init='rand'):
    if init == 'rand':
        x = 0.01 * np.random.rand(shape1, shape2)
    elif init == 'nestrov':
        x = np.random.uniform(-np.sqrt(1. / shape2), np.sqrt(1. / shape2), (shape1, shape2))
    else:
        raise NotImplementedError("don't know how to initialize the weight matrix")
    return theano.shared(floatX(x), name)

class BaseModel(object):
    __metaclass__ = ABCMeta

    def __init__(self, vocab_model, batch_size, embed_size, saved_model = None, noise_sample_size = None, noise_dist = None, reg = 0.0, optimize ='rms'):
        self.vocab_model = vocab_model
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.noise_dist = noise_dist
        self.noise_sample_size = noise_sample_size
        self.reg = reg
        self._eps = 1e-8
        self.params = []
        self.reg_params = []
        self.set_optimizer(optimize)
        if saved_model is None:
            W_in = _get_weights('W_IN', self.vocab_model.size, self.embed_size, 'nestrov')
            W_context = _get_weights('W_CONTEXT', self.embed_size, self.vocab_model.size, 'nestrov')
        else:
            W_in, W_context = self.load_model(saved_model)
        self.params = [W_in, W_context]
        self.reg_params = [W_in, W_context]
        assert isinstance(self.reg, float)
        assert isinstance(self.noise_sample_size, int) or self.noise_sample_size is None
        assert isinstance(self.noise_dist, np.ndarray) or self.noise_dist is None
        assert isinstance(self.vocab_model, Vocab)

    def set_optimizer(self, optimizer):
        if optimizer  == 'adam':
            self._update = adam
        elif optimizer == 'adagrad':
            self._update = adagrad 
        elif optimizer == 'momentum':
            self._update = momentum
        elif optimizer == 'sgd':
            self._update = sgd
        elif optimizer == 'rms_clipped':
            self._update = rmsprop_clipped
        elif optimizer == 'sgd_clipped':
            self._update = sgd_clipped
        elif optimizer == 'rms':
            self._update = rmsprop
        else:
            raise NotImplementedError("dont know what update this is...")

    def save_word_vecs(self, save_path):
        w = codecs.open(save_path, 'w', 'utf-8')
        win, wout = self.get_params()
        for voc_id, voc in self.vocab_model.id2voc.items():
            w.write(voc + ' ' + ' '.join(map(str,win[voc_id,:].tolist())) + '\n')
        w.flush()
        w.close()

    def load_model(self, load_path):
        _params = [floatX(np.asarray(i)) for i in json.loads(open(load_path, 'r').read())]
        W_in = theano.shared(floatX(_params[0]), name='W_in')
        W_context = theano.shared(floatX(_params[1]), name='W_context')
        return W_in, W_context

    def save_model(self, save_path):
        _params = json.dumps([i.get_value().tolist() for i in self.params])
        f = open(save_path, 'w')
        f.write(_params)
        f.flush()
        f.close()
        return _params

    def __xent_loss__(self, O, Y):
        return T.nnet.categorical_crossentropy(O, Y)

    def __nce_loss__(self, O, Y, N):
        # w is the next word in the training data
        pw = O[np.arange(0, self.batch_size), Y]
        qw = self.noise_dist[Y]
        # wb is the noise word in the noise samples
        pwb = T.take(O, N) # (noise_sample_size, )
        qwb = T.take(self.noise_dist, N) # (noise_sample_size, )

        # P(D = 1 | c, w)
        pd1 = pw / (pw + self.noise_sample_size * qw) # (batch_size, )
        # P(D = 0 | c, wb)
        pd0 = (self.noise_sample_size * qwb) / (pwb + self.noise_sample_size * qwb) # (noise_sample_size, )

        return T.sum(T.log(pd1) + T.sum(T.log(pd0))) # scalar

    def get_params(self,):
        return self.__params__()

class SkipGram(BaseModel):
    def __init__(self, vocab_model, batch_size, embed_size, saved_model = None, noise_sample_size = None, noise_dist = None, reg = 0.0, optimize = 'rms'):
        BaseModel.__init__(self, 
                vocab_model = vocab_model,
                batch_size = batch_size, 
                embed_size = embed_size, 
                noise_sample_size = noise_sample_size, 
                noise_dist = noise_dist,
                reg = reg, 
                optimize = optimize)
        self.__computation_graph__()

    def __computation_graph__(self,):
        lr = T.scalar('lr', dtype=theano.config.floatX) #for learning rates
        X = T.lvector('X') #(batch_size)
        Y = T.lvector('Y') #(batch_size)
        #exception_verbosity=highN= T.lvector('N') #(noise_size)

        W_in = self.params[0] 
        W_context = self.params[1] 

        y_out_unnormalized = W_in[X].dot(W_context) #(batch_size, vocab_model.size)
        y_pred = T.nnet.softmax(y_out_unnormalized)  #(batch_size, vocab_model.size)
        model_losses = self.__xent_loss__(y_pred, Y) #T.nnet.categorical_crossentropy(y_pred, Y) #(batch_size,)
        model_loss = model_losses.mean()
        reg_loss = 0.
        for rp in self.reg_params:
            reg_loss += T.sum(T.sqr(rp))
        loss = model_loss + (self.reg * reg_loss)

        #optimization methods
        self.__y_pred__ = theano.function(inputs = [X], outputs = y_pred)
        self.__model_losses__ = theano.function(inputs = [X, Y], outputs = model_losses)
        self.__loss__ = theano.function(inputs = [X, Y], outputs = [loss, model_loss, reg_loss])
        self.__params__ = theano.function(inputs = [], outputs = self.params) 
        self.__do_update__ = theano.function(inputs = [lr, X, Y], outputs = [loss, model_loss, reg_loss], updates = self._update(loss, self.params, lr)) 

    def loss(self, batch_size, X, Y):
        ave_dev_loss = []
        t_idx = np.arange(X.shape[0])
        batches = np.array_split(t_idx, X.shape[0] / batch_size)
        for b_idx, batch_idxs in enumerate(batches):
            _dev_batch_loss = self.__loss__(X[batch_idxs], Y[batch_idxs])
            ave_dev_loss.append(_dev_batch_loss[0])
        return np.mean(ave_dev_loss)

    def fit(self, batch_size, learning_rate, X, Y):
        t_idx = np.arange(X.shape[0])
        np.random.shuffle(t_idx)
        batches = np.array_split(t_idx, X.shape[0] / batch_size)
        ave_loss = []
        for b_idx, batch_idxs in enumerate(batches):
            _batch_loss  = self.__do_update__(learning_rate, X[batch_idxs], Y[batch_idxs])
            ave_loss.append(_batch_loss[0])
        return np.mean(ave_loss)


class CBOW(BaseModel):
    def __init__(self, vocab_model, batch_size, context_size, embed_size, saved_model = None, noise_sample_size = None, noise_dist = None, reg = 0.0, optimize ='rms'):
        BaseModel.__init__(self, 
                vocab_model = vocab_model,
                batch_size = batch_size, 
                embed_size = embed_size, 
                noise_sample_size = noise_sample_size, 
                noise_dist = noise_dist,
                reg = reg, 
                optimize = optimize)
        self.context_size = context_size
        self.__computation_graph__()

    def __computation_graph__(self,):
        lr = T.scalar('lr', dtype=theano.config.floatX) #for learning rates
        X = T.lmatrix('X') #(batch_size, 2 * context_size)
        Y = T.lvector('Y') #(batch_size)
        #exception_verbosity=highN= T.lvector('N') #(noise_size)

        W_in = self.params[0] 
        W_context = self.params[1] 

        w_in_x_sum = W_in[X,:].sum(axis = 1) #T.sum(w_in_x, axis = 1) #(batch_size, hidden_state) # selects the rows in W_in, per row in X, then sums those rows
        y_out_unnormalized = w_in_x_sum.dot(W_context) #(batch_size, vocab_model.size)
        y_pred = T.nnet.softmax(y_out_unnormalized)  #(batch_size, vocab_model.size)
        model_losses = self.__xent_loss__(y_pred, Y)
        model_loss = model_losses.mean()
        #model_losses = self.nce_loss(y_pred, Y, N) #T.nnet.categorical_crossentropy(y_pred, Y).mean() #scalar
        #model_loss = self.loss(y_pred, Y)
        reg_loss = 0.
        for rp in self.reg_params:
            reg_loss += T.sum(T.sqr(rp))
        loss = model_loss + (self.reg * reg_loss)
        #optimization methods
        self.__y_pred__ = theano.function(inputs = [X], outputs = y_pred)
        self.__model_losses__ = theano.function(inputs = [X, Y], outputs = model_losses)
        self.__loss__ = theano.function(inputs = [X, Y], outputs = [loss, model_loss, reg_loss])
        self.__params__ = theano.function(inputs = [], outputs = self.params) 
        self.__do_update__ = theano.function(inputs = [lr, X, Y], outputs = [loss, model_loss, reg_loss], updates = self._update(loss, self.params, lr)) 

    def loss(self, batch_size, X, Y):
        ave_dev_loss = []
        t_idx = np.arange(X.shape[0])
        batches = np.array_split(t_idx, X.shape[0] / batch_size)
        for b_idx, batch_idxs in enumerate(batches):
            _dev_batch_loss = self.__loss__(X[batch_idxs,:], Y[batch_idxs])
            ave_dev_loss.append(_dev_batch_loss[0])
        return np.mean(ave_dev_loss)

    def fit(self, batch_size, learning_rate, X, Y):
        t_idx = np.arange(X.shape[0])
        np.random.shuffle(t_idx)
        batches = np.array_split(t_idx, X.shape[0] / batch_size)
        ave_loss = []
        for b_idx, batch_idxs in enumerate(batches):
            _batch_loss  = self.__do_update__(learning_rate, X[batch_idxs,:], Y[batch_idxs])
            ave_loss.append(_batch_loss[0])
        return np.mean(ave_loss)

class Vocab(object):
    def __init__(self, vocab_file):
        self.vocab_file = vocab_file
        self.voc2id = {}
        self.id2voc = {}
        self.voc2count = {}
        self.size = -1
        self.voc_dist = None
        self.load(self.vocab_file)

    def load(self, vf):
        for line in codecs.open(vf, 'r', 'utf-8').readlines():
            voc_id, voc, voc_count = line.strip().split()
            self.voc2id[voc] = int(voc_id)
            self.voc2count[voc] = int(voc_count)
            self.id2voc[int(voc_id)] = voc
        self.voc_dist = np.zeros(len(self.voc2id),)
        self.size = len(self.voc2id) 
        for v,vid in self.voc2id.items():
            self.voc_dist[vid] = self.voc2count[v]
        self.voc_dist = self.voc_dist / self.voc_dist.sum()
