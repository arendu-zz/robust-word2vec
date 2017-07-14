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
VERBOSE = False

if theano.config.floatX == 'float32':
    intX = np.int32
    floatX = np.float32
else:
    intX = np.int64
    floatX = np.float64

def _get_weights(name, shape1, shape2, init='rand'):
    if init == 'rand':
        x = np.random.rand(shape1, shape2) 
    elif init == 'normal':
        x = np.random.normal(0., 0.01, (shape1, shape2))
    elif init == 'nestrov':
        x = np.random.uniform(-np.sqrt(1. / shape2), np.sqrt(1. / shape2), (shape1, shape2))
    else:
        raise NotImplementedError("don't know how to initialize the weight matrix")
    return theano.shared(floatX(x), name)

class BaseModel(object):
    __metaclass__ = ABCMeta

    def __init__(self, vocab_model, batch_size, embed_size, saved_model = None, noise_sample_size = 0, noise_dist = None, reg = 0.0, optimize ='rms'):
        self.vocab_model = vocab_model
        self.batch_size = batch_size
        self.embed_size = embed_size
        if noise_dist is not None:
            self.noise_dist_T = theano.shared(floatX(noise_dist), 'N_DIST')
            self.noise_dist = noise_dist
        else:
            self.noise_dist_T = theano.shared(floatX(np.ones(self.vocab_model.size,)), 'N_DIST')
            self.noise_dist = None
        self.noise_sample_size = noise_sample_size
        self.reg = reg
        self._eps = 1e-8
        self.params = []
        self.reg_params = []
        self.set_optimizer(optimize)
        if saved_model is None:
            W_in = _get_weights('W_IN', self.vocab_model.size, self.embed_size, 'normal')
            W_context = _get_weights('W_CONTEXT', self.embed_size, self.vocab_model.size, 'normal')
        else:
            W_in, W_context = self.load_model(saved_model)
        self.params = [W_in, W_context]
        self.reg_params = [W_in, W_context]
        assert isinstance(self.reg, float)
        assert isinstance(self.noise_sample_size, int)
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

    def __xent_loss__(self, Y_pred, Y):
        Y_pred = T.clip(Y_pred, floatX(self._eps), floatX(1.0 - self._eps))
        return T.nnet.categorical_crossentropy(Y_pred, Y) #(batch_size)

    def __nce_loss__(self, Y_out, Y, N):
        p_w = Y_out[T.arange(Y_out.shape[0]),Y] #(batch_size,)
        q_w = self.noise_dist_T[Y] #(batch_size,)
        p_c1_w = p_w / (p_w + self.noise_sample_size * q_w) #(batch_size,)  
        p_c1_w = T.clip(p_c1_w, floatX(self._eps), floatX(1.0 - self._eps))
        log_p_c1_w = T.log(p_c1_w) #(batch_size,)

        p_wn = Y_out[:, N] #(batch_size, noise_sample_size)
        q_wn = self.noise_dist_T[N] #(noise_sample_size,)
        p_c1_wn = p_wn / (p_wn + self.noise_sample_size * q_wn) #(batch_size, noise_sample_size)
        p_c0_wn = floatX(1.) - p_c1_wn
        p_c0_wn = T.clip(p_c0_wn, floatX(self._eps), floatX(1.0 - self._eps))
        sum_k_log_p_c0_wn = T.log(p_c0_wn).sum(axis = 1) #(batch_size,)
        return -(log_p_c1_w + sum_k_log_p_c0_wn) #(batch_size,)

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
        N = T.lvector('N') #(noise_size)

        W_in = self.params[0] 
        W_context = self.params[1] 

        y_out = W_in[X].dot(W_context) #(batch_size, vocab_model.size)
        self.__params__ = theano.function(inputs = [], outputs = [T.as_tensor_variable(p) for p in self.params]) 
        if self.noise_sample_size > 0:
            model_losses_nce = self.__nce_loss__(y_out, Y, N) #(batch_size,)
            loss_nce = model_losses_nce.mean()
            self.__loss_nce__ = theano.function(inputs = [X, Y, N], outputs = loss_nce)
            self.__do_update_nce__ = theano.function(inputs = [lr, X, Y, N], outputs = loss_nce, updates = self._update(loss_nce, self.params, lr)) 
        else:
            y_pred = T.nnet.softmax(y_out)  #(batch_size, vocab_model.size)
            self.__y_pred__ = theano.function(inputs = [X], outputs = y_pred)
            model_losses = self.__xent_loss__(y_pred, Y) #(batch_size,)
            loss = model_losses.mean()
            self.__loss__ = theano.function(inputs = [X, Y], outputs = loss)
            self.__do_update__ = theano.function(inputs = [lr, X, Y], outputs = loss, updates = self._update(loss, self.params, lr)) 

    def loss(self, batch_size, X, Y):
        ave_dev_loss = 0. 
        t_idx = np.arange(X.shape[0])
        batches = np.array_split(t_idx, X.shape[0] / batch_size)
        for b_idx, batch_idxs in enumerate(batches):
            if self.noise_dist is not None and self.noise_sample_size > 0:
                N = np.random.choice(self.vocab_model.size, self.noise_sample_size, p = self.noise_dist, replace = False)
                _batch_loss  = self.__loss_nce__(X[batch_idxs], Y[batch_idxs], N)
            else:
                _batch_loss  = self.__loss__(X[batch_idxs], Y[batch_idxs])
            if np.isnan(_batch_loss):
                raise Exception("loss is nan!")
            if VERBOSE:
                print _batch_loss.eval()
            ave_dev_loss += _batch_loss
        return ave_dev_loss / len(batches)

    def fit(self, batch_size, learning_rate, X, Y, nce = False):
        t_idx = np.arange(X.shape[0])
        np.random.shuffle(t_idx)
        batches = np.array_split(t_idx, X.shape[0] / batch_size)
        ave_loss = 0.
        for b_idx, batch_idxs in enumerate(batches):
            if self.noise_dist is not None and self.noise_sample_size > 0:
                N = np.random.choice(self.vocab_model.size, self.noise_sample_size, p = self.noise_dist, replace = False)
                _batch_loss  = self.__do_update_nce__(learning_rate, X[batch_idxs], Y[batch_idxs], N)
            else:
                _batch_loss  = self.__do_update__(learning_rate, X[batch_idxs], Y[batch_idxs])
            if np.isnan(_batch_loss):
                raise Exception("loss is nan!")
            if VERBOSE:
                print _batch_loss.eval()
            ave_loss += _batch_loss
        return ave_loss / len(batches)


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
        N = T.lvector('N') #(noise_size)

        W_in = self.params[0] 
        W_context = self.params[1] 

        w_in_x_sum = W_in[X,:].mean(axis = 1) #(batch_size, hidden_state) # selects the rows in W_in, per row in X, then means those rows
        #the paper says you sum them but gensim takes the means(gensim/gensim/models/word2vec.py -> def train_batch_cbow -> line~195 (dont think it changes the math)
        y_out = w_in_x_sum.dot(W_context) #(batch_size, vocab_model.size)
        self.__params__ = theano.function(inputs = [], outputs = [T.as_tensor_variable(p) for p in self.params]) 
        if self.noise_sample_size > 0:
            model_losses_nce = self.__nce_loss__(y_out, Y, N) #T.nnet.categorical_crossentropy(y_pred, Y) #(batch_size,)
            loss_nce = model_losses_nce.mean()
            self.__loss_nce__ = theano.function(inputs = [X, Y, N], outputs = loss_nce)
            self.__do_update_nce__ = theano.function(inputs = [lr, X, Y, N], outputs = loss_nce, updates = self._update(loss_nce, self.params, lr)) 
        else:
            y_pred = T.nnet.softmax(y_out)  #(batch_size, vocab_model.size)
            self.__y_pred__ = theano.function(inputs = [X], outputs = y_pred)
            model_losses = self.__xent_loss__(y_pred, Y) #T.nnet.categorical_crossentropy(y_pred, Y) #(batch_size,)
            loss = model_losses.mean()
            self.__loss__ = theano.function(inputs = [X, Y], outputs = loss)
            self.__do_update__ = theano.function(inputs = [lr, X, Y], outputs = loss, updates = self._update(loss, self.params, lr)) 

    def loss(self, batch_size, X, Y):
        ave_dev_loss = []
        t_idx = np.arange(X.shape[0])
        batches = np.array_split(t_idx, X.shape[0] / batch_size)
        for b_idx, batch_idxs in enumerate(batches):
            if self.noise_dist is not None and self.noise_sample_size > 0:
                N = np.random.choice(self.vocab_model.size, self.noise_sample_size, p = self.noise_dist, replace = False)
                _batch_loss  = self.__loss_nce__(X[batch_idxs,:], Y[batch_idxs], N)
            else:
                _batch_loss = self.__loss__(X[batch_idxs,:], Y[batch_idxs])
            if np.isnan(_batch_loss):
                raise Exception("loss is nan!")
            if VERBOSE:
                print _batch_loss.eval()
            ave_dev_loss += _batch_loss
        return ave_dev_loss / len(batches)

    def fit(self, batch_size, learning_rate, X, Y):
        t_idx = np.arange(X.shape[0])
        np.random.shuffle(t_idx)
        batches = np.array_split(t_idx, X.shape[0] / batch_size)
        ave_loss = 0.
        for b_idx, batch_idxs in enumerate(batches):
            if self.noise_dist is not None and self.noise_sample_size > 0:
                N = np.random.choice(self.vocab_model.size, self.noise_sample_size, p = self.noise_dist, replace = False)
                _batch_loss  = self.__do_update_nce__(learning_rate, X[batch_idxs,:], Y[batch_idxs], N)
            else:
                _batch_loss  = self.__do_update__(learning_rate, X[batch_idxs,:], Y[batch_idxs])
            if np.isnan(_batch_loss):
                raise Exception("loss is nan!")
            if VERBOSE:
                print _batch_loss.eval()
            ave_loss += _batch_loss
        return ave_loss / len(batches)

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
