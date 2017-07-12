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
        self.pad = theano.shared(floatX(np.zeros((1, self.embed_size))), 'Pad')
        if saved_model is None:
            W_in = _get_weights('W_IN', self.vocab_model.size, self.embed_size, 'nestrov')
            W_context = _get_weights('W_CONTEXT', self.embed_size, self.vocab_model.size, 'nestrov')
        else:
            W_in, W_context = self.load_model(saved_model)
        self.params = [W_in, W_context]
        self.reg_params = [W_in, W_context]
        if optimize  == 'adam':
            self._update = adam
        elif optimize == 'adagrad':
            self._update = adagrad 
        elif optimize == 'momentum':
            self._update = momentum
        elif optimize == 'sgd':
            self._update = sgd
        elif optimize == 'rms_clipped':
            self._update = rmsprop_clipped
        elif optimize == 'sgd_clipped':
            self._update = sgd_clipped
        elif optimize == 'rms':
            self._update = rmsprop
        else:
            raise NotImplementedError("dont know what update this is...")
        assert isinstance(self.reg, float)
        assert isinstance(self.noise_sample_size, int) or self.noise_sample_size is None
        assert isinstance(self.noise_dist, np.ndarray) or self.noise_dist is None
        assert isinstance(self.vocab_model, Vocab)
        #self.__make_cosine_sim_graph__()
    """
    def __make_cosine_sim_graph__(self,):
        #cosine similarity methods
        W_in = self.params[0]
        numerator = W_in.dot(W_in.T) #(self.vocab_model.size, self.vocab_model.size)
        l2_norms = T.sqrt(T.sum(W_in ** 2, axis = 1)) #(self.vocab_model.size)
        demoninator = l2_norms[:, np.newaxis].dot(l2_norms[np.newaxis, :])
        self.cosine_sim = numerator / demoninator
    """

    def loss(self, O, Y):
        return T.nnet.categorical_crossentropy(O, Y)

    def nce_loss(self, O, Y, N):
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

    @abstractmethod
    def do_update(self, *params):
        pass

    @abstractmethod
    def get_loss(self, *params):
        pass

    @abstractmethod
    def get_params(self,):
        pass

    @abstractmethod
    def save_model(self,):
        pass

    @abstractmethod
    def load_model(self, load_path):
        pass

    def fit(self, epochs, batche_size, X, Y, N = None, noise_dist = None, noise_sample_size = None):
        raise NotImplementedError("training routine should be inside base class?")

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
        self.make_network()

    def make_network(self,):
        lr = T.scalar('lr', dtype=theano.config.floatX) #for learning rates
        X = T.lmatrix('X') #(batch_size, 2 * context_size)
        Y = T.lvector('Y') #(batch_size)
        #exception_verbosity=highN= T.lvector('N') #(noise_size)

        #W_in = self.params[0] 
        padded_W_in = T.concatenate((self.pad, self.params[0]))
        #W_context = self.params[1] 
        padded_W_context = T.concatenate((self.pad.T, self.params[1]), axis = 1)

        w_in_x_sum = padded_W_in[X,:].sum(axis = 1) #T.sum(w_in_x, axis = 1) #(batch_size, hidden_state) # selects the rows in W_in, per row in X, then sums those rows
        y_out_unnormalized = w_in_x_sum.dot(padded_W_context) #(batch_size, vocab_model.size)
        y_pred = T.nnet.softmax(y_out_unnormalized)  #(batch_size, vocab_model.size)
        model_losses = T.nnet.categorical_crossentropy(y_pred, Y)
        model_loss = model_losses.mean()
        #model_loss = self.nce_loss(y_pred, Y, N) #T.nnet.categorical_crossentropy(y_pred, Y).mean() #scalar
        #model_loss = self.loss(y_pred, Y)
        reg_loss = 0.
        for rp in self.reg_params:
            reg_loss += T.sum(T.sqr(rp))
        loss = model_loss + (self.reg * reg_loss)
        #optimization methods
        self.get_y_unnorm = theano.function(inputs = [X], outputs = y_out_unnormalized)
        self.get_y_pred = theano.function(inputs = [X], outputs = y_pred)
        self.get_model_losses_t = theano.function(inputs = [X, Y], outputs = model_losses)
        self.get_loss_t = theano.function(inputs = [X, Y], outputs = [loss, model_loss, reg_loss])
        self.get_params_t = theano.function(inputs = [], outputs = self.params) 
        self.do_update_t = theano.function(inputs = [lr, X, Y], outputs = [loss, model_loss, reg_loss], updates = self._update(loss, self.params, lr)) 

        #V1 = T.lscalar('V1')
        #V2 = T.lscalar('V2')
        #self.cosine_similarity = theano.function(inputs = [], outputs = self.cosine_sim) 
        #self.cosine_similarity_cell = theano.function(inputs = [V1, V2], outputs = self.cosine_sim[V1, V2]) 
        #self.cosine_similarity_row = theano.function(inputs = [V1], outputs = self.cosine_sim[V1,:]) 

    def get_loss(self, *params):
        _X, _Y = params
        return self.get_loss_t(_X, _Y),

    def do_update(self, *params):
        _lr, _X, _Y = params
        return self.do_update_t(_lr, _X, _Y)

    def get_params(self,):
        return self.get_params_t()

    def save_word_vecs(self, save_path):
        w = codecs.open(save_path, 'w', 'utf-8')
        win, wout = self.get_params()
        for voc_id, voc in self.vocab_model.id2voc.items():
            if voc_id == 0:
                pass
            else:
                w.write(voc + ' ' + np.array2string(win[voc_id -1, :], separator = ' ', max_line_width = np.inf)[1:-1].strip() + '\n')
        w.flush()
        w.close()

    def load_model(self, load_path):
        sys.stderr.write('loading model' + load_path + '\n')
        _params = [floatX(np.asarray(i)) for i in json.loads(open(load_path, 'r').read())]
        W_in = theano.shared(floatX(_params[0]), name='W_in')
        W_context = theano.shared(floatX(_params[1]), name='W_context')
        return W_in, W_context

    def save_model(self, save_path):
        sys.stderr.write('saving model' + save_path + '\n')
        _params = json.dumps([i.get_value().tolist() for i in self.params])
        f = open(save_path, 'w')
        f.write(_params)
        f.flush()
        f.close()
        return _params


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
        self.size = len(self.voc2id) - 1 #-1 because of the PAD symbol 
        for v,vid in self.voc2id.items():
            self.voc_dist[vid] = self.voc2count[v]
        self.voc_dist = self.voc_dist / self.voc_dist.sum()
