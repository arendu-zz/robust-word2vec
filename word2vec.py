import sys
import codecs
import argparse
import numpy as np
import theano
import theano.tensor as T
from optimizers import adam, sgd, sgd_clipped, rmsprop
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

    def __init__(self, batch_size, vocab_size, embed_size, noise_sample_size = None, noise_dist = None, reg = 0.01, optimize ='sgd'):
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.noise_dist = noise_dist
        self.noise_sample_size = noise_sample_size
        self.reg = reg
        self._eps = 1e-8
        self.params = []
        self.reg_params = []
        if optimize  == 'adam':
            self._update = adam
        elif optimize == 'sgd':
            self._update = sgd
        elif optimize == 'sgd_clipped':
            self._update = sgd_clipped
        elif optimize == 'rms':
            self._update = rmsprop
        else:
            raise NotImplementedError("dont know what update this is...")

    @abstractmethod
    def do_update(self, *params):
        pass

    @abstractmethod
    def get_loss(self, *params):
        pass

    @abstractmethod
    def get_params(self,):
        pass

    def loss(self, O, Y):
        return T.nnet.categorical_crossentropy(O, Y).mean()

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

    def save_weights(self, save_path):
        _params = json.dumps([i.tolist() for i in self.get_params()])
        f = open(save_path, 'w')
        f.write(_params)
        f.flush()
        f.close()
        return _params

class CBOW(BaseModel):
    def __init__(self, batch_size, vocab_size, context_size, embed_size, noise_sample_size = None, reg = 0.01, optimize ='sgd'):
        BaseModel.__init__(self, batch_size, vocab_size, embed_size, noise_sample_size, reg, optimize)
        self.context_size = context_size
        W_in = _get_weights('W_IN', self.vocab_size, self.embed_size)
        W_context = _get_weights('W_CONTEXT', self.embed_size, self.vocab_size)
        self.params = [W_in, W_context]
        self.reg_params = [W_in, W_context]
        self.make_network()

    def make_network(self,):
        lr = T.scalar('lr', dtype=theano.config.floatX) #for learning rates
        X = T.lmatrix('X') #(batch_size, 2 * context_size)
        Y = T.lvector('Y') #(batch_size)
        #N= T.lvector('N') #(noise_size)
        W_in = self.params[0]
        W_context = self.params[1]
        w_in_x_sum = W_in[X,:].sum(axis = 1) #T.sum(w_in_x, axis = 1) #(batch_size, hidden_state) # selects the rows in W_in, per row in X, then sums those rows
        y_out_unnormalized = w_in_x_sum.dot(W_context) #(batch_size, vocab_size)
        y_pred = T.nnet.softmax(y_out_unnormalized)  #(batch_size, vocab_size)
        #model_loss = self.nce_loss(y_pred, Y, N) #T.nnet.categorical_crossentropy(y_pred, Y).mean() #scalar
        model_loss = self.loss(y_pred, Y)
        reg_loss = 0.
        for rp in self.reg_params:
            reg_loss += T.sum(T.sqr(rp))
        loss = model_loss + (self.reg * reg_loss)
        self.get_model_loss = theano.function(inputs = [X, Y], outputs = model_loss)
        self.get_reg_loss = theano.function(inputs = [], outputs = reg_loss)
        self.get_out = theano.function(inputs = [X], outputs = y_out_unnormalized)
        self.get_loss_t = theano.function(inputs = [X, Y], outputs = loss)
        self.get_params_t = theano.function(inputs = [], outputs = [T.as_tensor_variable(p) for p in self.params]) 
        self.do_update_t = theano.function(inputs = [lr, X, Y], outputs = loss, updates = self._update(loss, self.params, lr)) 

    def get_loss(self, *params):
        _X, _Y = params
        return self.get_loss_t(_X, _Y)

    def do_update(self, *params):
        _lr, _X, _Y = params
        return self.do_update_t(_lr, _X, _Y)

    def get_params(self,):
        return self.get_params_t()

class Vocab(object):
    def __init__(self, vocab_file):
        self.vocab_file = vocab_file
        self.voc2id = {}
        self.voc2count = {}
        self.vocab_size = -1
        self.voc_dist = None
        self.load(self.vocab_file)

    def load(self, vf):
        for line in codecs.open(vf, 'r', 'utf-8').readlines():
            voc_id, voc, voc_count = line.strip().split()
            self.voc2id[voc] = int(voc_id)
            self.voc2count[voc] = int(voc_count)
        self.voc_dist = np.zeros(len(self.voc2id),)
        self.vocab_size = len(self.voc2id)
        for v,vid in self.voc2id.items():
            self.voc_dist[vid] = self.voc2count[v]
        self.voc_dist = self.voc_dist / self.voc_dist.sum()


if __name__ == '__main__':
    opt= argparse.ArgumentParser(description="write program description here")
    #insert options here
    opt.add_argument('-v', action='store', dest='vocab_file', required = True)
    opt.add_argument('-t', action='store', dest='training_data', required = True)
    options = opt.parse_args()

    V = 100
    h = 5
    c = 3
    b = 10
    cbow = CBOW(b, V, c, h)
    x = np.random.randint(0, V, (b, c))
    yo = cbow.get_out(x)
    print yo.shape
