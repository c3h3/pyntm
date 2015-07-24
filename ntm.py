import theano
import theano.tensor as T
import numpy         as np
from theano_toolkit.parameters import Parameters
from theano_toolkit import updates
from theano_toolkit import utils as U

import sys
import math
import cPickle
import sys
sys.setrecursionlimit(100000)

import controller
import model


class NTM():
    
    def __init__(self, 
                 input_size, output_size, mem_size, mem_width, hidden_sizes,
                 max_epochs, momentum, learning_rate ,grad_clip, l2_norm):
        
        self.input_size = input_size
        self.output_size = output_size
        self.mem_size = mem_size
        self.mem_width = mem_width
        self.hidden_sizes = hidden_sizes
        self.max_epochs = max_epochs
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.l2_norm = l2_norm
        
        self.best_train_cost = np.inf
        self.best_valid_cost = np.inf
        #self.train = None
        #self.cost = None
        
        self.train_his = []
        
        P = Parameters()
        ctrl = controller.build( P, self.input_size, self.output_size, self.mem_size, self.mem_width, self.hidden_sizes)
        predict = model.build( P, self.mem_size, self.mem_width, self.hidden_sizes[-1], ctrl)

        input_seq = T.matrix('input_sequence')
        output_seq = T.matrix('output_sequence')
        
        [M_curr,weights,output] = predict(input_seq)
        # output_seq_pred = seqs[-1]
        
        cross_entropy = T.sum(T.nnet.binary_crossentropy(5e-6 + (1 - 2*5e-6)*output, output_seq),axis=1)
        
        self.params = P.values()
        
        l2 = T.sum(0)
        for p in self.params:
            l2 = l2 + (p ** 2).sum()
            
        cost = T.sum(cross_entropy) + self.l2_norm * l2
    #     cost = T.sum(cross_entropy) + 1e-3*l2
        
        grads  = [ T.clip(g, grad_clip[0], grad_clip[1]) for g in T.grad(cost, wrt=self.params) ]
    #     grads  = [ T.clip(g,-100,100) for g in T.grad(cost,wrt=params) ]
    #     grads  = [ T.clip(g,1e-9, 0.2) for g in T.grad(cost,wrt=params) ]

        self.train = theano.function(
                inputs=[input_seq,output_seq],
                outputs=cost,
    #             updates=updates.adadelta(params,grads)
                updates = updates.rmsprop(self.params, grads, momentum=self.momentum, learning_rate=self.learning_rate )
            )
        
        self.predict_cost = theano.function(
            inputs=[input_seq,output_seq],
            outputs= cost
        )
        
        self.predict = theano.function(
            inputs=[input_seq],
            outputs= [ weights, output]
        )
    
    def save(self, filename):
        cPickle.dump( self , open( filename , "wb"))

