import sys
import numpy as np
import math
from time import time

sys.path.append('..')
import ntm
import tasks

np.random.seed(22334)

model = ntm.NTM (
    input_size    = 8, 
    output_size   = 8, 
    mem_size      = 128, 
    mem_width     = 20, 
    hidden_sizes  = [100],
    max_epochs    = 100000, 
    momentum      = 0.9, 
    learning_rate = 1e-5 ,
    grad_clip     = [-10,10], 
    l2_norm       = 1e-4 
)


t0 = time()

train_cost = None

alpha = 0.95

for counter in xrange(model.max_epochs):

    length = np.random.randint(1,21)
    i,o = tasks.copy(8,length)
    if train_cost == None: train_cost = model.train(i,o)
    else: train_cost = alpha * train_cost + (1 - alpha) * model.train(i,o)
    
    train_cost = train_cost / ((o.shape[0]*o.shape[1]* 1.0))
    
#     if counter % 1000 == 0:
#         print "round:", counter,"length",length, "training_cost:", train_cost
    
    if math.isnan(train_cost):
        print "meet nan"
        break


    # calculate bits per character error with length np.random.randint(1,101) sequence
    length2 = np.random.randint(21, 101)
    i,o = tasks.copy(8,length2)
    valid_cost = model.predict_cost(i,o) / (o.shape[0]*o.shape[1]* 1.0)
    
    if counter % 1000 == 0:
        print "round:", counter,"length",length2, "validation_cost:", valid_cost, "time:", time()-t0
        t0 = time()
        
    model.train_his.append({"round":counter, "cost": valid_cost})

    if valid_cost < model.best_valid_cost:
        model.save('./model.pkl')
        model.best_valid_cost = valid_cost
#         print "~~~save validation~~~", counter, "length: ",length2,  valid_cost

    sys.stdout.flush()



