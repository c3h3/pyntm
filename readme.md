# Neural Turing Machine 

## NTM in theano

Example for running it:
````
cd copytask
python train.py
````

## The main difference compare with fumin/ntm:
Shifting method:
  - This version use softmax to implement shift
  - fumin use the special techniques which provide by the paper

## Future:
- Due to the numerical instability problem, we can only use a small learning rate like 1e-5 in training process, and it cause the training process slower
- The main problem maybe exist when we calculate the sharp
- Find out the numerical instability problem
- Check the scan op in theano, maybe we have other advanced techniques to accelerate our training

## The bug in Shawtan/neural turing machine:
- Shawtan map the output of hidden layer to a tanh function, this is the main reason that can't train with random length of sequence 

