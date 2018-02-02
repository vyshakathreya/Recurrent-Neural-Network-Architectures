# Recurrent-Neural-Network-Architectures

Dataset used is MIT/TIDIGITS corpus (Leonard and Doddington, 1993)

Aim is to
1. Extract words represented by feature sequences of variable length
2. Design a simple Voice Activity Detector based on Gaussian Mixture Models
3. Evaluate network using Tensorboard scalars
4. Iterate through dataset in minibatches
5. Experiment with recurrent network architectures comprising of LSTM, GRU and simpleRNN

Results:
1. Using VAD reduced errors by 14%
2. LSTM with Dropout having an architecture of 
LSTM: 128 nodes,
(Dropout: 0.25), 
LSTM: 128 nodes, 
(Dropout: 0.25), 
LSTM: 128 nodes, 
(Dropout: 0.25), 
LSTM: 128 nodes,
11 output nodes
gave the best accuracy
