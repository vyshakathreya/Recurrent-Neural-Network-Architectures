
import math
import time

from keras.callbacks import TensorBoard
from keras.utils import np_utils
from keras import metrics


import keras.backend as K

from dsp.utils import Timer
from .batchgenerator import PaddedBatchGenerator
from .histories import ErrorHistory, LossHistory


def train_and_evaluate(examples, labels, train_idx, test_idx, 
                              model, batch_size=100, epochs=100, 
                              name="model"):
    """train_and_evaluate(examples, labels, train_idx, test_idx,
            model, batch_size, epochs)
            
    Given:
        examples - List of examples in column major order
            (# of rows is feature dim)
        labels - list of corresponding labels
        train_idx - list of indices of examples and labels to be learned
        test_idx - list of indices of examples and labels of which
            the system should be tested.
        model- Keras model to learn
            e.g. result of buildmodels.build_model()
    Optional arguments
        batch_size - size of minibatch
        epochs - # of epochs to compute
        name - model name
        
    Returns error rate, model, and loss history over training
    """

    # Convert labels to a one-hot vector
    # https://keras.io/utils/#to_categorical
    onehotlabels = np_utils.to_categorical(labels)

    error = ErrorHistory()
    loss = LossHistory()
  
    model.compile(optimizer = "Adam",
                  #loss = seq_loss,  
                  loss = "categorical_crossentropy",
                  metrics = [metrics.categorical_accuracy])
    
    model.summary()  # display
    
    examplesN = len(train_idx)  # Number training examples
    # Approx # of times fit generator must be called to complete an epoch
    steps = int(math.ceil(examplesN / batch_size))  
    
    # for debugging
    tensorboard = TensorBoard(
        log_dir="logs/{}".format(time.strftime('%d%b-%H%M')),            
        histogram_freq=0,
        write_graph=True,
        write_grads=True
        )

    # Find the longest sequence so that we can zero-pad both
    # the training and test data to be sequences of the same
    # size with appropriate zero-padding.
    longest = PaddedBatchGenerator.longest_sequence(examples)
    
    # Evaluation data
    # Reformat examples into zero-padded numpy array
    # Let PaddedBatchGenerator do the work, generating
    # one single "batch" for the epic.  
    #
    # model.fit will take care of breaking it up into minibatches    
    testgen = PaddedBatchGenerator(examples[test_idx],
                                   onehotlabels[test_idx],
                                   batch_size = len(test_idx),
                                   pad_length = longest,
                                   flatten=True)    
    (testexamples, testlabels) = next(testgen)

    # Training data (similar)
    traingen = PaddedBatchGenerator(examples[train_idx], 
                                     onehotlabels[train_idx],
                                     batch_size=len(train_idx),
                                     pad_length = longest,
                                     flatten=True)    
    (trainexamples, trainlabels) = next(traingen)
    
    # train the net
    model.fit(trainexamples, trainlabels,
              validation_data=(testexamples, testlabels),
              epochs=epochs, callbacks=[loss, tensorboard])
    
    print("Training loss %s"%(["%f"%(loss) for loss in loss.losses]))
    
    result = model.evaluate(testexamples, testlabels,
                            verbose=False)

    return (1 - result[1], model, loss) 

    
    


