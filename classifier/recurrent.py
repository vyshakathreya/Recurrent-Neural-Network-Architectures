
import math
import time

import numpy as np

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
    """train_and_evaluate__model(examples, labels, train_idx, test_idx,
            model_spec, batch_size, epochs)
            
    Given:
        examples - List of examples in column major order
            (# of rows is feature dim)
        labels - list of corresponding labels
        train_idx - list of indices of examples and labels to be learned
        test_idx - list of indices of examples and labels of which
            the system should be tested.
        model_spec - Model specification, see feed_forward_model
            for details and example
    Optional arguments
        batch_size - size of minibatch
        epochs - # of epochs to compute
        name - model name
        
    Returns error rate, model, and loss history over training
    """

    # Convert labels to a one-hot vector
    # https://keras.io/utils/#to_categorical
    onehotlabels = np_utils.to_categorical(labels)

    # Get dimension of model
    dim = examples[0].shape[1]

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
        histogram_freq=1,
        write_graph=True,
        write_grads=True
        )

    
    # Evaluation data
    # Reformat examples into zero-padded numpy array
    # Let PaddedBatchGenerator do the work, generating
    # one single "batch."
    testgen = PaddedBatchGenerator(examples[test_idx],
                                   onehotlabels[test_idx],
                                   batch_size = len(test_idx))
    (testexamples, testlabels) = next(testgen)
    
    pad_batches_individually = False
    if pad_batches_individually:
        # Generator function to produce standardized length training
        # sequences for each batch
        generator = PaddedBatchGenerator(examples[train_idx], 
                                         onehotlabels[train_idx],
                                         batch_size=batch_size)
        model.fit_generator(generator, steps_per_epoch=steps,
                              epochs=epochs, callbacks=[loss, tensorboard],
                              validation_data=(testexamples, testlabels))
    else:
        generator = PaddedBatchGenerator(examples[train_idx], 
                                         onehotlabels[train_idx],
                                         batch_size=len(train_idx))
        (trainexamples, trainlabels) = next(generator)
        model.fit(trainexamples, trainlabels,
                  epochs=epochs, callbacks=[loss])
        # Noticed that providing tensorboard in the callback list
        # callbacks=[loss, tensorboard]
        # causes an error on the second model.  This appears to be
        # related to something in the previous model no longer existing.
        # It is unclear why this is not happening 
        

    print("Training loss %s"%(["%f"%(loss) for loss in loss.losses]))
    
    result = model.evaluate(testexamples, testlabels,
                            verbose=False)

    return (1 - result[1], model, loss) 
