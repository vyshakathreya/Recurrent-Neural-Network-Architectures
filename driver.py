
from dsp.pca import PCA
from dsp.utils import pca_analysis_of_spectra
from dsp.utils import get_corpus, get_class, Timer, \
    extract_tensors_from_corpus
from dsp.features import get_features
from classifier.batchgenerator import PaddedBatchGenerator

import classifier.feedforward
import classifier.recurrent
from classifier.crossvalidator import CrossValidator
from classifier.buildmodels import build_model

from classifier.voiceactivitydetector import UnsupervisedVAD

from keras.layers import Dense, Dropout, SimpleRNN, Masking, LSTM
from keras import regularizers

import numpy as np

    
def main():
   
    
    files = get_corpus("C:/users/corpora/tidigits/wav/train")
    # for testing
    if False:
        files[50:] = []  # truncate test for speed
    
    print("%d files"%(len(files)))
    
    adv_ms = 10
    len_ms = 20

    # If > 0, extract +/- offset_s, if None take everything unless
    # voice activity detector is used
    offset_s = None 

    print("Generating voice activity detection model")
    timer = Timer()
    vad = UnsupervisedVAD(files, adv_ms, len_ms)
    print("VAD training time {}, starting PCA analysis..."
          .format(timer.elapsed()))    
    timer.reset()
    
    pca = pca_analysis_of_spectra(files, adv_ms, len_ms, vad, offset_s)
    print("PCA feature generation and analysis time {}, feature extraction..."
          .format(timer.elapsed()))    
    timer.reset()


    # Read features - each row is a feature vector
    components = 40
    examples = extract_tensors_from_corpus(
        files, adv_ms, len_ms, vad, offset_s, pca, components)
    
    # Find the length of the longest time series
    max_frames = max([e.shape[0] for e in examples])
    print("Longest time series {} steps".format(max_frames))
    
    print("Time to generate features {}".format(timer.elapsed()))
    timer.reset()
    
    labels = get_class(files)
    
    
    outputN = len(set(labels))
    # Specify model architectures
    T = PaddedBatchGenerator.longest_sequence(examples)
    dim_recurrent = components
    dim_feedforward = T * components
    
    # This structure is used only for feed-forward networks
    # The RNN networks are built more traditionally as I need
    # to further develop the infrastructure for wrapping layers
    models_ff = [
        # Each list is a model that will be executed
        [(Dense, [30], {'activation':'relu', 'input_dim': dim_feedforward,
                        'kernel_regularizer':regularizers.l2(0.01)}),
         #(Dropout, [.25], {}),
         (Dense, [30], {'activation':'relu',
                        'kernel_regularizer':regularizers.l2(0.01)}),
         #(Dropout, [.25], {}),
         (Dense, [outputN], {'activation':'softmax',
                             'kernel_regularizer':regularizers.l2(0.01)})
         ]
    ]
    models_rnn = [    

      ]

    print("Time to build matrix {}, starting cross validation".format(
        timer.elapsed()))
    
    # Use recurrent classifiers if true
    recurrent = True
    if recurrent:
        # Use the recurrent neural net list and train/evaluation fn
        models = models_rnn
        train_eval = classifier.recurrent.train_and_evaluate
    else:
        # Use the feed forward neural net list and train/evaluation fn
        models = models_ff
        train_eval = classifier.feedforward.train_and_evaluate
        
    batch_size = 100
    epochs = 60
    
    debug = False
    if debug:
        models = [models[-1]]  # Only test the last model in the list

    results = []
    for architecture in models:
        model = build_model(architecture)
        results.append(CrossValidator(examples, labels, model, 
                        train_eval,
                        batch_size=batch_size, 
                        epochs=epochs))
            
    # do something useful with results... e.g. generate tables/graphs, etc.

 
    
if __name__ == '__main__':
    
    
    main()
