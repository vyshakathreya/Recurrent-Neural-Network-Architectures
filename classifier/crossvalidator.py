
from sklearn.model_selection import StratifiedKFold
import numpy as np

from dsp.utils import Timer
        
class CrossValidator:
    debug = False
    
    def __init__(self, Examples, Labels, model, model_train_eval,
                 n_folds=10, batch_size=100, epochs=100): 
        """CrossValidator(Examples, Labels, model_spec, n_folds, batch_size, epochs)
        Given a list of training examples in Examples and a corresponding
        set of class labels in Labels, train and evaluate a learner
        using cross validation.
        
        arguments:
        Examples:  feature matrix, each row is a feature vector
        Labels:  Class labels, one per feature vector
        model: Keras model to learn
            e.g. result of buildmodels.build_model()
        model_train_evel: function that can be called to train and test
            a model.  Must conform to an interface that expects the following
            arguments:
                examples - list or tensor of examples
                labels - categories corresponding to examples
                train_idx - indices of examples, labels to be used
                    for training
                test_idx - indices of examples, labels to be used to
                    evaluate the model
                model - keras network to be used
                batch_size - # examples to process per batch
                epochs - Number of passes through training data
                name - test name
        n_folds - # of cross validation folds
        batch_size - # examples to process per batch
        epochs - Number of passes through training data       
        """
        
        # Create a plan for k-fold testing with shuffling of examples
        # http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html    #
        kfold = StratifiedKFold(n_folds, shuffle=True)
        
    
        foldidx = 0
        errors  = np.zeros([n_folds, 1])
        models = []
        losses = []
        timer = Timer()
        
        for (train_idx, test_idx) in kfold.split(Examples, Labels):
            (errors[foldidx], model, loss) = \
                model_train_eval(
                    Examples, Labels, train_idx, test_idx, model,
                    batch_size, epochs, name="f{}".format(foldidx)) 
            models.append(model)
            losses.append(loss)
            print(
                "Fold {} error {}, cumulative cross-validation time {}".format(
                    foldidx, errors[foldidx], timer.elapsed()))
            foldidx = foldidx + 1
                    
        # Show architecture of last model (all are the same)    
        print("Model summary\n{}".format(model.summary()))
        
        print("Fold errors:  {}".format(errors))
        print("Mean error {} +- {}".format(np.mean(errors), np.std(errors)))
        
        print("Experiment time: {}".format(timer.elapsed()))
        
        self.errors = errors
        self.models = models
        self.losses = losses
            
  
    def get_models(self):
        "get_models() - Return list of models created by cross validation"
        return self.models
    
    def get_errors(self):
        "get_errors - Return list of error rates from each fold"
        return self.errors
    
    def get_losses(self):
        "get_losses - Return list of loss histories associated with each model"
        return self.losses
                  
