
from keras.layers import Dense, Input, TimeDistributed, SimpleRNN, LSTM, GRU, Masking
from keras.models import Sequential, Model
import keras.backend as K

def build_model(specification, name="model"):
    """build_model - specification list
    Create a model given a specification list
    Each element of the list represents a layer and is formed by a tuple.
    
    (layer_constructor, 
     positional_parameter_list,
     keyword_parameter_dictionary)
    
    Example, create M dimensional input to a 3 layer network with 
    20 unit ReLU hidden layers and N unit softmax output layer
    
    [(Dense, [20], {'activation':'relu', 'input_dim': M}),
     (Dense, [20], {'activation':'relu', 'input_dim':20}),
     (Dense, [N], {'activation':'softmax', 'input_dim':20})
    ]
    
    There is no current way to handle nested models, e.g.
    the use of layer wrappers such as TimeDistributed

    """
    
    K.name_scope(name)
    model = Sequential()

    for item in specification:
        layertype = item[0]
        # Construct layer and add to model
        # This uses Python's *args and **kwargs constructs
        #
        # In a function call, *args passes each item of a list to 
        # the function as a positional parameter
        #
        # **args passes each item of a dictionary as a keyword argument
        # use the dictionary key as the argument name and the dictionary
        # value as the parameter value
        #
        # Note that *args and **args can be used in function declarations
        # to accept variable length arguments.
        layer = layertype(*item[1], **item[2])
        model.add(layer)
        
    return model