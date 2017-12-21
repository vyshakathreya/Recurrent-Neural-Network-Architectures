
import numpy as np

class PaddedBatchGenerator(object):
    """PaddedBatchGenerator
    Class for sequence length normalization.
    Each sequence is normalized to the longest sequence in the batch
    or a specified length by zero padding. 
    """
    
    debug = True
    
    @classmethod
    def longest_sequence(cls, features):
        """longest_sequence(features)
        Return the longest sequence in a set of features
        
        # Arguments
        features:  list of arrays, each one is a time x dim array
        """
        
        # Find longest sequence
        N = max(f.shape[0] for f in features)
        return N
    
    def __init__(self, features, targets, batch_size=100,
                 flatten=False, replicate_target=False, pad_length=None):
        """"PaddedBatchGenerator(features, targets, batch_size)
        Create object for generating padded batches.
        
        # Arguments
        features: list of arrays, each one is a time x dim array.
        targets: array of scalars indicating sequence class
        replicate_target: boolean indicating whether the target should
            be replicated (True) for each time step or if a single vector
            is to be produced (False) 
        pad_length:  If specified, zero-padding is always to a fixed
            length rather than the longest sequence in the data.  This
            permits the specification of uniform batch sizes.  Note that
            pad_length must be >= the length of the longest sequence.
        """
        
        self.batch_size = batch_size
        self.features = features
        self.targets = targets
        self.replicate_target = replicate_target
        self._epoch = 0    # current number of completed epochs
        
        self.current = 0  # next index of data to use
        self.N = len(self.features)
        
        self.flatten = flatten # Convert time-series to flat vectors?
        self.pad_length = pad_length
        
        assert(batch_size <= len(features))
        assert(self.N == len(self.targets))
        
    @property
    def epoch(self):
        return self._epoch
    
    
    def __get_indices(self):
        """__get_indices() - private method for generating indices of the next
        batch.  Returns a list of one or two ranges.  Two ranges
        is an indication of wrap around.
        """
        
        if self.current == self.N:
            # Previous batch returned last index, wrap around
            self._epoch = self._epoch+1  # return index
            self.current = 0            
             
        last_index = self.current + self.batch_size
        ranges = []
        
        # Next batch starts here (possibly)
        next_start = last_index
        
        if last_index > self.N:
            # Compute end slice index of beginning features
            wrapped_index = last_index % self.N
            ranges.append(slice(0, wrapped_index))
            next_start = wrapped_index
            
            last_index = self.N
            self._epoch = self._epoch + 1
            
        ranges.append(slice(self.current, last_index))
        self.current = next_start
        
        ranges.reverse()  # Make sure in order
        
        return ranges
            
        
    def __next__(self):
        """__next__() - Return next set of training examples and target values
        All training vectors in batch will be normalized to the longest
        sequence in the batch.
        """
        
        ranges = self.__get_indices()
        
        # Build training and target lists with elements from batch
        training = []
        targets = []
        for r in ranges:
            training.extend(self.features[r])
            targets.extend(self.targets[r])
            
        # Find longest sequence
        N = max(e.shape[0] for e in training)
        if self.pad_length is not None:
            if self.pad_length < N:
                raise ValueError(
                    "pad_length={} < input sequence length".format(self.pad_length))
            else:
                N = self.pad_length
        
        
        # label vector size
        dim_label = targets[0].size
        
        # Create a tensor with standard length
        batchfeats = np.zeros([self.batch_size, N, training[0].shape[1]])
        if self.replicate_target:
            batchtargets = np.zeros([self.batch_size, N, dim_label])
        else:
            batchtargets = np.zeros([self.batch_size, dim_label])
        idx = 0
        # For each example and label in the batch
        for e, l in zip(training, targets):
            e_N = e.shape[0]
            # Copy example into beginning of possibly longer array
            #batchfeats[idx,0:e_N,:] = e
            # Copy example into end of possibly longer array
            try:
                batchfeats[idx,-e_N:,:] = e
            except Exception:
                print(idx)
                
            if self.replicate_target:
                # Copy label for all time steps
                batchtargets[idx,:,:] = l * np.ones([N, dim_label])
            else:
                batchtargets[idx,:] = l
            idx = idx + 1
        
        if self.flatten:
            batchfeatstmp = batchfeats.reshape([batchfeats.shape[0],-1])
            batchfeats = batchfeatstmp
            
            
        if self.debug:
            print("batch features {}".format(
                batchfeats.shape))
        return (batchfeats, batchtargets)
             
        
        
if __name__ == '__main__':
    # Code to test PaddedFeatureGenerator
    
    import keras.layers
    from keras.utils import to_categorical
    
    targets = [0, 1, 2, 3, 4]
    onehot = to_categorical(targets)

    lengths = [3, 4, 5, 4, 3]
    f = []
    idx = 0
    for idx in range(len(lengths)):
        f.append(np.zeros([lengths[idx], 3]) + (10 + targets[idx]))
    
    b = PaddedBatchGenerator(f, onehot, batch_size = 3)
    for idx in range(10):
        [e,l] = b.__next__()
        print(e)
        print(l)
        print("Above is batch {} epoch {}".format(idx, b.epoch))
        
        