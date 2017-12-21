
from sklearn.mixture import GaussianMixture
import numpy as np

from dsp.audioframes import AudioFrames
from dsp.multifileaudioframes import MultiFileAudioFrames
from dsp.rmsstream import RMSStream
 
class UnsupervisedVAD:
    """
    UnsupervisedVAD
    Unsupervised voice activity detector
    
    Uses a GMM to learn a two class distribution of RMS energy vectors
    
    # Arguments
    files - List of audio files from which to learn
    adv_ms - frame advance (milliseconds)
    len_ms - frame length (milliseconds)
    """
    Nclasses = 2    # two classes, speech/noise
        
    def __init__(self, files, adv_ms=10, len_ms=20):

        # Write me
        
    def classify(self, file):
        """classify - Extract RMS from file using the same
        framing parameters as the constructor and return vector
        of booleans where True indicates that speech occurred
        """
        # Write me

        
            
    