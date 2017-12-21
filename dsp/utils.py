
from .pca import PCA
from .multifileaudioframes import MultiFileAudioFrames
from .dftstream import DFTStream
from .features import get_features

import os.path
from datetime import datetime
import numpy as np

import hashlib  # hash functions


def hash_filenames(files):
    "hash_filenames(files) - Generate hash key from list of files"
    
    md5 = hashlib.md5() # Merkle–Damgård 5 hash function 
    string = "".join(files)  # Concatenate file names into long str
    md5.update(string.encode('utf-8'))  # Generate key from UTF-8 characters
    hashkey = md5.hexdigest()  # Convert key to hexadecimal digest

    return hashkey

def pca_analysis_of_spectra(files, adv_ms, len_ms, vad=None, offset_s=None): 
    """"pca_analysis_of_spectra(files, advs_ms, len_ms, vad, offset_s)
    Conduct PCA analysis on spectra of the given files
    using the given framing parameters.  
    
    # Arguments
    adv_ms - frame advance in ms
    len_ms - frame length in ms
    vad - voice activity detector (endpointer) object  Detect speech
        when loading files
    offset_s - portion of spectra to retain
        None - everything
        >0 - Retain +/- offset_s of frames around center of each file
        offset_s is ignored when vad is specified
    
    """

    # Generating takes a bit of time, use cached features if available,
    # otherwise generate and cache
    hashkey = hash_filenames(files)    
    filename = "VarCovar-" + hashkey + ".pcl"
    
    try:
        pca = PCA.load(filename)

    except FileNotFoundError:
        example_list = []
        for f in files:
            example = get_features(f, adv_ms, len_ms, vad=vad, 
                                   offset_s=offset_s, flatten=False)
            example_list.append(example)
            
        # row oriented examples
        spectra = np.vstack(example_list)
    
        # principal components analysis
        pca = PCA(spectra)

        # Save it for next time
        pca.save(filename)
        
    return pca


def extract_tensors_from_corpus(files, adv_ms, len_ms, vad, offset_s, pca, 
                                 components):
    """extract_tensors_from_corpus(files, adv_ms, len_ms, vad, offset_s,
        pca, components)
        
    Return a set of tensors.  First dimension is tensor index.  Dimension two
    is the number of feature vectors, Dimension 3 is the size of the feature
    vector derived from a frame of audio.
    
    Spectral features are extracted based on framing parameters advs_ms, len_ms.    
    
    These spectra are projected into a PCA space of the specified number
    of components using the PCA space contained in object pca which is of
    type dsp.pca.PCA.
    
    This method will attempt to read from cached data as opposed to
    computing the features.  If the cache does not exist, it will be
    created.  Note the the cache files are not portable across machine
    architectures.
    
    # Arguments
    adv_ms - frame advance in ms
    len_ms - frame length in ms
    vad - voice activity detector (endpointer) object  Detect speech
        when loading files
    offset_s - portion of spectra to retain
        None - everything
        >0 - Retain +/- offset_s of frames around center of each file
        offset_s is ignored when vad is specified
    pca - pca.PCA object
    components - Number of principal components to retain        
    """

    # This part takes a bit of time, use cache based on files and parameters
    md5 = hashlib.md5()
    string = "".join(files)
    md5.update(string.encode('utf-8'))
    hashkey = md5.hexdigest()

    if vad is not None:
        trim_indicator_str = "VAD"    
    else:
        if offset_s is None:
            trim_indicator_str = "EntireFile"
        else:
            trim_indicator_str = "%d"%(offset_s * 1000)
        
    filename = "features-adv_ms{}-len_ms{}-trim{}-hash{}.npy".format(
        adv_ms, len_ms, trim_indicator_str, hashkey)
    
    generate = False
    try:
        print("Trying to load cached features from {}".format(filename))
        features = np.load(filename) 
        print("loaded")
    except IOError:
        generate = True
        
    if generate:
        print("No cached features, generating...")
        features = []
        log_file = open("vad.txt", "w")
        for idx, f in enumerate(files):
            example = get_features(f, adv_ms, len_ms, pca, components, 
                                   vad, offset_s, flatten=False,
                                   log_handle = log_file)
            features.append(example)
            if idx % 100 == 0:
                print("Extracted {} of {}".format(idx, len(files))) 
        print("Completed feature extraction")

        # Cache on secondary storage for quicker computation next time
        np.save(filename, features)

    return features

       
def get_corpus(dir, filetype=".wav"):
    """get_corpus(dir, filetype=".wav"
    Traverse a directory's subtree picking up all files of correct type
    """
    
    files = []
    
    # Standard traversal with os.walk, see library docs
    for dirpath, dirnames, filenames in os.walk(dir):
        for filename in [f for f in filenames if f.endswith(filetype)]:
            files.append(os.path.join(dirpath, filename))
                         
    return files
    
def get_class(files):
    """get_class(files)
    Given a list of files, extract numeric class labels from the filenames
    """
    
    # TIDIGITS single digit file specific
    
    classmap = {'z': 0, '1': 1, '2': 2, '3': 3, '4': 4,
                '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'o': 10}

    # Class name is given by first character of filename    
    classes = []
    for f in files:        
        dir, fname = os.path.split(f) # Access filename without path
        classes.append(classmap[fname[0]])
        
    return classes
    
class Timer:
    """Class for timing execution
    Usage:
        t = Timer()
        ... do stuff ...
        print(t.elapsed())  # Time elapsed since timer started        
    """
    def __init__(self):
        "timer() - start timing elapsed wall clock time"
        self.start = datetime.now()
        
    def reset(self):
        "reset() - reset clock"
        self.start = datetime.now()
        
    def elapsed(self):
        "elapsed() - return time elapsed since start or last reset"
        return datetime.now() - self.start
    
