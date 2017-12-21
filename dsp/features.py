
from .audioframes import AudioFrames
from .dftstream import DFTStream

import numpy as np
import hashlib  # hash functions
from _tracemalloc import start


def get_features(file, adv_ms, len_ms, pca=None, components=0, vad=None,
                 offset_s=None, flatten=True, log_handle=None):
    """get_features(file, adv_ms, len_ms, pca, components, offset_s, flatten=True)
    
    Given a file path (file), compute a spectrogram with
    framing parameters of adv_ms, len_ms.  To remove frames
    use vad or offset_s (see below)

    If a pca object is given, reduce the dimensionality of the spectra to the
    specified number of components using a PCA analysis (dsp.PCA object in
    variable pca).
    
    If flatten is True, convert to 1D feature vector
    
    # Arguments
    file - Audio file to read
    adv_ms - frame advance in ms
    len_ms - frame length in ms
    vad - voice activity detector (endpointer) object  Detect speech
        when loading files
    offset_s - portion of spectra to retain
        None - everything
        >0 - Retain +/- offset_s of frames around center of each file
        NOTE:  offset_s is ignored when vad is specified
    pca - pca.PCA object
    components - Number of principal components to retain     
    log_handle - If present, is a handle to a file stream.  The file's
        name will be logged along with the start and end time used
        in seconds
    """
    
    framestream = AudioFrames(file, adv_ms, len_ms)
    dftstream = DFTStream(framestream)
    
    spectra = []
    for s in dftstream:
        spectra.append(s)
    # Row oriented spectra
    spectra = np.asarray(spectra)


    frames = spectra.shape[0]  # Number of spectral frames
    if vad is None:
        if offset_s is None:
            # User wants everything
            features = spectra
            span = [0, frames]
        else:
            # Compute start and end frame, then extract
            
            # Take center of spectra +/- offset_s    
            offset_frames = int(offset_s * 1.0 / (adv_ms/1000))
    
            center = int(frames / 2.0)
            left = max(0, center - offset_frames)
            right = min(frames, center + offset_frames)  
            span = [left, right]      
        
            if frames < 2 * offset_frames + 1:
                raise RuntimeError("File {} too short".format(file))
        
            # Extract center -/+ offset_s
            features = spectra[slice(span[0],span[1]),:]
    else:
        # Make sure that classification has the saem frame parameters
        assert(vad.adv_ms == adv_ms and vad.len_ms == len_ms)
        
        # Write me
        # Use the vad to label frames as speech or noise
        # and then populate list span such span[0] is
        # the left most part of the speech to be retained
        # and span[1] is the right most.
            
        features = spectra[slice(span[0], span[1]), :]
        
    # Convert spectra to PCA space
    if not pca is None:
        features = pca.transform(features, components)
        
    # Convert matrix to vector for input
    if flatten:
        features = features.flatten()

    if log_handle is not None:
        ms_per_s = 1000
        # Start times of the beginning and ending frames used.
        start_s = float(span[0]) * adv_ms / ms_per_s
        end_s = float(span[1] - 1) * adv_ms / ms_per_s 
        log_handle.write("%s extracted %.2f - %.2f s frames=%d\n"%(
            file, start_s, end_s, span[1] - span[0]))
        
    return features



    
    
    
    
            
        
        
        
    
        