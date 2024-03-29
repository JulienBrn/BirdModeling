import numpy as np, collections
from tqdm.auto import tqdm

def processPitch(data, fs, pitch_lower_bound=None, pitch_upper_bound=None, n_ffts=10):
    """
        Function to calculate pitch over provided data array.
        data -> Data points
        n_ffts -> No. of coarse ffts to calculate
        All coarse ffts will be combined to form the fine fft.
    """
    fineFFT = {}
    
    # Compute n_fft coarse ffts
    for fft_ind in np.arange(n_ffts):
        
        # Each fft is computed with one less data point.
        # This makes the algorithm compute energies at different frequencies
        courseFFT = np.abs(np.fft.fft(data[fft_ind:]).real)         # Energy values ()
        courseFreq = np.fft.fftfreq(data.shape[-1]-fft_ind) * fs   # Frequency values
        
        # All coarse ffts are combined into one pool
        # Only positive frequency values are considered
        for k in np.arange(int(courseFreq.size/2)):
            fineFFT[courseFreq[k]] = courseFFT[k]
                        
        # Uncomment this if you wish to plot the coarse FFTs.
#         plotCourseFFT(courseFFT, courseFreq, fft_ind)
        
    # Sort FFT elements (freq-energy pair) as per frequency value
    # Not necessary if you don't wish to interpolate
    sortedFineFFt = collections.OrderedDict(sorted(fineFFT.items()))

    # Uncomment this if you wish to plot the fine FFT.
#     plotFineFFT(sortedFineFFt)
            
    # Here, pitch = frequency with maximum energy within given frequency range
    sortedFineFFt_keys = np.array(list(sortedFineFFt.keys()))
    if pitch_lower_bound == -1:
        pitch = max(sortedFineFFt, key=sortedFineFFt.get)
    else:
        sortedFineFFt_values = np.array(list(sortedFineFFt.values()))
        ind=np.where((sortedFineFFt_keys>pitch_lower_bound) & (sortedFineFFt_keys<pitch_upper_bound))
        pitch = sortedFineFFt_keys[ind][np.argmax(sortedFineFFt_values[ind])]
    pitch_ind = np.argwhere(sortedFineFFt_keys == pitch)[0,0]

    # Uncomment this if you wish to interpolate around the peak
    pitch = interpolate_pitch(sortedFineFFt, pitch_ind)
            
    return pitch


def interpolate_pitch(FFT, ind):
        """
            Optional function to calculate pitch via interpolation.
            First, you compute the frequency at which the energy is maximum.
            Then, you take four points around it and calculate the weighted average.
        """
        
        # peak_freq = max(FFT, key=FFT.get)                # Frequency with maximum energy
        
        freq = np.array(list(FFT.keys()))
        # ind = np.argwhere(freq==peak_freq)[0,0]          # Twisted way to get the index because Python. ¯\_(ツ)_/¯ 
            
        # Weighted average across 4 points around the peak
        weighted_sum = 0
        weighted_sum += freq[ind-2] * FFT[freq[ind-2]]
        weighted_sum += freq[ind-1] * FFT[freq[ind-1]]
        weighted_sum += freq[ind]   * FFT[freq[ind]]
        weighted_sum += freq[ind+1] * FFT[freq[ind+1]]
        weighted_sum += freq[ind+2] * FFT[freq[ind+2]]
        
        # Note: There is a slight possibility that the adjacent positions end up having the same frequency value.
        # In the cpp version, I run a check for this and find the next different frequency index.
        # I let you decide if you wish to do this in python. The probability is quite low, possibly zero.
        
        sum_weights = FFT[freq[ind-2]] + FFT[freq[ind-1]] + FFT[freq[ind]] + FFT[freq[ind+1]] + FFT[freq[ind+2]]
        
        pitch = weighted_sum / sum_weights
        
        return pitch