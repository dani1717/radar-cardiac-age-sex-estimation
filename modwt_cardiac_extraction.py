import numpy as np


def cardiacSignal_MODWT_extract(displacement, fs, MATLAB_eng, level=7, wavelet='db4', levels_to_take=(1,5), padding_pct=5, fs_new=100):
    """
    Isolate the cardiac component embedded in thoracic displacement using MODWT (Maximal Overlap Discrete Wavelet Transform) 
    computed via MATLAB.
    This function applies wavelet-based multiresolution analysis to a preprocessed displacement signal—obtained from radar I/Q 
    channels through ellipse fitting and arctangent demodulation—in order to enhance the periodic patterns associated with 
    cardiac activity while suppressing noise and respiratory components.
        
    Parameters:
        displacement (array): Signal representing thoracic displacement over time, obtained after
                              ellipse fitting and arctangent demodulation of the radar's I/Q channels.
                              This signal captures minute chest wall movements caused by the cardiac cycle,
                              and serves as the base for further wavelet-based filtering.
        fs (int): Original sampling frequency.
        MATLAB_eng: Active MATLAB engine object.
        level (int): MODWT decomposition level.
        wavelet (str): Wavelet type.
        levels_to_take (tuple): Indices of MODWT levels to retain for signal reconstruction.
        padding_pct (float or None): Optional percentage of symmetric padding applied before decomposition
                                     to reduce boundary artifacts.
        
    Returns:
        tuple: (Reconstructed cardiac signal, new sampling frequency)
    """
    signal = FIR_filter(displacement, fs, order=100, lowcut=0.7, highcut=2)
    signal = resample_signal(signal, fs_old=fs, fs_new=fs_new)
    
    max_level = np.floor(np.log2(len(signal)))
    if level > max_level:
        print("The specified level is greater than the maximum allowed level for DWT decomposition.")
        return None

    if padding_pct is not None:
        padding_length = int(len(signal) * padding_pct / 100)
        signal_padded = np.pad(signal, (padding_length, padding_length), mode='reflect')
    else:
        signal_padded = signal

    wcoefs = MATLAB_eng.modwt(signal_padded.tolist(), wavelet, level)
    wcoefs = np.array(wcoefs)

    wcoefs_filtered = np.zeros_like(wcoefs)
    for i in range(levels_to_take[0], levels_to_take[1] + 1):
        wcoefs_filtered[i, :] = wcoefs[i, :]

    reconstructed_signal = MATLAB_eng.modwtmra(wcoefs_filtered.tolist(), wavelet)
    reconstructed_signal = np.array(reconstructed_signal)
    reconstructed_signal = np.sum(reconstructed_signal, axis=0)

    if padding_pct is not None:
        reconstructed_signal = reconstructed_signal[padding_length:padding_length + len(signal)]

    return reconstructed_signal, fs_new