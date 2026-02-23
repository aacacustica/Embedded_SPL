import numpy as np
from scipy import signal



def third_octave_filter(x, fs, order=6, limits=[12, 20000]):
    """
    Filter a signal with a 1/3-octave filter bank.
    
    Parameters:
      x                : Input signal (list, tuple, or numpy array)
      fs               : Sampling frequency in Hz.
      order            : Order of the Butterworth filter (default=6).
      limits           : Frequency limits [min, max] for the analysis (default: [12, 20000] Hz).
      calibration_coeff: Optional list of calibration coefficients (in dB) for each band.
    
    Returns:
        spl  : List of sound pressure levels (dB) for each band.
        freq : List of center frequencies for each band.
        xb   : List of filtered signals (one per frequency band).
    """
    # For a third-octave analysis, the fraction is fixed to 3.
    fraction = 3

    # Ensure the signal is in list format
    x = _typesignal(x)

    # Generate the center frequencies and band edge frequencies
    freq, freq_d, freq_u = _genfreqs(limits, fraction, fs)

    # Compute a downsampling factor for each band to improve filter accuracy
    factor = _downsamplingfactor(freq_u, fs)

    # Get the second-order section (SOS) coefficients for each band
    sos = _buttersosfilter(freq_d, freq_u, fs, order, factor)

    # Process each band: resample, filter, and compute the SPL
    spl = np.zeros(len(freq))
    for idx in range(len(freq)):
        sd = signal.resample(x, round(len(x) / factor[idx]))
        y = signal.sosfilt(sos[idx], sd)
        spl[idx] = 20 * np.log10(np.std(y) / 2e-5)

    return spl.tolist(), freq



def _typesignal(x):
    """Convert input signal to a list."""
    if isinstance(x, (list, tuple)):
        return list(x)
    elif isinstance(x, np.ndarray):
        return x.tolist()
    else:
        raise TypeError("Unsupported signal type. Provide a list, tuple, or numpy array.")



def _buttersosfilter(freq_d, freq_u, fs, order, factor):
    """Compute the SOS coefficients for each third-octave band."""
    sos = []
    for idx, (low, high) in enumerate(zip(freq_d, freq_u)):
        fsd = fs / factor[idx]  # Adjusted sampling frequency for current band
        sos_coeff = signal.butter(
            N=order,
            Wn=np.array([low, high]) / (fsd / 2),
            btype='bandpass',
            output='sos'
        )
        sos.append(sos_coeff)
    return sos



def _genfreqs(limits, fraction, fs):
    """Generate center and edge frequencies then remove bands above fs/2."""
    freq, freq_d, freq_u = getansifrequencies(fraction, limits)
    freq, freq_d, freq_u = _deleteouters(freq, freq_d, freq_u, fs)
    return freq, freq_d, freq_u



def _deleteouters(freq, freq_d, freq_u, fs):
    """Remove frequency bands with upper edges above fs/2."""
    valid = np.array(freq_u) <= fs / 2
    return (list(np.array(freq)[valid]),
            list(np.array(freq_d)[valid]),
            list(np.array(freq_u)[valid]))



def getansifrequencies(fraction, limits=[12, 20000]):
    """
    Compute center frequencies and band edges for a 1/3-octave filter bank
    using the ANSI/IEC standards.
    """
    # g is the octave ratio defined as 10^(3/10)
    g = 10 ** (3 / 10)
    fr = 1000  # Reference frequency in Hz
    x = _initindex(limits[0], fr, g, fraction)
    freq = _ratio(g, x, fraction) * fr
    freq_list = [freq]
    # Keep adding bands until the upper edge exceeds the specified limit
    while freq * _bandedge(g, fraction) < limits[1]:
        x += 1
        freq = _ratio(g, x, fraction) * fr
        freq_list.append(freq)
    freq_array = np.array(freq_list)
    freq_d = freq_array / _bandedge(g, fraction)
    freq_u = freq_array * _bandedge(g, fraction)
    return freq_array.tolist(), freq_d.tolist(), freq_u.tolist()



def _initindex(f, fr, g, b):
    """
    Compute the starting index for the band whose lower edge is greater than f.
    For third-octave (b=3) the odd branch of the ANSI formula is used.
    """
    return np.round((b * np.log(f / fr) + 30 * np.log(g)) / np.log(g))



def _ratio(g, x, b):
    """Compute the center frequency ratio using the ANSI formula (odd branch for third-octave)."""
    return g ** ((x - 30) / b)



def _bandedge(g, b):
    """Compute the band-edge ratio."""
    return g ** (1 / (2 * b))



def _downsamplingfactor(freq, fs):
    """
    Calculate a downsampling factor for each band based on the upper edge frequency,
    ensuring that the downsampled rate remains sufficient.
    """
    guard = 0.10  # small guard factor
    factor = np.floor((fs / (2 + guard)) / np.array(freq)).astype(int)
    return np.clip(factor, 1, 50)
