import sys
import numpy as np
from scipy import signal
import sounddevice as sd
import wcslib as wcs

def main():
    # Parameters
    Tb = 0.04           # Symbol width in seconds
    fs = 33600          # Sampling frequency in Hz
    fc = 4200           # Carrier frequency in Hz

    # FILTERS ###############################################################################
    def band_limit_filter(fs, signal_input):
        w_s1 = 4100 / (fs/2)
        w_p1 = 4150 / (fs/2)
        w_p2 = 4250 / (fs/2)
        w_s2 = 4300 / (fs/2)

        b, a = signal.iirdesign(wp=[w_p1,w_p2],
                                ws= [w_s1,w_s2],
                                gstop=40, gpass=1,
                                ftype='cheby2',
                                output='ba',
                                analog = False)
        # Frequency response of the filter
        w, h = signal.freqz(b, a, worN=8000, fs=fs, whole=True)
        return signal.lfilter(b, a, signal_input)
    #########################################################################################

        # Detect input or set defaults
    string_data = True
    if len(sys.argv) == 2:
        data = str(sys.argv[1])

    elif len(sys.argv) == 3 and str(sys.argv[1]) == '-b':
        string_data = False
        data = str(sys.argv[2])

    else:
        print('Warning: No input arguments, using defaults.', file=sys.stderr)
        data = "Hello, World! How are you "

    # Convert string to bit sequence or string bit sequence to numeric bit sequence
    if string_data:
        bs = wcs.encode_string(data)
    else:
        bs = np.array([bit for bit in map(int, data)])

    # Encode baseband signal
    xb = wcs.encode_baseband_signal(bs, Tb, fs)

    t = np.arange(len(xb))/fs
    xt = xb * 2*np.sin(fc*2*np.pi * t)

    #Band limit the transmittion
    xt_filtered = band_limit_filter(fs, xt)

    # Transmit signal through speakers
    sd.play(xt_filtered, fs)
    sd.wait()

if __name__ == "__main__":
    main()