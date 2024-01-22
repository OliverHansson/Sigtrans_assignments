#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation template for the wireless communication system project in Signals 
and Transforms.

For plain text inputs, run:
$ python3 simulation.py "Hello World!"

For binary inputs, run:
$ python3 simulation.py -b 010010000110100100100001

2020-present -- Roland Hostettler <roland.hostettler@angstrom.uu.se>
"""

import sys
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import wcslib as wcs

def main():
    # Parameters

    channel_id = 16     # Your channel ID
    Tb = 0.04           # Symbol width in seconds
    fs = 33600          # Sampling frequency in Hz
    fc = 4200           # Carrier frequency in Hz

    # FILTERS ###############################################################################

    def band_limit_filter(fs, signal_input):
        w_s1 = 4115 / (fs/2)
        w_p1 = 4150 / (fs/2)
        w_p2 = 4250 / (fs/2)
        w_s2 = 4285 / (fs/2)

        b, a = signal.iirdesign(wp=[w_p1, w_p2],
                                ws=[w_s1, w_s2],
                                gstop=50, gpass=3,
                                ftype='cheby2',
                                output='ba')

        # Frequency response of the filter
        # w, h = signal.freqz(b, a, worN=8000, fs=fs, whole=True)

        return signal.lfilter(b, a, signal_input)

    def low_pass_filter(fs, signal_input):
        w_p = 4250 / (fs/2)
        w_s = 4285 / (fs/2)

        b, a = signal.iirdesign(w_p,
                                w_s,
                                gstop=50,
                                gpass=15,
                                ftype='cheby2',
                                output='ba')

        # Frequency response of the filter
        # w, h = signal.freqz(b, a, worN=8000, fs=fs, whole=True)

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
        data = "Hello World!"

    # Convert string to bit sequence or string bit sequence to numeric bit sequence
    if string_data:
        bs = wcs.encode_string(data)
    else:
        bs = np.array([bit for bit in map(int, data)])

    # Encode baseband signal
    # print("HERE:")
    # print(bs)
    xb = wcs.encode_baseband_signal(bs, Tb, fs)
    # print(xb)

    # TODO: Put your transmitter code here (feel free to modify any other parts too, of course)
    t = np.arange(len(xb))/fs
    # np.cos(2 * np.pi * fc * np.arange(len(xb)) / fs)
    xt = xb * 2*np.sin(fc*2*np.pi * t)

    xt_filtered = band_limit_filter(fs, xt)

    # Channel simulation
    yr = wcs.simulate_channel(xt_filtered, fs, channel_id)

    # Apply bandpass filter to the received signal
    # signal.lfilter(bandpass_filter[0], bandpass_filter[1], yr)
    yr_bandpass = band_limit_filter(fs, yr)

    # TODO: Put your receiver code here. Replace the three lines below, they
    # are only there for illustration and as an MWE. Feel free to modify any
    # other parts of the code as you see fit, of course.
    # Demodulate received signal

    t2 = np.arange(len(yr_bandpass))/fs
    yI_d = yr_bandpass * np.cos(2*np.pi*fc * t2)
    yQ_d = -yr_bandpass * np.sin(2*np.pi*fc * t2)

    yI_d_LP = low_pass_filter(fs, yI_d)
    yQ_d_LP = low_pass_filter(fs, yQ_d)

    # y_b = yI_d_LP + yQ_d_LP

    xm = np.sqrt((yI_d_LP**2) + (yQ_d_LP**2))  # np.abs(y_b)
    xp = np.arctan2(yI_d_LP, yQ_d_LP)  # np.angle(y_b)

    # Baseband decoding using the library
    br = wcs.decode_baseband_signal(xm, xp, Tb, fs)

    data_rx = wcs.decode_string(br)

    # Print additional information for debugging
    print('Received (binary): ' + ''.join(map(str, br)))
    print('Received (hex): ' + ''.join(format(x, '02x') for x in bytes(br)))
    print('Received (decoded): ' + data_rx)


if __name__ == "__main__":
    main()
