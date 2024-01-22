import sys
import numpy as np
import sounddevice as sd
from scipy import signal
import wcslib as wcs

def main():
    # Parameters
    fs = 33600          # Sampling frequency in Hz
    fc = 4200           # Carrier frequency in Hz
    Tb = 0.04           # Pulse width (to fit 1 sidelobe)
    channel_id = 16     # Your channel ID

    # FILTERS ###############################################################################
    def band_limit_filter(fs, signal_input):
        w_s1 = 4100 / (fs/2)
        w_p1 = 4150 / (fs/2)
        w_p2 = 4250 / (fs/2)
        w_s2 = 4300 / (fs/2)

        b, a = signal.iirdesign(wp = [w_p1, w_p2],
                                ws = [w_s1, w_s2],
                                gstop = 40, 
                                gpass = 1,
                                ftype = 'cheby2',
                                output = 'ba',
                                analog = False)
        
        # Frequency response of the filter
        w, h = signal.freqz(b, a, worN = 8000, fs = fs, whole = True)
        return signal.lfilter(b, a, signal_input)
    
    def low_pass_filter(fs, signal_input):
        w_p = 50 / (fs/2)
        w_s = 200 / (fs/2)

        b, a = signal.iirdesign(w_p, 
                                w_s, 
                                gstop = 40, 
                                gpass = 1, 
                                analog = False, 
                                ftype = 'cheby2', 
                                output = 'ba')

        # Frequency response of the filter
        w, h = signal.freqz(b, a, worN=8000, fs=fs, whole=True)
        return signal.lfilter(b, a, signal_input)

    #########################################################################################
    
    # Receive signal from microphone
    duration = 10 # Adjust the duration as needed
    print("Receiving signal!")
    yr = sd.rec(int(duration * fs), fs, channels=1, dtype=np.float32, blocking=True)
    sd.wait()
    print("Signal recieved!")

    # Convert received signal to 1D array
    #yr = np.squeeze(yr)

    # Apply bandpass filter to the received signal
    yr_bandpass = band_limit_filter(fs, yr)


    # Demodulate received signal
    t2 = np.arange(len(yr_bandpass))/fs
    yI_d = yr_bandpass[:,0] * np.cos(2*np.pi*fc * t2)
    yQ_d = -yr_bandpass[:,0] * np.sin(2*np.pi*fc * t2)

    yI_d_LP = low_pass_filter(fs, yI_d)
    yQ_d_LP = low_pass_filter(fs, yQ_d)
    

    y_b = yI_d_LP + 1j*yQ_d_LP

    y_mag = np.abs(y_b) #np.sqrt((yI_d_LP**2) + (yQ_d_LP**2))
    y_phase =  np.arctan2(yI_d_LP, yQ_d_LP) #np.angle(y_b)  
    
    # Baseband decoding using the library
    br = wcs.decode_baseband_signal(y_mag, y_phase, Tb, fs)

    data_rx = wcs.decode_string(br)


    # Print received data
    print("Received Signal:", data_rx)
    
if __name__ == "__main__":
    main()