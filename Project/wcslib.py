#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Library for the wireless communication system project in Signals and Transforms

2020-present -- Roland Hostettler <roland.hostettler@angstrom.uu.se>
"""

import numpy as np
from scipy import signal
from scipy.stats import chi2

# List of channels and their max average power [fl, fu, Pmax]^T
_channels = np.array([
    [np.nan,  900, 1150, 1300, 1550, 1725, 1950, 2100, 2400, 2700, 3050,
        3200, 3475, 3550, 3750, 3900, 4150, 4300, 4550, 4750, 4900, 5200],
    [800, 1100, 1250, 1500, 1650, 1875, 2050, 2300, 2600, 2900, 3150, 3400,
        3525, 3650, 3850, 4100, 4250, 4500, 4650, 4850, 5100, np.nan],
    [np.nan,   30,   33,   27,   33,   33,   33,   30,   27,   30,   33,
        33,   27,   33,   30,   30,   33,   27,   33,   30,   27, np.nan]
])


def encode_string(instr):
    """
    Converts a string to a binary numpy array.

    Parameters
    ----------
    instr : str
        A UTf-8 encoded Python string.

    Returns
    -------
    binary : numpy.array
        A binary array encoding the string.
    """
    tmp = [np.uint8(ord(c)) for c in instr]
    return np.unpackbits(tmp)


def decode_string(inbin):
    """
    Converts a binary numpy array to string.

    Parameters
    ----------
    inbin : numpy.array
        A binary array of ones and zeros encoding a string.

    Returns
    -------
    outstr : str
        A UTF-8 encoded Python string.
    """
    tmp = np.packbits(inbin)
    outstr = "".join([chr(b) for b in tmp])
    return outstr


def encode_baseband_signal(b, Tb, fs):
    """
    Encodes a binary sequence into a baseband signal. In particular, generates 
    a discrete-time signal that encodes the binary signal `b` into pulses of 
    width `Tb` seconds at a sampling frequency `fs`, of amplitude 1 (for bits 
    that are 1) and -1 (for bits that are 0).

    To generate the discrete-time signal, the function determines the pulse 
    width in samples that corresponds to the specified parameters `Tb` and 
    `fs`. This is calculated by

        Kb = np.floor(Ts*fs),

    where `Kb` is the pulse width in samples.

    The function also prepends the bits [1, 0] (encoded as [-1, 1]) to the 
    message. These bits are used as a known sequence of bits and used in the 
    decoder (on the receiving side) to determine the time delay between the 
    sender and receiver to synchronize the decoding process with the signal. 

    Parameters
    ----------
    b : numpy.array
        A binary array of 1s and 0s encoding a message.
    Tb : float
        Pulse width in seconds to encode the bits to.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    xb : numpy.array
        Encoded baseband signal.
    """

    # Prepend synchronization sequence and a trailing zero
    b = np.concatenate(([1, 0], b))
    b = np.append(b, 0)

    # Encode bit values
    s = [-1, 1]
    b[b == 0] = s[0]
    b[b == 1] = s[1]

    # Expand
    Kb = int(np.floor(Tb*fs))
    Nx = b.shape[0]
    xb = np.zeros(Nx*Kb)
    xb[np.arange(0, Nx*Kb, Kb)] = b

    # "Lowpass filtering"
    b = np.ones(Kb)
    xb = signal.lfilter(b, 1, xb)

    return xb


def decode_baseband_signal(xm, xp, Tb: float, fs: float):
    """
    Decodes an IQ-demodulated baseband signal consisting of a magnitude signal
    `xm` and a phase signal `xp` into a binary bit sequence.

    The bit sequence is recovered by first determining the window of the 
    transmission. This is achieved by averaging the magnitude signal over a 
    sliding window of the length of a symbol (`Tb`) (implemented using a filter
    with a rect of length `Tb` as the impulse response). Then, the average 
    is compared to a threshold, where the threshold is determined using the 
    tail probability of a chi-squared distribution (in essence, the test checks
    whether there is a signal or only noise with a probability of 99 %).

    Then, a filter with impulse response consisting of two pulses corresponding
    to the mirrored synchronization sequence is used to find the first 
    occurence of this pulse sequence in the signal. This corresponds to 
    correlating the signal with the synchronization sequence and is used to 
    determine the time delay introduced by filters and the transmission itself
    (i.e., to synchronize the data stream).

    Next, the (averaged) phases of the first two bits are used to find a 
    complex representation of the symbols (which will be different from -1 and
    1 due to the phase shift introduced the transmission and filtering).

    Finally, the whole phase signal is converted to a complex number with angle
    corresponding to the phase shift and bit values are determined by comparing
    averages of length `Tb` to the previously recovered symbols.

    Parameters
    ----------
    xm : numpy.array
        The magnitude of the IQ-demodulated baseband signal.
    xp : numpy.array
        The phase of the IQ-demodulated baseband signal.
    Tb : float
        Pulse width in seconds to encode the bits to.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    b : numpy.array
        A binary array of 1s and 0s encoding a message.
    """

    # 1. Signal detection:
    Kb = int(np.floor(Tb*fs))
    hd = np.ones((Kb,))
    xm2 = signal.lfilter(hd, 1, xm**2)
    xm_var = np.var(xm)
    xtest = chi2.cdf(xm2/xm_var, 2*Kb)
    d = xtest > 0.99
    m = np.argmax(d)

    # 2. Synchronization
    # Synchronize using a matched filter. N.B: Expects the first to bits to be
    # [1, 0] as prepended by encode_baseband_signal()
    # TODO: Remove unwrapping? by doing this in the complex domain as well.
    # TODO: Synchronization sequence is hardcoded here.
    xpd = _unwrap(xp)
    hb = 1/(2*Kb)*np.concatenate((-np.ones(Kb), np.ones(Kb)))
    xd = np.sign(xpd)*d
    xs = signal.lfilter(hb, 1, xd)

    # The peak of the synchronization sequence is within m+Nsynch*Kb. Hence, we
    # can get an exact match within that window to get "perfect"
    # synchronization
    # TODO: "2" is hardcoded here, assumes two synchronization bits.
    k0 = np.argmax(abs(xs[:m+2*Kb]))
    xx = np.vstack((
        1/Kb*signal.lfilter(hd, 1, np.cos(xp)),
        1/Kb*signal.lfilter(hd, 1, np.sin(xp))
    ))
    b1 = xx[:, k0-Kb]
    b0 = xx[:, k0]

    # 3. Recover the bits
    # Calculate th projection of the complex number onto the symbol of the bit
    # `1` (b1; inner product), at every Kb starting from k0. Then, remove bits
    # where no signal was detected (radio silence) and threshold to get ones
    # and zeros (the inner product is close to 1 if the bit is close to the
    # symbol for `1`` or close to -1 if the bit is close to the symbol for
    # `0`).
    b = b1@xx[:, k0+Kb::Kb]
    b = b[d[k0+Kb::Kb]] > 0

    return b


def _unwrap(xp, alpha: float = np.pi/8):
    """
    Unwraps the phase for binary phase-shift keying modulated signals.

    Parameters
    ----------
    xp : numpy.array
        Phase signal.
    alpha : float, default: pi/8
        Closeness threshold.

    Returns
    -------
    xp : numpy.array
        Unwrapped signal.
    """

    # Wrap +pi to -pi
    dpi = np.pi-xp
    iclose = (dpi < alpha) & (dpi > 0)
    xp[iclose] = xp[iclose]-2*np.pi

    # Rotate by alpha to remove sign switches around 0 (but preserve curve)
    xp = xp + alpha

    return xp


def simulate_channel(x, fs: float, channel_id: int, SNR: float = 20.0, eta: float = 0.25, dmax: float = 5.0):
    """
    Takes the modulated (discrete-time) signal `x` (generated at sampling 
    frequency `fs`) and simulates a wireless transmission through open space at
    a random distance in the interval from 0 to `dmax` meter. The channel model
    consists of

    * Signal attenuation and delay due to its propagation in space,
    * random noise, and
    * random out-of-channel interference.

    Channel model: The chanel model is based on an exponential decay 
    (attenuation) and time-of-flight-based time-delay, that is,

        y[k] = e^(-eta*d)*x[k - m]

    where `d` is a random distance uniformly drawn on the interval [0, `dmax`]
    and

        m = np.round(d/c*fs)

    Random noise: Random noise is added such that a constant (transmitter) 
    signal-to-noise ratio of `SNR` is achieved.

    Out-of-channel interference: In addition to the random noise, an out-of-
    channel interference is added, which corresponds to a random transmission
    at another frequency. The transmission frequency is sampled uniformly from
    the whole spectrum outside the given channel, while the amplitude is 
    sampled from a Gaussian distribution with mean 1 and standard deviation 
    0.2.

    /!\ The default values for `SNR`, `eta`,  as well as `dmax` should not be
        changed unless you know what you are doing. /!\

    Parameters
    ----------
    x : numpy.array
        The modulated signal to be transmitted.

    fs : float
        Sampling frequency.

    channel_id : int
        The id of the communication channel.

    SNR : float, default 20.0
        The signal-to-noise ratio at the transmitter (in dBm). The default is 
        20 dBm, which corresponds to a signal that is 10x as strong (in terms
        of power) as the noise.

    eta : float, default 0.25
        Fading coefficient.

    dmax : float, default 5.0
        The maximum transmission distance.

    Returns
    -------
    y : numpy.array
        The signal received by the receiver.
    """

    # Get channel parameters
    if not (channel_id >= 1 and channel_id < _channels.shape[1]-1):
        raise ValueError(
            f'channel_id must be between 1 and {_channels.shape[1]}, but {channel_id} given.')
    channel = _channels[:, channel_id]

    # Create the channel impulse response: A Kronecker delta with amplitude
    # exp(-eta*d) at sample m
    c = 340
    d = dmax*np.random.rand(1)
    m = int(np.round(d/c*fs))
    h = np.zeros(m+1)
    h[m] = np.exp(-eta*d)

    # Zero-pad x to make sure the whole signal is preserved
    x = np.concatenate((x, np.zeros(m,)))

    # Calculate the noise variance based on the SNR (and the sampling
    # frequency)
    fb = (channel[1] - channel[0])/2            # One-sided channel bandwidth
    # In-band noise power for given SNR
    Pnoise = 10**((channel[2] - SNR)/10)*1e-3
    # White noise power for given SNR
    sigma2 = Pnoise*fs/(4*fb)
    Nx = x.shape[0]
    vn = np.sqrt(sigma2)*np.random.randn(Nx)

    # Add out-of-band interference at a random channel, uniformly distributed
    # outside the channel's frequency band taking aliasing into account (i.e.,
    # making sure that the interference doesn't cause aliasing). The latter is
    # achieved by noting that the sampling frequency ws >> 4*wc, which means
    # that channels with carrier frequency <= 2*wc are allowed.
    fc = (channel[0]+channel[1])/2
    fcs = (_channels[0, :]+_channels[1, :])/2
    ichannels = (fcs <= 2*fc) & (fcs != fc)
    fcs = fcs[ichannels]
    ichannel = np.random.randint(0, fcs.shape[0])
    fi = fcs[ichannel]

    # Now, sample the interference amplitude with a mean of 1 (30 dBm) and
    # a standard deviation of 0.2 (95 % between 0.6 and 1.4). Then add
    # everything together to generate the interference signal
    Ai = 1 + 0.2*np.random.rand(1)
    k = np.arange(0, x.shape[0])
    vi = Ai*np.sin(2*np.pi*fi*k/fs)

    # Construct received signal
    y = signal.lfilter(h, 1, x) + vn + vi

    return y
