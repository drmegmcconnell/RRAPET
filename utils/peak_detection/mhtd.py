"""
MHTD

Author: Meghan McConnell
"""

import numpy as np
from scipy.signal import hilbert, firwin


def MHTD(dat, fs=360, fpass=5, fstop=30, thr_ratio=1.15, sbl=5, mag_lim=0.10, eng_lim=0.05, min_L=0.3):
    """

    :param dat:
    :param fs:
    :param fpass:
    :param fstop:
    :param thr_ratio:
    :param sbl:
    :param mag_lim:
    :param eng_lim:
    :param min_L:
    :return:
    """
    dat = _crop_ecg(dat)
    flt_dat = _myfilter(dat, fs, fpass, fstop)
    x = _HillTransform(dat, flt_dat[:])
    thr = _VariableThresh(x, fs * sbl, thr_ratio)
    pred = _PeakSearch(x, dat, thr)
    R_t = _Correction_Al2(pred, fs, dat, mag_lim=mag_lim, energy_lim=eng_lim, min_L=min_L)

    return np.reshape(R_t, [-1, ])


def _myfilter(dat, fs, fpass=0.5, fstop=45):
    """

    :param dat:
    :param fs:
    :param fpass:
    :param fstop:
    :return:
    """
    n = int(fs * 0.2225 + 220)
    wind = firwin((n + np.remainder(n, 2)) + 1, [fpass, fstop], window='flattop', pass_zero=False, nyq=fs / 2)

    dat2 = np.reshape(dat, [len(dat), ])
    flt_dat = np.convolve(dat2, wind)
    filtered_data = flt_dat[int(len(wind) / 2):len(flt_dat) - (int(len(wind) / 2) + 1)]
    return filtered_data


def _crop_ecg(ecg, amp=1000):
    """
    :param ecg: 
    :param amp:
    :return:
    """
    ecg[ecg > amp] = amp
    ecg[ecg < -amp] = -amp

    return ecg


def _HillTransform(raw, k_filt):
    """

    :param raw:
    :param k_filt:
    :return:
    """
    s = k_filt[:]
    N = raw.size

    if len(s) != 1:
        s = np.transpose(s)

    xe = np.abs(hilbert(s))
    h = np.true_divide(np.ones((1, 31)), 31)
    Delay = 15
    x = np.convolve(np.ravel(xe), np.ravel(h))
    x = x[np.arange(Delay, N)]

    return x


def _VariableThresh(data, SBL, TD, no_max=5):
    """
    :param data: data moving threshold is applied upon - different for different applications (e.g. plain data,
                 filtered data or post HT data)
    :param SBL: search back length for the moving threshold ;
    :param TD: threshold difficulty factor (ratio of height between mean and max)
    :param no_max:
    :return:
    """

    LD = len(data)
    thresh = np.ones(LD)

    if LD < SBL:
        return np.mean(data) * thresh

    for i in range(SBL):
        thresh[i] = np.mean(data[0:SBL - 1]) + 0.8 * np.std(data[0:SBL - 1])
    TD -= 1
    temp_max = np.zeros(no_max)
    fac = int(SBL / no_max)

    for i in range(SBL, LD, SBL):
        for j in range(1, no_max):
            temp_max[j] = np.max(data[(i - (fac * j)):(i - fac * (j - 1))])

        thresh[i:LD] = TD * (np.mean(temp_max) - np.mean(data[(i - SBL):i])) + np.mean(data[(i - SBL):i])

    return thresh


def _PeakSearch(data, raw, thr):
    """

    :param data:
    :param raw:
    :param thr:
    :return:
    """
    # sets minimum amount of values which must be above threshold
    min_L = 10
    left, right = _edges(data, thr)

    try:
        NN = left.size
    except AttributeError:
        return 0

    R_loc = np.zeros(NN)
    for i in range(NN):
        z = np.arange(int(left[i]), int(right[i]) + 1)
        if np.size(z) > min_L:
            temp1, temp2 = np.max(raw[z]), np.argmax(raw[z])
            temp3, temp4 = np.min(raw[z]), np.argmin(raw[z])

            if np.abs(temp1 - raw[int(left[i])]) > np.abs(temp3 - raw[int(left[i])]):
                R_loc[i] = temp2
            else:
                R_loc[i] = temp4
            R_loc[i] = R_loc[i] + left[i]

    R_loc = R_loc[R_loc != 0]

    return R_loc


def _Correction_Al2(peaks, fs, raw, mag_lim=0.10, energy_lim=0.05, min_L=0.3):
    """

    :param peaks:
    :param fs:
    :param raw:
    :param mag_lim:
    :param energy_lim:
    :param min_L:
    :return:
    """
    try:
        min_L = int(min_L * fs)
        peaks = peaks[peaks > 10]
        peaks = peaks[peaks < (len(raw) - 10)]
    except TypeError:
        return []

    M = peaks.size

    for i in range(1, M):
        if peaks[i] - peaks[i - 1] <= min_L:
            peak1 = int(peaks[i - 1])
            peak2 = int(peaks[i])
            temp1 = np.abs(raw[peak1])
            temp2 = np.abs(raw[peak2])

            eng1 = np.sum(np.power(np.abs(np.fft.fft(raw[peak1 - 10:peak1 + 10])), 2))
            eng2 = np.sum(np.power(np.abs(np.fft.fft(raw[peak2 - 10:peak2 + 10])), 2))

            slope1 = ((raw[peak1]) - (raw[peak1 - 3])) / (3 * (1 / fs))
            slope2 = ((raw[peak2]) - (raw[peak2 - 3])) / (3 * (1 / fs))

            if (temp1 < (temp2 + mag_lim * temp2)) & (temp1 > (temp2 - mag_lim * temp2)):
                if (eng1 > (eng2 + energy_lim * eng2)) | (eng1 < (eng2 - energy_lim * eng2)):
                    if slope1 > slope2:
                        peaks[i] = 0
                    else:
                        peaks[i - 1] = 0
            else:
                if slope1 > slope2:
                    peaks[i] = 0
                else:
                    peaks[i - 1] = 0

    peaks = peaks[peaks != 0]

    return peaks


def _edges(x, thresh):
    """
    :param x:
    :param thresh:
    :return:
    """

    poss_reg = np.array(x > thresh)
    Hold = np.where(poss_reg[:-1] != poss_reg[1:])[0]

    if len(Hold) < 2:
        return 1, 2

    if not poss_reg[0]:
        Left = Hold[::2]
        Right = Hold[1::2]
    else:
        Right = Hold[::2]
        Left = np.append(1, Hold[1::2])

    if len(Left) > len(Right):
        Right = np.append(Right, x[-1])

    return Left, Right
