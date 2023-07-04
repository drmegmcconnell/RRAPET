"""
Functions for the calculation of HRV metrics
Author: Meghan McConnell
Date: July 2023
"""

# ========================== FUNCTIONS =========================#
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import kaiserord, firwin, freqz, convolve, lombscargle
import scipy.linalg as lin
import warnings
from scipy.interpolate import CubicSpline


def bandpassKaiser(dat, fs: int, pass1: int, pass2: int, viewfilter: bool = False):
    """

    :param dat: ECG single lead time series
    :param fs: Sampling frequency of ECG
    :param pass1: Pass-band cut-off frequency (high-pass)
    :param pass2: Pass-band cut-off frequency (low-pass)
    :param viewfilter: Set to True to view filter. Default is False.
    :return:
    """
    n, beta = kaiserord(401, 0.1)

    wind = firwin((n + np.remainder(n, 2)) + 1, [pass1, pass2], window=('kaiser', beta), pass_zero='False', nyq=180)

    if viewfilter:
        w, h = freqz(wind)
        plt.figure(2)
        plt.plot(0.5 * fs * w / np.pi, np.abs(h), 'b')
        plt.xlim(0, 0.5 * fs)
        plt.title("BandPass Filter Frequency Response")
        plt.xlabel('Frequency [Hz]')
        plt.grid()

    wind = np.reshape(wind, [len(wind), 1])

    flt = convolve(dat, wind, mode='full')

    s = len(flt)

    filtered_dat = flt[(int(n / 2) + 1):(s - int(n / 2) - 1)]

    return filtered_dat


def acc(tp: int, fp: int, fn: int) -> float:
    """
    Calculate accuracy of method using TP, FP, and FN scores.
    :param tp: True Positive count
    :param fp: False Positive count
    :param fn: False Negative Count
    :return: Acc
    """
    y = np.true_divide(tp, (tp + fp + fn)) * 100
    return y


def acc2(tp: int, fp: int, fn: int) -> Tuple[float, float, float, float]:
    """
    Calculate accuracy, sensitivity, positive predictability and der of method using TP, FP, and FN scores.
    :param tp: True Positive count
    :param fp: False Positive count
    :param fn: False Negative Count
    :returns: Se, Pp, Acc, DER
    """
    sen = np.true_divide(tp, (tp + fn)) * 100
    posp = np.true_divide(tp, (tp + fp)) * 100
    Acc = np.true_divide(tp, (tp + fp + fn)) * 100
    der = np.true_divide((tp + fn), tp) * 100

    return round(sen, 2), round(posp, 2), round(Acc, 2), round(der, 2)


# Acc check for created data
def test(r_pred: np.ndarray, r_act: np.ndarray, thr: int) -> Tuple[int, int, int]:
    """

    :param r_pred:
    :param r_act:
    :param thr:
    :return:
    """
    err = 0.1
    counter = 1

    r_pred = r_pred[:]
    r_act = r_act[:]

    if len(r_pred) == 1:
        r_pred = np.transpose(r_pred)

    if len(r_act) == 1:
        r_act = np.transpose(r_act)

    lp = len(r_pred)
    lr = len(r_act)

    i = 0
    j = 0
    TP = 0  # True positive - Correctly Predicted QRS peak (within accuracy of given frequency)
    FP = 0  # False positive - Idenification of a Peak which is not a peak
    FN = 0  # Missing the identification of a peak

    while (i < lp - 1) & (j < lr - 1):

        if (r_pred[i] >= (r_act[j] - thr - err)) & (r_pred[i] <= (r_act[j] + thr + err)):
            TP += 1
            i += 1
            j += 1

        elif r_pred[i] > (r_act[j] - err):
            FN += 1
            j += 1

        elif r_pred[i] < (r_act[j] + err):
            FP += 1
            i += 1

        else:
            print('Error')

        counter += 1

    return TP, FP, FN


def test2(r_pred: np.ndarray, r_act: np.ndarray, thr: int) -> Tuple[int, int, int]:
    """
    True positive - Correctly Predicted QRS peak (within accuracy of given frequency
    False positive - Idenification of a Peak which is not a peak
    False Negative -  Missing the identification of a peak
    :param r_pred:
    :param r_act:
    :param thr:
    :return:
    """
    TP = 0
    FN = 0
    M = len(r_pred)
    flag = 0
    marker = np.zeros(M)
    for i in range(len(r_act)):
        j = 0
        test_val = r_act[i]
        while j < M:
            compare = r_pred[j]
            flag = 0

            edge1 = test_val - thr
            edge2 = test_val + thr
            if compare > edge1:
                if compare < edge2:
                    TP = TP + 1
                    marker[j] = 1
                    j = M
                    flag = 1

            j = j + 1

        if flag != 1:
            FN = FN + 1
            flag = 1

    marker = marker[marker != 1]
    FP = len(marker)

    return TP, FP, FN


def Calculate_Features(r_peaks: np.ndarray = None, fs: int = 1, decim: int = 2):
    """

    :param r_peaks:
    :param fs:
    :param decim: Number of decimal places to return results with
    :return:
    """

    # Calculating SDNN
    rloc = r_peaks / fs  # Turn R-Peak locations to time stamps
    R_td = np.diff(rloc)
    MeanRR = np.mean(R_td) * 1e3
    SDNN = np.std(R_td) * 1e3

    # Calculating SDANN
    timejump = 300  # 5 minutes
    timestamp = timejump
    runs = int(rloc[-1] / timestamp)
    SDNN_5 = np.zeros(runs)
    i = 0
    while timestamp <= timejump * runs:
        section = rloc[rloc <= timestamp]
        rloc = rloc[rloc > timestamp]
        timestamp += timejump
        R_td_5 = np.diff(section)
        SDNN_5[i] = np.std(R_td_5)
        i += 1
    SDANN = np.mean(SDNN_5) * 1e3

    # Calculating pNN50                      pNN50 = (NN50 count) / (total NN count)
    total_NN = len(rloc)
    NN_50 = abs(np.diff(R_td))
    count = 0
    for i in range(len(NN_50)):
        if NN_50[i] > 0.050:
            count = count + 1
    pNN50 = (count / total_NN * 100)

    # Calculating RMSSD
    RMSSD = np.sqrt((np.sum(np.power(np.diff(R_td), 2))) / (total_NN - 1)) * 1e3

    return round(SDNN, decim), round(SDANN, decim), round(MeanRR, decim), round(RMSSD, decim), round(pNN50, decim)


def autocorr(x, k_):
    """

    :param x:
    :param k_:
    :return:
    """
    N = len(x)
    rxx = np.zeros(k_ + 1)
    for k in range(k_ + 1):
        n = np.arange(N - k)
        rxx[k] += np.sum(x[n] * x[n + k])
    rxx /= N
    return rxx


def blackmanTukeyPSD(x, l_, k_):
    """
    Blackman Tukey method for PSD estimation. Written by Stephen So. Computes biased autocorrelations up to lag k_
    :param x:
    :param l_:
    :param k_:
    :return:
    """
    rxx = autocorr(x, k_)
    # rxx should be symmetric
    rxx = np.concatenate((np.flipud(rxx[1:]), rxx))
    P = np.abs(np.fft.fft(rxx, l_))
    # return P
    return P[:int(len(x) / 2)]


def lpc(x, p):
    """
    LPC method for PSD estimation. Written by Stephen So. Computes Autoregression where p is the LPC/AR order
    :param x:
    :param p:
    :return:
    """
    rxx = autocorr(x, p + 1)
    A = lin.toeplitz(rxx[0:p])
    b = -rxx[1:(p + 1)]
    a = lin.solve(A, b)
    a = np.concatenate(([1], a))
    k = np.arange(p + 1)
    J = np.sum(rxx[k] * a[k])
    return a, J


def lpcPSD(x, p, l_: int = None):
    """

    :param x:
    :param p:
    :param l_: Used to define length of zero padding
    :return:
    """
    N = len(x)
    if l_ is None or l_ < N:
        l_ = calc_zero_padding(N)

    (a, J) = lpc(x, p)
    psd = J / (np.abs(np.fft.fft(a, l_)) ** 2)
    # return psd
    return psd[:int(N / 2)]


def calc_zero_padding(le):
    """

    :param le:
    :return:
    """
    L = 2

    while L < le:
        L *= 2

    return L


def welchPSD(x, m, o, l_=None):
    """
    Welch's Method for PSD estimation. Written by Stephen So. Welch's method.
    :param x:
    :param m: Segment length (As percentage)
    :param o: Overlap (As percentage)
    :param l_: Length of Fourier Transform. Defaults to None - which calcs best length automatically.
    :return:
    """
    # M and O come in as percentages -> convert to closest possible integers:
    N = len(x)
    m = int(m / 100 * N)
    o = int(o / 100 * m)
    nSeg = int(N / o)

    if l_ is None or l_ < N:
        l_ = calc_zero_padding(N)

    # determine if there are not a whole number of segments
    rem = N - nSeg * o
    if rem > 0:
        x = np.concatenate((x, np.zeros(rem)))
        nSeg += 1
    P = np.zeros((nSeg, l_))
    start = 0
    for i in range(nSeg):
        xs = x[start:start + m]
        # IF using windowing technique - apply it here
        P[i, :] = (np.abs(np.fft.fft(xs, l_)) ** 2) / m
        start += m - o
    # compute the average of P
    Pavg = np.mean(P, axis=0)

    return Pavg[:int(N / 2)]


def RQA_matrix(rri, m=10, l_=1):
    """

    :param rri:
    :param m:
    :param l_:
    :return:
    """
    lenx = np.size(rri)
    rri = np.reshape(rri, [lenx, ])
    N = lenx - ((m - 1) * l_)  # N = number of points in recurrence plot
    r = np.sqrt(m) * np.std(rri)  # r = fixed radius (Comparison point for Euclidian distance between two vectors)
    # i.e. if ||X_i - X_j || < r then Vec(i,j) = 1
    X = np.zeros((N, m))  # X = multi dimensional process of the time series as a trajectory in m-dim space

    # Generate vector X using X_i =(x(i),x(i+L),...,x(i+(m-1)L))
    for i in range(N):
        for j in range(m):
            X[i, j] = rri[i + (j - 1) * l_]

    Matrix = np.zeros((N, N))  # Vec = recurrence plot vector

    # Determine recurrence matrix (i.e. if 'closeness' is < given radius)
    for i in range(N):
        dist = np.sqrt(np.sum(np.power((X[i, :] - X), 2), axis=1))
        Matrix[i, :] = dist < r

    return Matrix, N


def RQA(rri, m=10, l_=1, decim=2):
    """

    :param rri:
    :param m:
    :param decim:
    :param l_:
    :return:
    """

    Matrix, N = RQA_matrix(rri=rri, m=m, l_=l_)

    # Analyse Diagonals of RP
    FlVec = np.copy(Matrix)
    diagsums = np.zeros((N, N))
    for i in range(N):
        vert = np.diag(FlVec, k=i)
        init = 0
        dsums = 0
        for j in range(len(vert)):
            if vert[j] == 1:
                init = init + 1
                if j == len(vert) & (init > 1):
                    diagsums[i, dsums] = init
            else:
                if init > 1:
                    diagsums[i, dsums] = init
                    dsums = dsums + 1
                    init = 0
                else:
                    init = 0

    V_Matrix = np.copy(Matrix)
    for i in range(N):
        for j in range(i, N):
            V_Matrix[i, j] = 0  # Zeros out half of the matrix

    vertsums = np.zeros((N, N))
    for i in range(N):
        vert = V_Matrix[:, i]
        init = 0
        vsums = 1
        for j in range(len(vert)):
            if vert[j] == 1:
                init = init + 1
                if (j == len(vert)) & (init > 1):
                    vertsums[i + 1, vsums] = init
            else:
                if init > 1:
                    vertsums[i + 1, vsums] = init
                    vsums = vsums + 1
                    init = 0
                else:
                    init = 0

    # %Calculate Features
    REC = np.sum(Matrix) / np.power(N, 2)
    diagsums = diagsums[2:N, :]
    DET = np.sum(diagsums) / (np.sum(FlVec) / 2)
    nzdiag = np.sum(diagsums > 0)  # Number of non-zero diagonals
    Lmean = np.round(np.sum(diagsums) / nzdiag, decim)
    Lmax = int(np.max(diagsums))
    LAM = np.sum(vertsums) / np.sum(V_Matrix)
    nzvert = np.sum(vertsums > 0)  # Number of non-zero verticals
    Vmean = np.round(np.sum(vertsums) / nzvert, decim)
    Vmax = int(np.max(vertsums))

    REC = '{0:.2f}'.format(REC * 100)
    DET = '{0:.2f}'.format(DET * 100)
    LAM = '{0:.2f}'.format(LAM * 100)

    return REC, DET, LAM, Lmean, Lmax, Vmean, Vmax


def Poincare(rri, decim=3):
    """
    :param rri: RR interval series
    :param decim:
    :return:
    """
    lenx = np.size(rri)
    rri = np.reshape(rri, [lenx, ])
    x = rri[0:lenx - 1]
    y = rri[1:lenx]
    c1 = np.mean(x)
    c2 = np.mean(y)

    sd1_sqed = 0.5 * np.power(np.std(np.diff(x)), 2)
    sd1 = np.sqrt(sd1_sqed)

    sd2_sqed = 2 * np.power(np.std(x), 2) - sd1_sqed
    sd2 = np.sqrt(sd2_sqed)

    if decim is not None:
        return np.round(sd1 * 1e3, decim), np.round(sd2 * 1e3, decim), c1, c2, x, y
    else:
        return sd1, sd2, c1, c2, x, y


def DFA(rri, min_box=4, max_box=64, inc=1, cop=12, decim=3, figure: bool = False):
    """

    :param rri:
    :param min_box: Optional. Minimum point. Default is 4
    :param max_box: Optional. Maximum point. Default is 64.
    :param inc: Optional. Increment/step size. Default is 1.
    :param cop: Optional. Cross-over point for SD1 and SD2 or up and lower trends. Default is 12.
    :param decim:
    :param figure:
    :return:
    """

    NN = np.size(rri)
    rri = np.reshape(rri, [NN, ])
    box_lengths = np.arange(min_box, max_box + 1, inc)  # Box length
    y = np.zeros(NN)
    mm = np.mean(rri)
    y[0] = rri[0] - mm
    for k in range(1, NN):
        y[k] = y[k - 1] + rri[k] - mm

    M = len(box_lengths)

    F = np.zeros(M)
    for q in range(M):
        n = box_lengths[q]
        N = int(np.floor(len(y) / n))
        y_n2 = np.zeros((n, N))
        y2 = np.reshape(y[0:N * n], [n, N],
                        order='F')  # Order 'F' fills column by column, whereas order 'C' fills row by row
        k = np.reshape(np.arange(N * n), [n, N])
        for m in range(N):
            P = np.polyfit(k[:, m], y2[:, m], 1)
            y_n2[:, m] = np.polyval(P, k[:, m])
        if NN > N * n:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=np.RankWarning)
                y3 = y[N * n:len(y)]
                k = np.arange(N * n, NN)
                P = np.polyfit(k, y3, 1)
                y_n3 = np.polyval(P, k)
                y_n = np.append(y_n2.flatten('F'), y_n3.flatten('F'))
        else:
            y_n = y_n2.flatten('F')

        F[q] = np.sqrt(np.sum(np.power((y.flatten('F') - y_n.flatten('F')), 2)) / NN)
    # Short-term DFA - alpha 1
    x_alp1 = box_lengths[box_lengths <= cop]
    F_alp1 = F[0:len(x_alp1)]
    x_vals1 = np.log10(x_alp1)
    y_vals1 = np.log10(F_alp1)
    P1 = np.polyfit(x_vals1, y_vals1, 1)

    # Long-term DFA - alpha 2
    x_alp2 = box_lengths[box_lengths >= (cop + 1)]
    x_vals2 = np.log10(x_alp2)
    F_alp2 = F[len(x_alp1):len(F)]
    y_vals2 = np.log10(F_alp2)
    P_2 = np.polyfit(x_vals2, y_vals2, 1)

    if figure is True:
        y_new1 = np.polyval(P1, x_vals1)
        y_new2 = np.polyval(P_2, x_vals2)
        a1 = np.round(P1[0], decim)
        a2 = np.round(P_2[0], decim)

        return x_vals1, y_vals1, y_new1, x_vals2, y_vals2, y_new2, a1, a2

    else:

        alp1 = np.round(P1[0], decim)
        alp2 = np.round(P_2[0], decim)
        F = np.round(F, decim)

        return alp1, alp2, F


def Freq_Analysis(r_peaks, meth=1, decim=3, m=5, o=50, bt_val=10, omega_max=500, order=100, figure: bool = False):
    """

    :param r_peaks:
    :param meth:
    :param decim:
    :param m:
    :param o:
    :param bt_val:
    :param omega_max:
    :param order:
    :param figure:
    :return:
    """
    lenx = np.size(r_peaks)
    r_peaks = np.reshape(r_peaks, [lenx, ])
    RRI = np.diff(r_peaks)
    r_peaks = r_peaks[1:lenx]

    # Resample x at even intervals
    FS = 100
    cs = CubicSpline(r_peaks, RRI)
    x_sampled = np.arange(0, np.round(r_peaks[-1]), 1 / FS)
    RRI_sampled = cs(x_sampled)
    N = len(RRI_sampled)
    xt = RRI_sampled - np.mean(RRI_sampled)

    L = calc_zero_padding(N)
    f = np.arange(L) / L * FS
    centre = int(L / 2 + 1)
    f = f[0:centre]
    XX = np.concatenate((xt, np.zeros(L - N)))

    # MAKE A WAY TO CHOOSE METHOD e.g. method1, method2, method3, etc.
    if meth == 1:
        # Welch method (M = segement length, O = overlap)
        P = welchPSD(XX, L, m, o)
        P_2 = P[1:centre + 1] / FS

    elif meth == 2:
        # Blackman-Tukey's method
        K = int(L / bt_val)
        P = blackmanTukeyPSD(XX, L, K)
        P_2 = P[0:centre] / FS

    elif meth == 3:
        RRI = RRI - np.mean(RRI)
        omega = np.linspace(0.0001, np.pi * 2, omega_max)
        P_2 = lombscargle(r_peaks, RRI, omega, normalize=False)
        f = omega / (2 * np.pi)

    else:
        #        RRI = RRI - np.mean(RRI)
        psd = lpcPSD(XX, order, L)  # psd is double-sided power spectra
        P_2 = psd[0:centre]

    if figure:
        return f, P_2

    else:

        # Calculate parameters
        # Power in VLF, LF, & HF frequency ranges
        VLF_upperlim = len(f[f < 0.04])
        LF_upperlim = len(f[f < 0.15])
        HF_upperlim = len(f[f < 0.4])
        powVLF = np.sum(P_2[0:VLF_upperlim]) * 1e3  # Convert to milliseconds
        powLF = np.sum(P_2[VLF_upperlim:LF_upperlim]) * 1e3  # Convert to milliseconds
        powHF = np.sum(P_2[LF_upperlim:HF_upperlim]) * 1e3  # Convert to milliseconds
        perpowVLF = powVLF / (powVLF + powLF + powHF) * 100
        perpowLF = powLF / (powVLF + powLF + powHF) * 100
        perpowHF = powHF / (powVLF + powLF + powHF) * 100

        # Peak Frequencies
        posVLF = np.argmax(P_2[0:VLF_upperlim])
        peak_freq_VLF = f[posVLF]
        posLF = np.argmax(P_2[VLF_upperlim:LF_upperlim])
        peak_freq_LF = f[posLF + VLF_upperlim]
        posHF = np.argmax(P_2[LF_upperlim:HF_upperlim])
        peak_freq_HF = f[posHF + LF_upperlim]
        LFHF = np.true_divide(powLF, powHF)

        if decim is not None:
            return round(powVLF, decim), round(powLF, decim), round(powHF, decim), round(perpowVLF, decim), \
                   round(perpowLF, decim), round(perpowHF, decim), round(peak_freq_VLF, decim), \
                   round(peak_freq_LF, decim), round(peak_freq_HF, decim), round(LFHF, decim)
        else:
            return powVLF, powLF, powHF, perpowVLF, perpowLF, perpowHF, peak_freq_VLF, peak_freq_LF, peak_freq_HF, LFHF
