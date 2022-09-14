"""
Functions for the calculation of HRV metrics
Author: Meghan McConnell
"""

# ========================== FUNCTIONS =========================#
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
from scipy.signal import kaiserord, firwin, freqz, convolve, lombscargle
import scipy.linalg as lin
import warnings
from scipy.interpolate import CubicSpline
from matplotlib.figure import Figure


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

    wind = firwin((n + np.remainder(n, 2)) + 1, [pass1, pass2], window=('kaiser', beta), pass_zero=False, nyq=180)

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


def acc(TP: int, FP: int, FN: int) -> float:
    """
    Calculate accuracy of method using TP, FP, and FN scores.
    :param TP: True Positive count
    :param FP: False Positive count
    :param FN: False Negative Count
    :return: Acc
    """
    y = np.true_divide(TP, (TP + FP + FN)) * 100
    return y


def acc2(TP: int, FP: int, FN: int) -> Tuple[float, float, float, float]:
    """
    Calculate accuracy, sensitivity, positive predictability and der of method using TP, FP, and FN scores.
    :param TP: True Positive count
    :param FP: False Positive count
    :param FN: False Negative Count
    :returns: Se, Pp, Acc, DER
    """
    sen = np.true_divide(TP, (TP + FN)) * 100
    posp = np.true_divide(TP, (TP + FP)) * 100
    Acc = np.true_divide(TP, (TP + FP + FN)) * 100
    der = np.true_divide((FP + FN), TP) * 100

    return np.around(sen, 2), np.around(posp, 2), np.around(Acc, 2), np.around(der, 2)


# Acc check for created data
def test(R_pred: np.ndarray, R_act: np.ndarray, thr: int) -> Tuple[int, int, int]:
    """

    :param R_pred:
    :param R_act:
    :param thr:
    :return:
    """
    err = 0.1
    counter = 1

    R_pred = R_pred[:]
    R_act = R_act[:]

    if len(R_pred) == 1:
        R_pred = np.transpose(R_pred)

    if len(R_act) == 1:
        R_act = np.transpose(R_act)

    lp = len(R_pred)
    lr = len(R_act)

    i = 0
    j = 0
    TP = 0  # True positive - Correctly Predicted QRS peak (within accuracy of given frequency)
    FP = 0  # False positive - Idenification of a Peak which is not a peak
    FN = 0  # Missing the identification of a peak

    while (i < lp - 1) & (j < lr - 1):

        if (R_pred[i] >= (R_act[j] - thr - err)) & (R_pred[i] <= (R_act[j] + thr + err)):
            TP += 1
            i += 1
            j += 1

        elif R_pred[i] > (R_act[j] - err):
            FN += 1
            j += 1

        elif R_pred[i] < (R_act[j] + err):
            FP += 1
            i += 1

        else:
            print('Error')

        counter += 1

    return TP, FP, FN


def test2(R_pred: np.ndarray, R_act: np.ndarray, thr: int) -> Tuple[int, int, int]:
    """
    True positive - Correctly Predicted QRS peak (within accuracy of given frequency
    False positive - Idenification of a Peak which is not a peak
    False Negative -  Missing the identification of a peak
    :param R_pred:
    :param R_act:
    :param thr:
    :return:
    """
    TP = 0
    FN = 0
    M = len(R_pred)
    flag = 0
    marker = np.zeros(M)
    for i in range(len(R_act)):
        j = 0
        test_val = R_act[i]
        while j < M:
            compare = R_pred[j]
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


def Calculate_Features(R_peaks: np.ndarray, fs: int = 1, decim: int = 2):
    """

    :param R_peaks:
    :param fs:
    :param decim: Number of decimal places to return results with
    :return:
    """

    # Calculating SDNN
    R_peaks2 = R_peaks / fs  # Turn R-Peak locations to time stamps
    R_td = np.diff(R_peaks2)
    MeanRR = np.mean(R_td) * 1e3
    SDNN = np.std(R_td) * 1e3

    # Calculating SDANN
    timejump = 300  # 5 minutes
    timestamp = timejump
    runs = int(R_peaks2[-1] / timestamp)
    SDNN_5 = np.zeros(runs)
    i = 0
    while timestamp <= timejump * runs:
        section = R_peaks2[R_peaks2 <= timestamp]
        R_peaks2 = R_peaks2[R_peaks2 > timestamp]
        timestamp += timejump
        R_td_5 = np.diff(section)
        SDNN_5[i] = np.std(R_td_5)
        i += 1
    SDANN = np.mean(SDNN_5) * 1e3

    # Calculating pNN50                      pNN50 = (NN50 count) / (total NN count)
    total_NN = len(R_peaks)
    NN_50 = abs(np.diff(R_td))
    count = 0
    for i in range(len(NN_50)):
        if NN_50[i] > 0.050:
            count = count + 1
    pNN50 = (count / total_NN * 100)

    # Calculating RMSSD
    RMSSD = np.sqrt((np.sum(np.power(np.diff(R_td), 2))) / (total_NN - 1)) * 1e3

    return np.around(SDNN, decim), np.around(SDANN, decim), np.around(MeanRR, decim), np.around(RMSSD, decim),\
           np.around(pNN50, decim)


def autocorr(x, K):
    """

    :param x:
    :param K:
    :return:
    """
    N = len(x)
    rxx = np.zeros(K + 1)
    for k in range(K + 1):
        n = np.arange(N - k)
        rxx[k] += np.sum(x[n] * x[n + k])
    rxx /= N
    return rxx


def blackmanTukeyPSD(x, L, K):
    """
    Blackman Tukey method for PSD estimation. Written by Stephen So. Computes biased autocorrelations up to lag K
    :param x:
    :param L:
    :param K:
    :return:
    """
    rxx = autocorr(x, K)
    # rxx should be symmetric
    rxx = np.concatenate((np.flipud(rxx[1:]), rxx))
    P = np.abs(np.fft.fft(rxx, L))
    # return P
    return P[:int(len(x)/2)]


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


def lpcPSD(x, p, L: int = None):
    """

    :param x:
    :param p:
    :param L: Used to define length of zero padding
    :return:
    """
    N = len(x)
    if L is None or L < N:
        L = calc_zero_padding(N)

    (a, J) = lpc(x, p)
    psd = J / (np.abs(np.fft.fft(a, L)) ** 2)
    # return psd
    return psd[:int(N/2)]


def calc_zero_padding(le):
    """

    :param le:
    :return:
    """
    L = 2

    while L < le:
        L *= 2

    return L


def welchPSD(x, M, O, L=None):
    """
    Welch's Method for PSD estimation. Written by Stephen So. Welch's method.
    :param x:
    :param M: Segment length (As percentage)
    :param O: Overlap (As percentage)
    :param L: Length of Fourier Transform. Defaults to None - which calcs best length automatically.
    :return:
    """
    # M and O come in as percentages -> convert to closest possible integers:
    N = len(x)
    M = int(M / 100 * N)
    O = int(O / 100 * M)
    nSeg = int(N / O)

    if L is None or L < N:
        L = calc_zero_padding(N)

    # determine if there are not a whole number of segments
    rem = N - nSeg * O
    if rem > 0:
        x = np.concatenate((x, np.zeros(rem)))
        nSeg += 1
    P = np.zeros((nSeg, L))
    start = 0
    for i in range(nSeg):
        xs = x[start:start + M]
        # IF using windowing technique - apply it here
        P[i, :] = (np.abs(np.fft.fft(xs, L)) ** 2) / M
        start += M - O
    # compute the average of P
    Pavg = np.mean(P, axis=0)

    return Pavg[:int(N/2)]


# RECURRENCE PLOT
def RQA_plot(Matrix, N, Fig: Figure = None):
    """

    :param Matrix:
    :param N:
    :param Fig:
    :return:
    """
    xplot = np.zeros((N, N))
    yplot = np.zeros((N, N))

    RQA_plt = Fig.add_subplot(111)
    RQA_plt.clear()
    for i in range(0, len(Matrix)):
        yplot[:, i] = np.arange(1, len(Matrix) + 1) * Matrix[:, i]
        xplot[:, i] = np.ones(len(Matrix)) * (i + 1)
    RQA_plt.scatter(xplot, yplot, c='k', s=0.5)
    RQA_plt.set_title('Recurrence Plot')
    RQA_plt.set_xlabel('Heart beat (sample number)')
    RQA_plt.set_ylabel('Heart beat (sample number)')

    return Fig


def RQA(RRI, m=10, L=1, decim=2):
    """

    :param RRI:
    :param m:
    :param L:
    :param decim:
    :return:
    """

    Matrix, N = RQA_matrix(RRI=RRI, m=m, L=L)

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


def RQA_matrix(RRI, m=10, L=1):
    """

    :param RRI:
    :param m:
    :param L:
    :return:
    """
    lenx = np.size(RRI)
    RRI = np.reshape(RRI, [lenx, ])
    N = lenx - ((m - 1) * L)  # N = number of points in recurrence plot
    r = np.sqrt(m) * np.std(RRI)  # r = fixed radius (Comparison point for Euclidian distance between two vectors)
    # i.e. if ||X_i - X_j || < r then Vec(i,j) = 1
    X = np.zeros((N, m))  # X = multi dimensional process of the time series as a trajectory in m-dim space

    # Generate vector X using X_i =(x(i),x(i+L),...,x(i+(m-1)L))
    for i in range(N):
        for j in range(m):
            X[i, j] = RRI[i + (j - 1) * L]

    Matrix = np.zeros((N, N))  # Vec = recurrence plot vector

    # Determine recurrence matrix (i.e. if 'closeness' is < given radius)
    for i in range(N):
        dist = np.sqrt(np.sum(np.power((X[i, :] - X), 2), axis=1))
        Matrix[i, :] = dist < r

    return Matrix, N


# Poincare plot and info
def Poincare_Plot(sd1, sd2, c1, c2, x, y, Fig: Figure = None):
    """

    :param sd1:
    :param sd2:
    :param c1:
    :param c2:
    :param x:
    :param y:
    :param Fig:
    """
    A = sd2 * np.cos(np.pi / 4)
    B = sd1 * np.sin(np.pi / 4)

    ellipse = patch.Ellipse((c1, c2), sd2 * 2, sd1 * 2, 45, facecolor="none", edgecolor="b", linewidth=2, zorder=5)
    poin_plt = Fig.add_subplot(111)
    poin_plt.clear()
    if poin_plt.axes.axes.yaxis_inverted() == 1:
        poin_plt.axes.axes.invert_yaxis()
    poin_plt.plot(x, y, 'ko', markersize=3, zorder=0)  # ,'MarkerFaceColor', 'k', 'MarkerSize',4)
    poin_plt.add_patch(ellipse)
    poin_plt.set_title('Poincare Plot')
    poin_plt.set_xlabel('RRI$_{n}$ (s)')
    poin_plt.set_ylabel('RRI$_{n+1}$ (s)')
    poin_plt.plot([c1, c1 + A], [c2, c2 + A], 'm', label="SD1", zorder=10)
    poin_plt.plot([c1 - 4 * A, c1 + 4 * A], [c2 - 4 * A, c2 + 4 * A], 'b', dashes=[6, 2])
    poin_plt.plot([c1, c1 - B], [c2, c2 + B], 'c', label="SD2", zorder=10)
    poin_plt.plot([c1 + B * 4, c1 - B * 4], [c2 - 4 * B, c2 + 4 * B], 'b', dashes=[4, 2, 10, 2])
    poin_plt.legend()


def Poincare(RRI, decim=3):
    """
    :param RRI: RR interval series
    :param decim:
    :return:
    """
    lenx = np.size(RRI)
    RRI = np.reshape(RRI, [lenx, ])
    x = RRI[0:lenx - 1]
    y = RRI[1:lenx]
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


def DFA(RRI, min_box=4, max_box=64, inc=1, cop=12, decim=3):
    """

    :param RRI:
    :param min_box: minimum point
    :param max_box: max point
    :param inc: increment/step size
    :param cop: cross-over point for SD1 and SD2 or up and lower division
    :param decim:
    :return:
    """

    NN = np.size(RRI)
    RRI = np.reshape(RRI, [NN, ])
    box_lengths = np.arange(min_box, max_box + 1, inc)  # Box length
    y = np.zeros(NN)
    mm = np.mean(RRI)
    y[0] = RRI[0] - mm
    for k in range(1, NN):
        y[k] = y[k - 1] + RRI[k] - mm

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

    alp1 = np.round(P1[0], decim)
    alp2 = np.round(P_2[0], decim)
    F = np.round(F, decim)

    return alp1, alp2, F


def DFA_fig(RRI, min_box=4, max_box=64, inc=1, cop=12, decim=3):
    """

    :param RRI:
    :param min_box:
    :param max_box:
    :param inc:
    :param cop:
    :param decim:
    :return:
    """
    NN = np.size(RRI)
    RRI = np.reshape(RRI, [NN, ])
    box_lengths = np.arange(min_box, max_box + 1, inc)  # Box length
    y = np.zeros(NN)
    mm = np.mean(RRI)
    y[0] = RRI[0] - mm
    for k in range(1, NN):
        y[k] = y[k - 1] + RRI[k] - mm

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
    y_new1 = np.polyval(P1, x_vals1)

    # Long-term DFA - alpha 2
    x_alp2 = box_lengths[box_lengths >= (cop + 1)]
    x_vals2 = np.log10(x_alp2)
    F_alp2 = F[len(x_alp1):len(F)]
    y_vals2 = np.log10(F_alp2)
    P_2 = np.polyfit(x_vals2, y_vals2, 1)
    y_new2 = np.polyval(P_2, x_vals2)

    a1 = np.round(P1[0], decim)
    a2 = np.round(P_2[0], decim)

    return x_vals1, y_vals1, y_new1, x_vals2, y_vals2, y_new2, a1, a2


def Freq_Analysis(Rpeaks, meth=1, decim=3, M=5, O=50, BTval=10, omega_max=500, order=100):
    """

    :param Rpeaks:
    :param meth:
    :param decim:
    :param M:
    :param O:
    :param BTval:
    :param omega_max:
    :param order:
    :return:
    """
    lenx = np.size(Rpeaks)
    Rpeaks = np.reshape(Rpeaks, [lenx, ])
    RRI = np.diff(Rpeaks)
    Rpeaks = Rpeaks[1:lenx]

    # Resample x at even intervals
    FS = 100
    cs = CubicSpline(Rpeaks, RRI)
    x_sampled = np.arange(0, np.round(Rpeaks[-1]), 1 / FS)
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
        P = welchPSD(XX, L, M, O)
        P_2 = P[1:centre + 1] / FS

    elif meth == 2:
        # Blackman-Tukey's method
        K = int(L / BTval)
        P = blackmanTukeyPSD(XX, L, K)
        P_2 = P[0:centre] / FS

    elif meth == 3:
        RRI = RRI - np.mean(RRI)
        omega = np.linspace(0.0001, np.pi * 2, omega_max)
        P_2 = lombscargle(Rpeaks, RRI, omega, normalize=False)
        f = omega / (2 * np.pi)

    else:
        #        RRI = RRI - np.mean(RRI)
        psd = lpcPSD(XX, order, L)  # psd is double-sided power spectra
        P_2 = psd[0:centre]

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
        return np.around(powVLF, decim), np.around(powLF, decim), np.around(powHF, decim), np.around(perpowVLF, decim),\
               np.around(perpowLF, decim), np.around(perpowHF, decim), np.around(peak_freq_VLF, decim),\
               np.around(peak_freq_LF, decim), np.around(peak_freq_HF, decim), np.around(LFHF, decim)
    else:
        return powVLF, powLF, powHF, perpowVLF, perpowLF, perpowHF, peak_freq_VLF, peak_freq_LF, peak_freq_HF, LFHF


def Freq_Analysis_fig(Rpeaks, meth=1, decim=3, Fig=0, M=5, O=50, BTval=10, omega_max=500, order=100):
    """

    :param Rpeaks:
    :param meth:
    :param decim:
    :param Fig:
    :param M:
    :param O:
    :param BTval:
    :param omega_max:
    :param order:
    :return:
    """
    lenx = np.size(Rpeaks)
    Rpeaks = np.reshape(Rpeaks, [lenx, ])
    RRI = np.diff(Rpeaks)
    Rpeaks = Rpeaks[1:lenx]

    # Resample x at even intervals
    FS = 100
    cs = CubicSpline(Rpeaks, RRI)
    x_sampled = np.arange(0, np.round(Rpeaks[-1]), 1 / FS)
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
        P = welchPSD(XX, L, M, O)
        P2 = P[0:centre] / FS

    elif meth == 2:
        # Blackman-Tukey's method

        K = int(L / BTval)
        P = blackmanTukeyPSD(XX, L, K)
        P2 = P[0:centre] / FS

    elif meth == 3:
        RRI = RRI - np.mean(RRI)
        omega = np.linspace(0.0001, np.pi * 2, omega_max)
        P2 = lombscargle(Rpeaks, RRI, omega, normalize=False)
        f = omega / (2 * np.pi)

    else:
        RRI = RRI - np.mean(RRI)
        psd = lpcPSD(RRI, order, L)  # psd is double-sided power spectra
        P2 = psd[0:centre]

    return f, P2


def savemetrics(rpeaks, preferences, t_rpeaks: np.ndarray = None):
    """
    :param rpeaks:
    :param preferences:
    :param t_rpeaks:
    :return:
    """

    # Time-domain Statistics
    SDNN, SDANN, MeanRR, RMSSD, pNN50 = Calculate_Features(Rpeakss, Fs)

    # Frequency-domain Statistics
    Rpeak_input = Rpeakss / Fs
    powVLF, powLF, powHF, perpowVLF, perpowLF, perpowHF, peak_freq_VLF, peak_freq_LF, peak_freq_HF, LF_HF_ratio = Freq_Analysis(
        Rpeak_input, meth=1, decim=3, M=welchL, O=welchO, BTval=btval_input, omega_max=omax_input, order=order)
    powVLF2, powLF2, powHF2, perpowVLF2, perpowLF2, perpowHF2, peak_freq_VLF2, peak_freq_LF2, peak_freq_HF2, LF_HF_ratio2 = Freq_Analysis(
        Rpeak_input, meth=2, decim=3, M=welchL, O=welchO, BTval=btval_input, omega_max=omax_input, order=order)
    powVLF3, powLF3, powHF3, perpowVLF3, perpowLF3, perpowHF3, peak_freq_VLF3, peak_freq_LF3, peak_freq_HF3, LF_HF_ratio3 = Freq_Analysis(
        Rpeak_input, meth=3, decim=3, M=welchL, O=welchO, BTval=btval_input, omega_max=omax_input, order=order)
    powVLF4, powLF4, powHF4, perpowVLF4, perpowLF4, perpowHF4, peak_freq_VLF4, peak_freq_LF4, peak_freq_HF4, LF_HF_ratio4 = Freq_Analysis(
        Rpeak_input, meth=4, decim=3, M=welchL, O=welchO, BTval=btval_input, omega_max=omax_input, order=order)

    mbox = int(Preferences[10])
    print(mbox)
    COP = int(Preferences[11])
    print(COP)
    m2box = int(Preferences[12])
    print(m2box)
    In = int(Preferences[13])
    print(In)

    # Nonlinear statistics
    RRI = np.diff(Rpeak_input)
    #    (pvp, Min=self.minbox, Max=self.maxbox, Inc=self.increm, COP=self.copbox)
    REC, DET, LAM, Lmean, Lmax, Vmean, Vmax = RQA_plot(RRI, m=int(Preferences[14]), L=int(Preferences[15]))
    SD1, SD2, c1, c2 = Poincare(RRI)
    alp1, alp2, F = DFA(RRI, min_box=mbox, max_box=m2box, cop=COP, inc=In, decim=3)

    if windows_compile:
        saveroot = filedialog.asksaveasfilename(title="Select file", defaultextension=".*",
                                                filetypes=(("text files", "*.txt"), ("all files", "*.*")))
    if linux_compile:
        saveroot = filedialog.asksaveasfilename(title="Select file",
                                                filetypes=(("text files", "*.txt"), ("all files", "*.*")))
    fname, file_extension = os.path.splitext(saveroot)

    if file_extension == '.h5':
        fileh = tb.open_file(saveroot, mode='w')
        table = fileh.create_table(fileh.root, 'Time_Domain_Metrics', TimeDomain, "HRV analysis - Time-Domain metrics")
        table.append([(SDNN, SDANN, MeanRR, RMSSD, pNN50)])

        table2 = fileh.create_table(fileh.root, 'Frequency_Domain_Metrics', FrequencyDomain,
                                    "HRV analysis - Frequency-Domain metrics")
        table2.append([(powVLF, powLF, powHF, perpowVLF, perpowLF, perpowHF, peak_freq_VLF, peak_freq_LF, peak_freq_HF,
                        LF_HF_ratio),
                       (powVLF2, powLF2, powHF2, perpowVLF2, perpowLF2, perpowHF2, peak_freq_VLF2, peak_freq_LF2,
                        peak_freq_HF2, LF_HF_ratio2),
                       (powVLF3, powLF3, powHF3, perpowVLF3, perpowLF3, perpowHF3, peak_freq_VLF3, peak_freq_LF3,
                        peak_freq_HF3, LF_HF_ratio3),
                       (powVLF4, powLF4, powHF4, perpowVLF4, perpowLF4, perpowHF4, peak_freq_VLF4, peak_freq_LF4,
                        peak_freq_HF4, LF_HF_ratio4)])

        table3 = fileh.create_table(fileh.root, 'Nonlinear_Metrics', NonlinearMets, "HRV analysis - Nonlinear metrics")
        table3.append([(REC, DET, LAM, Lmean, Lmax, Vmean, Vmax, SD1, SD2, alp1, alp2)])

        fileh.close()


    elif file_extension == '.txt':
        with open(saveroot, "w") as text_file:
            if ((labelled_flag == 1) & (loaded_ann == 1)):
                print(saveroot, '\n\n \t Quantified HRV Metrics and R-peak detection method analysis \n',
                      file=text_file)
            else:
                print(saveroot, '\n\n \t\t\t Quantified HRV Metrics', file=text_file)

            print('Time-Domain ', file=text_file)
            print(f"  SDNN (ms): \t\t {SDNN} \n  SDANN (ms): \t\t {SDANN} \n  Mean RRI (ms): \t {MeanRR} \n" +
                  f"  RMSSD (ms): \t\t {RMSSD} \n  pNN50 (ms): \t\t {pNN50} \n", file=text_file)
            print('Frequency-Domain \t Welch \t BTuk \t LScarg\t AutoR', file=text_file)
            print(' Absolute Power', file=text_file)
            print(f"  VLF (s^2/Hz): \t {powVLF} \t {powVLF2} \t {powVLF3} \t {powVLF4} \n" +
                  f"  LF (s^2/Hz): \t\t {powLF} \t {powLF2} \t {powLF3} \t {powLF4} \n" +
                  f"  HF (s^2/Hz): \t\t {powHF} \t {powHF2} \t {powHF3} \t {powHF4} \n", file=text_file)
            print(' Percentage Power', file=text_file)
            print(f"  VLF (%): \t\t {perpowVLF} \t {perpowVLF2} \t {perpowVLF3} \t {perpowVLF4} \n" +
                  f"  LF (%): \t\t {perpowLF} \t {perpowLF2} \t {perpowLF3} \t {perpowLF4} \n" +
                  f"  HF (%): \t\t {perpowHF} \t {perpowHF2} \t {perpowHF3} \t {perpowHF4} \n", file=text_file)
            print(' Peak Frequency', file=text_file)
            print(f"  VLF (Hz): \t\t {peak_freq_VLF} \t {peak_freq_VLF2} \t {peak_freq_VLF3} \t {peak_freq_VLF4} \n" +
                  f"  LF (Hz): \t\t {peak_freq_LF} \t {peak_freq_LF2} \t {peak_freq_LF3} \t {peak_freq_LF4} \n" +
                  f"  HF (Hz): \t\t {peak_freq_HF} \t {peak_freq_HF2} \t {peak_freq_HF3} \t {peak_freq_HF4} \n",
                  file=text_file)
            print(' Frequency Ratio', file=text_file)
            print(f"  LF/HF (Hz): \t\t {LF_HF_ratio} \t {LF_HF_ratio2} \t {LF_HF_ratio3} \t {LF_HF_ratio4} \n",
                  file=text_file)
            print('Nonlinear Metrics ', file=text_file)
            print(' Recurrence Analysis', file=text_file)
            print(
                f"  REC (%): \t\t {REC} \n  DET (%): \t\t {DET} \n  LAM (%): \t\t {LAM} \n  Lmean (bts): \t\t {Lmean} \n" +
                f"  Lmax (bts): \t\t {Lmax} \n  Vmean (bts): \t\t {Vmean} \n  Vmax (bts): \t\t {Vmax} \n",
                file=text_file)
            print(' Poincare Analysis', file=text_file)
            print(f"  SD1 (%): \t\t {SD1} \n  SD2 (%): \t\t {SD2} \n", file=text_file)
            print(' Detrended Fluctuation Analysis', file=text_file)
            print(f"  alpha1 (%): \t\t {alp1} \n  alpha2 (%): \t\t {alp2} \n", file=text_file)

    elif file_extension == '.mat':

        metrics = np.zeros((3,), dtype=np.object)
        metrics = {}
        metrics['TimeDomain'] = {}
        metrics['TimeDomain']['SDNN'] = SDNN
        metrics['TimeDomain']['SDANN'] = SDANN
        metrics['TimeDomain']['MeanRR'] = MeanRR
        metrics['TimeDomain']['RMSSD'] = RMSSD
        metrics['TimeDomain']['pNN50'] = pNN50
        metrics['FrequencyDomain'] = {}
        metrics['FrequencyDomain']['VLF_power'] = powVLF
        metrics['FrequencyDomain']['LF_power'] = powLF
        metrics['FrequencyDomain']['HF_power'] = powHF
        metrics['FrequencyDomain']['VLF_P_power'] = perpowVLF
        metrics['FrequencyDomain']['LF_P_power'] = perpowLF
        metrics['FrequencyDomain']['HF_P_power'] = perpowHF
        metrics['FrequencyDomain']['VLF_PF'] = peak_freq_VLF
        metrics['FrequencyDomain']['LF_PF'] = peak_freq_LF
        metrics['FrequencyDomain']['HF_PF'] = peak_freq_HF
        metrics['FrequencyDomain']['LFHFRatio'] = LF_HF_ratio
        metrics['Nonlinear'] = {}
        metrics['Nonlinear']['Recurrence'] = REC
        metrics['Nonlinear']['Determinism'] = DET
        metrics['Nonlinear']['Laminarity'] = LAM
        metrics['Nonlinear']['L_mean'] = Lmean
        metrics['Nonlinear']['L_max'] = Lmax
        metrics['Nonlinear']['V_mean'] = Vmean
        metrics['Nonlinear']['V_max'] = Vmax
        metrics['Nonlinear']['SD1'] = SD1
        metrics['Nonlinear']['SD2'] = SD2
        metrics['Nonlinear']['Alpha1'] = alp1
        metrics['Nonlinear']['Alpha2'] = alp2

        sio.savemat(saveroot, {'Metrics': metrics})

    else:
        print('Cannot export this file type')

