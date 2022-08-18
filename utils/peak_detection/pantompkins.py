"""
R-peak Detection Methods
Pan-Tompkin
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, lfilter, butter, filtfilt
from .detector import detect_peaks


# PANTOMPKINS IMPLEMENTED IN PYTHON FROM MATLAB SCRIPT
def pan_tompkin(ecg, fs, gr=0):
    """

    :param ecg: ECG waveform as 1D signal (assumes 1 lead)
    :param fs: Sampling frequency of ECG signal
    :param gr: Return graphs. 1 = True
    :return:
    """
    plt.close('all')
    if isinstance(ecg, list) == 0:
        print('ECG must be a row or column vector')

    ecg = ecg[:]

    # ~~~~~~~~~~~~~~ INITIALISE VARIABLES ~~~~~~~~~~~~~~~~~~~#
    qrs_c = []  # Amplitude of R
    qrs_i = []  # Index
    nois_c = []
    nois_i = []
    delay = 0
    skip = 0  # Becomes one when a T wave is detected
    m_selected_RR = 0
    mean_RR = 0
    qrs_i_raw = []
    qrs_amp_raw = []
    ser_back = 0
    SIGL_buf = []
    NOISL_buf = []
    THRS_buf = []
    SIGL_buf1 = []
    NOISL_buf1 = []
    THRS_buf1 = []

    # ax1 = plt.subplot2grid((3, 3), (0, 0))
    # ax2 = plt.subplot2grid((3, 3), (0, 1), colspan=2)
    # ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
    # ax4 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)

    # ax1 = plt.subplot(221)
    # ax2 = plt.subplot(223)
    # ax3 = plt.subplot(122)

    # ~~~~~~~~~~~~~~ Plot differently based on filtering settings ~~~~~~~~~~~~~~~~~~~#
    if gr:
        f, axarr = plt.subplots(3, 2)
        if fs == 200:
            axarr[0, 0].plot(ecg)
            axarr[0, 0].set_title('Raw ECG Signal')
        else:
            axarr[0, 0].plot(ecg)
            axarr[0, 0].set_title('Raw ECG Signal')
    else:
        axarr = None
        # axarr[0, 1].plot(ecg)
        # axarr[0, 1].set_title('Raw ECG Signal')
    # f.subplots_adjust(hspace=0.3)

    # ~~~~~~~~~~~~~~ Noise cancelation (Filtering) % Filters (Filter in between 5-15 Hz) ~~~~~~~~~~~~~~~~~~~#
    if fs == 200:

        # ~~~~~~~~~~~~~~ Low Pass Filter  H(z) = ((1 - z^(-6))^2)/(1 - z^(-1))^2 ~~~~~~~~~~~~~~~~~~~#
        b = list(np.float_([1, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 1]))
        a = [1.0, -2.0, 1.0]
        x = np.zeros(13)
        x[0] = 1.0
        h_l = lfilter(b, a, x)

        ecg_l = convolve(ecg, h_l)
        ecg_l = ecg_l / np.max(np.abs(ecg_l))
        delay = 6  # based on the paper
        if gr:
            axarr[0, 1].plot(ecg_l)
            axarr[0, 1].set_title('Low Pass Filtered')

        # ~~~~~~~~~~~~~~ High Pass filter H(z) = (-1+32z^(-16)+z^(-32))/(1+z^(-1)) ~~~~~~~~~~~~~~~~~~~#

        b = list(np.float_(
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, -32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
        a = [1.0, -1.0]
        x = np.zeros(33)
        x[0] = 1.0
        h_h = lfilter(b, a, x)
        ecg_h = convolve(ecg_l, h_h)
        ecg_h = ecg_h / np.max(np.abs(ecg_h))
        delay = delay + 16

        if gr:
            axarr[1, 0].plot(ecg_h)
            axarr[1, 0].set_title('High Pass Filtered')

    else:
        low = 5.0 * 2 / fs  # Cuttoff low frequency to get rid of baseline wander
        high = 15.0 * 2 / fs  # Cuttoff frequency to discard high frequency noise
        N = 3  # order of 3 less processing
        b, a = butter(N, (low, high), btype='band')  # bandpass filtering
        ecg_h = filtfilt(b, a, ecg, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1))
        ecg_h = ecg_h / np.max(np.abs(ecg_h))

        if gr:
            axarr[1, 0].plot(ecg_h)
            axarr[1, 0].set_title('Band Pass Filtered')

    # ~~~~~~~~~~~~~~  derivative filter H(z) = (1/8T)(-z^(-2) - 2z^(-1) + 2z + z^(2)) ~~~~~~~~~~~~~~~~~~~#
    h_d = np.array([-1, -2, 0, 2, 1]) / 8
    ecg_d = convolve(ecg_h, h_d)
    ecg_d = ecg_d / np.max(ecg_d)
    delay = delay + 2  # delay of derivative filter 2 samples
    if gr:
        axarr[1, 1].plot(ecg_d)
        axarr[1, 1].set_title('Filtered with the derivative filter')

    # ~~~~~~~~~~~~~~  Squaring nonlinearly enhance the dominant peaks ~~~~~~~~~~~~~~~~~~~#
    ecg_s = np.power(ecg_d, 2)
    if gr:
        axarr[2, 0].plot(ecg_s)
        axarr[2, 0].set_title('Squared')

    # ~~~~~~~~~~~~~~ Moving average Y(nt) = (1/N)[x(nT-(N - 1)T)+ x(nT - (N - 2)T)+...+x(nT)] ~~~~~~~~~~~~~~~~~~~#
    ecg_m = convolve(ecg_s, np.ones(int(0.150 * fs)) / int(0.150 * fs))
    delay = delay + 15
    if gr:
        axarr[2, 1].plot(ecg_m)
        axarr[2, 1].set_title('Averaged with 30 samples length')
        # ('Black noise, Green Adaptive Threshold, RED Sig Level, Red circles QRS adaptive threshold')

    # ~~~~~~~~~~~~~~ Fiducial Marker Detection ~~~~~~~~~~~~~~~~~~~#
    locs = detect_peaks(ecg_m, mpd=int(0.2 * fs))
    pks = ecg_m[locs]

    # ~~~~~~~~~~~~~~ Init training phase (len = 2 sec) to determine the THR_SIG and THR_NOISE ~~~~~~~~~~~~~~~~~~~#
    THR_SIG = np.max(ecg_m[1:2 * fs]) * 1 / 3  # 0.25 of the max amplitude
    THR_NOISE = np.mean(ecg_m[1:2 * fs]) * 1 / 2  # 0.5 of the mean signal is considered to be noise
    SIG_LEV = THR_SIG
    NOISE_LEV = THR_NOISE

    # ~~~~~~~~~~~~~~ Initialize bandpath filter threshold(2 seconds of the bandpass signal) ~~~~~~~~~~~~~~~~~~~#
    THR_SIG1 = np.max(ecg_h[1:2 * fs]) * 1 / 3  # 0.25 of the max amplitude
    THR_NOISE1 = np.mean(ecg_h[1:2 * fs]) * 1 / 2
    SIG_LEV1 = THR_SIG1  # Signal level in Bandpassed filter
    NOISE_LEV1 = THR_NOISE1  # Noise level in Bandpassed filter

    # ~~~~~~~~~~~~~~ Thresholding and online desicion rule ~~~~~~~~~~~~~~~~~~~#

    for i in range(len(pks)):
        sbl = int(0.150 * fs)  # Length of the moving window
        sbl2 = int(0.200 * fs)
        # ~~~~~~~~~~~~~~ locate the corresponding peak in the filtered signal  ~~~~~~~~~~~~~~~~~~~#
        if ((locs[i] - sbl) >= 1) & (locs[i] <= len(ecg_h)):
            y_i = np.max(ecg_h[(locs[i] - sbl) - 1:locs[i] + 1])
            x_i = np.argmax(ecg_h[(locs[i] - sbl) - 1:locs[i] + 1])

        else:
            if i == 1:
                y_i = np.max(ecg_h[0:locs[i] + 1])
                x_i = np.argmax(ecg_h[0:locs[i] + 1])
                ser_back = 1
            else:  # elif locs[i] >= len(ecg_h)
                y_i = np.max(ecg_h[(locs[i] - sbl) - 1:-1])
                x_i = np.argmax(ecg_h[(locs[i] - sbl) - 1:-1])

        # ~~~~~~~~~~~~~~ update the HR (Two HR means one the most recent and the other selected)  ~~~~~~~~~~~~~~~~~~~#
        if len(qrs_c) >= 9:
            diffRR = np.diff(qrs_i[-8:len(qrs_i)])  # calculate RR interval
            mean_RR = np.mean(diffRR)  # calculate the mean of 8 previous R waves interval
            comp = qrs_i[-1] - qrs_i[-2]  # latest RR
            if (comp <= 0.92 * mean_RR) | (comp >= 1.16 * mean_RR):  # lower down thresholds to detect better
                THR_SIG = 0.5 * THR_SIG  # lower down thresholds to detect better in Bandpass filtered
                THR_SIG1 = 0.5 * THR_SIG1
            else:
                m_selected_RR = mean_RR  # the latest regular beats mean

        # ~~~~~~~~~~~~~~ calc the mean of the last 8 R waves to make sure that QRS is not missing ~~~~~~~~~~~~~~~~~~~#
        # (If no R detected , trigger a search back) 1.66*mean
        if m_selected_RR:
            test_m = m_selected_RR  # if the regular RR availabe use it
        elif (mean_RR == 0) & (m_selected_RR == 0):
            test_m = mean_RR
        else:
            test_m = 0

        if test_m:
            if (locs[i] - qrs_i[-1]) >= int(1.66 * test_m):  # it shows a QRS is missed
                pks_temp = np.max(
                    ecg_m[qrs_i[-1] + sbl2:locs[i] - sbl2 + 1])  # search back and locate the max in this interval
                locs_temp = np.argmax(ecg_m[qrs_i[-1] + sbl2:locs[i] - sbl2 + 1])
                locs_temp = qrs_i[-1] + sbl2 + locs_temp - 1  # location
                # locs_temp = qrs_i[-1] + sbl2 + locs_temp

                if pks_temp > THR_NOISE:
                    qrs_c = np.concatenate(qrs_c, pks_temp)
                    qrs_i = np.concatenate(qrs_i, locs_temp)

                    # find the location in filtered sig
                    if locs_temp <= len(ecg_h):
                        y_i_t = np.max(ecg_h[locs_temp - sbl:locs_temp + 1])
                        x_i_t = np.argmax(ecg_h[locs_temp - sbl:locs_temp + 1])
                    else:
                        y_i_t = np.max(ecg_h[locs_temp - sbl:len(ecg_h)])
                        x_i_t = np.argmax(ecg_h[locs_temp - sbl:len(ecg_h)])
                    # take care of bandpass signal threshold
                    if y_i_t > THR_NOISE1:
                        t1 = locs_temp - sbl + (x_i_t - 1)
                        qrs_i_raw = np.concatenate(qrs_i_raw, t1)  # save index of bandpass
                        qrs_amp_raw = np.concatenate(qrs_amp_raw, y_i_t)  # save amplitude of bandpass

                    #                    not_nois = 1
                    SIG_LEV = 0.25 * pks_temp + 0.75 * SIG_LEV
            else:
                pass
        #                not_nois = 0

        # ~~~~~~~~~~~~~~ find noise and QRS peaks ~~~~~~~~~~~~~~~~~~~#

        if pks[i] >= THR_SIG:
            sbl3 = 0.075 * fs
            # if a QRS candidate occurs within 360ms of the previous QRS the algorithm determines if its T wave or QRS
            if len(qrs_c) >= 3:
                if (locs[i] - qrs_i[-1]) <= int(0.36 * fs):
                    Slope1 = np.mean(
                        np.diff(ecg_m[locs[i] - sbl3 - 1:locs[i] + 1]))  # mean slope of the waveform at that position
                    Slope2 = np.mean(np.diff(ecg_m[qrs_i[-1] - sbl3:qrs_i[-1]]))  # mean slope of previous R wave
                    if np.abs(Slope1) <= np.abs(0.5 * Slope2):  # slope less then 0.5 of previous R
                        nois_c = np.append(nois_c, pks[i])
                        nois_i = np.append(nois_i, locs[i])
                        skip = 1  # T wave identification
                        # adjust noise level in both filtered and MVI
                        NOISE_LEV1 = 0.125 * y_i + 0.875 * NOISE_LEV1
                        NOISE_LEV = 0.125 * pks[i] + 0.875 * NOISE_LEV
                    else:
                        skip = 0

            if skip == 0:  # skip is 1 when a T wave is detected
                qrs_c = np.append(qrs_c, pks[i])
                qrs_i = np.append(qrs_i, locs[i])
                # bandpass filter check threshold
                if y_i >= THR_SIG1:
                    if ser_back:
                        qrs_i_raw = np.append(qrs_i_raw, x_i)  # save index of bandpass
                    else:
                        qrs_i_raw = np.append(qrs_i_raw, (locs[i] - sbl + (x_i - 1)))  # save index of bandpass

                    qrs_amp_raw = np.append(qrs_amp_raw, y_i)  # save amplitude of bandpass
                    SIG_LEV1 = 0.125 * y_i + 0.875 * SIG_LEV1  # adjust threshold for bandpass filtered sig
                # adjust Signal level
                SIG_LEV = 0.125 * pks[i] + 0.875 * SIG_LEV

        elif (THR_NOISE <= pks[i]) & (pks[i] < THR_SIG):

            # adjust Noise level in filtered sig
            NOISE_LEV1 = 0.125 * y_i + 0.875 * NOISE_LEV1
            # adjust Noise level in MVI
            NOISE_LEV = 0.125 * pks[i] + 0.875 * NOISE_LEV

        elif pks[i] < THR_NOISE:
            nois_c = np.append(nois_c, pks[i])
            nois_i = np.append(nois_i, locs[i])
            # noise level in filtered signal
            NOISE_LEV1 = 0.125 * y_i + 0.875 * NOISE_LEV1
            # adjust Noise level in MVI
            NOISE_LEV = 0.125 * pks[i] + 0.875 * NOISE_LEV

            # ~~~~~~~~~~~~~~ adjust the threshold with SNR ~~~~~~~~~~~~~~~~~~~#
        if (NOISE_LEV != 0) | (SIG_LEV != 0):
            THR_SIG = NOISE_LEV + 0.25 * (np.abs(SIG_LEV - NOISE_LEV))
            THR_NOISE = 0.5 * THR_SIG

        # adjust the threshold with SNR for bandpassed signal
        if (NOISE_LEV1 != 0) | (SIG_LEV1 != 0):
            THR_SIG1 = NOISE_LEV1 + 0.25 * (np.abs(SIG_LEV1 - NOISE_LEV1))
            THR_NOISE1 = 0.5 * THR_SIG1

        # take a track of thresholds of smoothed signal
        SIGL_buf = np.append(SIGL_buf, SIG_LEV)
        NOISL_buf = np.append(NOISL_buf, NOISE_LEV)
        THRS_buf = np.append(THRS_buf, THR_SIG)

        # take a track of thresholds of filtered signal
        SIGL_buf1 = np.append(SIGL_buf1, SIG_LEV1)
        NOISL_buf1 = np.append(NOISL_buf1, NOISE_LEV1)
        THRS_buf1 = np.append(THRS_buf1, THR_SIG1)

        # reset parameters
        skip = 0
        #        not_nois = 0
        ser_back = 0
        # ~~~~~~~~~~~~~~ Plotting the signals ~~~~~~~~~~~~~~~~~~~#

    if gr:
        axarr[2, 1].scatter(qrs_i, qrs_c, facecolors='none', edgecolors='m')
        axarr[2, 1].plot(locs, NOISL_buf, '--k', LineWidth=2)
        axarr[2, 1].plot(locs, SIGL_buf, '--r', LineWidth=2)
        axarr[2, 1].plot(locs, THRS_buf, '--g', LineWidth=2)
        # axarr[2, 1].legend( ['HT', 'Noise', 'Signal', 'Thr'])
    if gr:
        f2, axarr2 = plt.subplots(2, 1)
        axarr2[0].plot(ecg_h)
        axarr2[0].set_title('QRS on Filtered Signal')
        axarr2[0].scatter(qrs_i_raw, qrs_amp_raw, facecolors='none', edgecolors='m')
        axarr2[0].plot(locs, NOISL_buf1, '--k', LineWidth=2)
        axarr2[0].plot(locs, SIGL_buf1, '-.r', LineWidth=2)
        axarr2[0].plot(locs, THRS_buf1, '-.g', LineWidth=2)
        axarr2[1].plot(ecg_m)
        axarr2[1].set_title('QRS on MWI signal')
        axarr2[1].scatter(qrs_i, qrs_c, facecolors='none', edgecolors='m')
        axarr2[1].plot(locs, NOISL_buf, '--k', LineWidth=2)
        axarr2[1].plot(locs, SIGL_buf, '-.r', LineWidth=2)
        axarr2[1].plot(locs, THRS_buf, '-.g', LineWidth=2)
    #   axarr2[2].plot(ecg-np.mean(ecg))
    #   axarr2[2].set_title('Pulse train of the found QRS on ECG signal')
    # line(repmat(qrs_i_raw,[2 1]),repmat([min(ecg-mean(ecg))/2; max(ecg-mean(ecg))/2],
    # size(qrs_i_raw)),'LineWidth',2.5,'LineStyle','-.','Color','r');
    # linkaxes(az,'x');
    # zoom on;

    return qrs_amp_raw, qrs_i_raw, delay
