"""
Author: Meghan McConnell
"""
# import sys
import os
from os import listdir
from os.path import isfile, join, splitext
import numpy as np
import tkinter.constants as TKc
import scipy.io as sio
import h5py
import matplotlib.patches as patch
import tables as tb
import wfdb
from tkinter import Frame, Tk, Scale, Menu, END, Button, StringVar, filedialog, font, Toplevel, Listbox, Scrollbar, \
    Label, ANCHOR, FALSE, BOTTOM, TOP, BOTH, messagebox, RIGHT, Entry, PhotoImage
from tkinter.ttk import Style, OptionMenu, Treeview
from tkinter.ttk import Button as Button2
from tkinter.ttk import Label as Label2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from utils import MHTD, pan_tompkin, ECG_processing, Exporter, H5_Selector
from utils.HRV_Functions import *
from utils.Style_Functions import headerStyles

global enable, tt, edit_flag, loaded_ann, invert_flag, warnings_on, enable2, plot_pred, plot_ann, R_t, \
    h, pref, lower_RHS, button_help_on, delete_flag, TOTAL_FRAME, windows_compile, linux_compile, cnt, \
    R_amp, dat, True_R_t, True_R_amp, xminn, labelled_flag, Preferences, disp_length, pred_mode, Fs, ECG_pref_on, \
    graphCanvas, fig, t, fig2, RR_interval_Canvas, Slider, edit_btn, screen_height, screen_width, upper, plot_fig, \
    plot_fig2, rhs_ecg_frame, test_txt, Outerframe, F, F2, big_frame, btn, trybtn, resetbtn, okbtn, Meth, entry, \
    fs_frame_2, fs_frame, x, draw_figure, graphCanvas2, plt_options, pltmenu, load_dat, tcol, plot_num, \
    cid, h5window, h5window2, contact, helper, plot_wind, mets, xmaxx, font_, fontsize_


def close_fs(event):
    fs_frame.withdraw()


# PSEUDO CODE LIBRARY
# IMPORT LIBRARIES/TOOLBOXES HERE AS NECESSARY


def my_filter(raw_data, hh):
    filtered_data = np.convolve(raw_data, hh)
    filtered_data = filtered_data[int(len(hh) / 2):len(filtered_data) - (int(len(hh) / 2) + 1)]

    # REPLACE THIS SECTION WITH YOUR OWN FILTER or hh-coefficients
    # OR USE THOSE PROVIDED IN MHTD

    return (filtered_data)


def my_preprocessing_stage(filtered_data, var1=1, var2=1):
    pre_processed_data = filtered_data * var1 / var2

    # REPLACE THIS SECTION WITH YOUR OWN PREPROCESSING SEGEMENT OR REMOVE
    # For example, MHTD uses the HillTransform as a preprocessing tool
    return (pre_processed_data)


def my_QRS_detection(pre_processed_data, threshold=0):
    if threshold == 0:
        threshold = 10  # generate own threshold here or pass one to algorithm

    if (pre_processed_data > threshold):
        R_peak_predictions = 1
    else:
        R_peak_predictions = 0
    # Use this section to create or import a threshold and detect R_peaks based on decision rules
    # For example, MHTD creates a variable threshold and uses the peak search algorithm here

    return (R_peak_predictions)


def my_post_processing_tool(R_peak_predicitons, raw_data, rule1=0, rule2=1):
    corrected_R_peak_preds = R_peak_predicitons

    if (R_peak_predicitons != rule1):
        corrected_R_peak_preds = 0
    elif (R_peak_predicitons == rule2):
        corrected_R_peak_preds = 0

    pk_amplitude = raw_data[corrected_R_peak_preds]
    # Use this section to update R-peak predictions if required or remove if not required

    return (corrected_R_peak_preds, pk_amplitude)


# IMPORT FILTER COEFFICIENTS HERE
hh = [1, 2, 3, 4]


# The following function is a compliation function which calls all of the segements one at a time to produce a set of time stamp and amplitude value R-peak predictions

def my_function(raw_data):
    filtered_data = my_filter(raw_data, hh)
    pre_processed_data = my_preprocessing_stage(filtered_data)
    R_peak_predictions = my_QRS_detection(pre_processed_data)
    corrected_R_peak_preds, pk_amplitude = my_post_processing_tool(R_peak_predictions, raw_data)

    return (corrected_R_peak_preds, pk_amplitude)


# ========================== FUNCTIONS =========================#

# RECURRENCE PLOT
def RQA_plot(RRI, m=10, L=1, decim=2, Fig=0, fig_return=0):
    lenx = np.size(RRI)
    RRI = np.reshape(RRI, [lenx, ])
    N = lenx - ((m - 1) * L)  # N = number of points in recurrence plot
    r = np.sqrt(m) * np.std(
        RRI)  # r = fixed radius. Used as comparison point for Euclidian distance between two vectors
    # i.e. if ||X_i - X_j || < r then Vec(i,j) = 1
    X = np.zeros((N, m))  # X = multidimensional process of the time series as a trajectory in m-dim space

    # Generate vector X using X_i =(x(i),x(i+L),...,x(i+(m-1)L))
    for i in range(N):
        for j in range(m):
            X[i, j] = RRI[i + (j - 1) * L]

    Matrix = np.zeros((N, N))  # Vec = recurrence plot vector

    # Determine recurrence matrix (i.e. if 'closeness' is < given radius)
    for i in range(N):
        dist = np.sqrt(np.sum(np.power((X[i, :] - X), 2), axis=1))
        Matrix[i, :] = dist < r

    if fig_return == 1:
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

    else:
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

        # Analyse Verticals of RP
        ##
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


def RQA_plot_fig(RRI, m=10, L=1, decim=2):
    lenx = np.size(RRI)
    RRI = np.reshape(RRI, [lenx, ])
    N = lenx - ((m - 1) * L)  # N = number of points in recurrence plot
    r = np.sqrt(m) * np.std(
        RRI)  # r = fixed radius. Used as comparison point for Euclidian distance between two vectors
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
def Poincare(RRI, decim=3, Fig=0, fig_return=0):
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

    A = sd2 * np.cos(np.pi / 4)
    B = sd1 * np.sin(np.pi / 4)

    ellipse = patch.Ellipse((c1, c2), sd2 * 2, sd1 * 2, 45, facecolor="none", edgecolor="b", linewidth=2, zorder=5)

    if fig_return == 1:
        poin_plt = Fig.add_subplot(111)
        poin_plt.clear()
        if (poin_plt.axes.axes.yaxis_inverted() == 1):
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

    sd1 = np.round(sd1 * 1e3, decim)
    sd2 = np.round(sd2 * 1e3, decim)

    return Fig if fig_return else (sd1, sd2, c1, c2)


def set_initial_values():
    """
    Setting all initial values
    """
    global enable, tt, edit_flag, loaded_ann, invert_flag, warnings_on, enable2, plot_pred, plot_ann, R_t, \
        h, pref, lower_RHS, button_help_on, delete_flag, TOTAL_FRAME, windows_compile, linux_compile, \
        labelled_flag, Preferences, disp_length, pred_mode, Fs, ECG_pref_on, True_R_t, True_R_amp, x, dat, R_amp, \
        plot_wind, mets, font_, fontsize_
    with open("Preferences.txt", 'r') as f:
        Preferences = f.read().split()
    disp_length = int(Preferences[2])
    pred_mode = int(Preferences[3])
    Fs = int(Preferences[4])
    ECG_pref_on = int(Preferences[21])
    edit_flag = 0
    tt = 27
    loaded_ann = 0
    invert_flag = 0
    labelled_flag = 0
    warnings_on = 1
    enable = 0
    enable2 = 0
    plot_pred = 1
    plot_ann = 0
    R_t = []
    h = None  # Help class not created yet
    pref = None
    lower_RHS = None
    button_help_on = 1
    delete_flag = 0
    TOTAL_FRAME = None
    windows_compile = 0
    linux_compile = 1
    True_R_t = None
    True_R_amp = None
    x = None
    dat = None
    R_amp = None
    mets = None
    plot_wind = None
    font_ = int(Preferences[1])
    fontsize_ = Preferences[0]


root = Tk()
root.style = Style()
root.style.theme_use("alt")
root.style.configure('C.TButton', background='light grey', padding=0, relief='raised', width=0, highlightthickness=0)
root.style.configure('B.TButton', background='white smoke', padding=0, relief='raised', width=0, highlightthickness=0)
set_initial_values()

# ~~~~~~~~~~~~~~ KEY PRESS FUNCTIONS ~~~~~~~~~~~~~~~~~~~#
cust_text, cust_subheader, cust_subheadernb, cust_header = headerStyles(font_, fontsize_)


def open_plot():
    global plot_wind
    if plot_wind is not None:
        plot_wind.destroy()
    plot_wind = Toplevel()
    plot_viewer(plot_wind)
    if windows_compile:
        plot_wind.bind('<Escape>', close_plot_viewer)
    if linux_compile:
        plot_wind.bind('<Control-Escape>', close_plot_viewer)


def Invert(event):
    global invert_flag
    invert_flag ^= 1
    if invert_flag & warnings_on:
        messagebox.showwarning("Warning", "You have selected the inverted peak option. \n\nWhen editing annotations, " +
                               "the program will now find local minima with the left mouse click and local maxima with " +
                               "the right mouse click. To reverse this selection, simply press the 'i' key.")
    elif warnings_on & invert_flag != 1:
        messagebox.showwarning("Warning", "The inverted peak option has been cancelled.")


def InvertDelete(event):
    global delete_flag
    delete_flag ^= 1
    if delete_flag & warnings_on:
        messagebox.showwarning("Warning",
                               "Inverted peak add/delete option selected.\n\nMouse left click now deletes closest peak and mouse wheel button adds peak (polarity determined by inversion option).")
    elif warnings_on & invert_flag != 1:
        messagebox.showwarning("Warning",
                               "Reverted to normal conditions. Mouse left click now adds peak and mouse wheel button removes peak (polarity determined by inversion option).")


def shut(event):
    if mets is not None:
        mets.destroy()
    if plot_wind is not None:
        plot_wind.destroy()
    root.withdraw()
    os._exit(1)


def shut2():
    if mets is not None:
        mets.destroy()
    if plot_wind is not None:
        plot_wind.destroy()
    root.withdraw()
    os._exit(1)


def close_pref(event):
    pref.withdraw()


def close_fs2(event):
    fs_frame_2.withdraw()


def close_h5win(event):
    h5window.withdraw()


def close_h5win2(event):
    h5window2.withdraw()


def close_mets(event):
    mets.withdraw()


def close_plot_viewer(event):
    plot_wind.withdraw()


def onNoFSdata():
    global fs_frame
    fs_frame = Toplevel()
    Sampling_rate(fs_frame)

    if windows_compile:
        fs_frame.bind('<Escape>', close_fs)
    if linux_compile:
        fs_frame.bind('<Control-Escape>', close_fs)

    # ~~~~~~~~~~~~~~ MOUSE PRESS FUNCTIONS ~~~~~~~~~~~~~~~~~~~#


def onclick(event):
    global Fs
    global cnt
    global invert_flag
    global R_t
    global R_amp
    global dat
    global True_R_t
    global True_R_amp
    global plot_ann
    global plot_pred
    # print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
    #      (event.button, event.x, event.y, event.xdata, event.ydata))

    bound1 = int(0.05 / (1 / Fs))  # Gets approximately +- 50ms le-way with button click

    if ((event.button == 1) and (delete_flag == 0)) or ((event.button == 2) and (delete_flag == 1)):
        if (plot_ann == 1 and plot_pred == 0):  # Checks to see if editing loaded annotations or predicted annotations
            if (loaded_ann == 0 & warnings_on):
                messagebox.showwarning("Warning",
                                       "Cannot edit 'imported annotations' \n\nPlease note: Switch set the RRI plot to RR_Predicitions"
                                       + " or load in previously determined annotations.")
            else:
                xx = int(event.xdata * Fs)
                ll = np.size(True_R_t)

                if (invert_flag == 0):
                    R_amp_temp = np.max(dat[xx - bound1:xx + bound1])
                    pl = np.argmax(dat[xx - bound1:xx + bound1])
                else:
                    R_amp_temp = np.min(dat[xx - bound1:xx + bound1])
                    pl = np.argmin(dat[xx - bound1:xx + bound1])

                R_t_temp = xx - (bound1 + 1) + pl

                pl2 = np.argmin(np.abs(True_R_t - R_t_temp))

                if (R_t_temp < True_R_t[pl2]):
                    a = True_R_t[0:pl2]
                    b = True_R_t[pl2:ll]

                    True_R_t = np.append(a, R_t_temp)
                    True_R_t = np.append(True_R_t, b)

                    c = True_R_amp[0:pl2]
                    d = True_R_amp[pl2:ll]

                    True_R_amp = np.append(c, R_amp_temp)
                    True_R_amp = np.append(True_R_amp, d)
                else:
                    a = True_R_t[0:pl2 + 1]
                    b = True_R_t[pl2 + 1:ll]

                    True_R_t = np.append(a, R_t_temp)
                    True_R_t = np.append(True_R_t, b)

                    c = True_R_amp[0:pl2 + 1]
                    d = True_R_amp[pl2 + 1:ll]

                    True_R_amp = np.append(c, R_amp_temp)
                    True_R_amp = np.append(True_R_amp, d)

                draw1()

        elif (plot_pred == 1 and plot_ann == 0):

            if (len(R_t) == 0 & warnings_on):
                messagebox.showwarning("Warning",
                                       "Cannot edit empty annotations \n\nPlease generate annotations using the prediction functionality.")
            else:
                xx = int(event.xdata * Fs)
                ll = np.size(R_t)

                if (invert_flag == 0):
                    R_amp_temp = np.max(dat[xx - bound1:xx + bound1])
                    pl = np.argmax(dat[xx - bound1:xx + bound1])
                else:
                    R_amp_temp = np.min(dat[xx - bound1:xx + bound1])
                    pl = np.argmin(dat[xx - bound1:xx + bound1])

                R_t_temp = xx - (bound1 + 1) + pl

                pl2 = np.argmin(np.abs(R_t - R_t_temp))

                if (R_t_temp < R_t[pl2]):
                    a = R_t[0:pl2]
                    b = R_t[pl2:ll]

                    R_t = np.append(a, R_t_temp)
                    R_t = np.append(R_t, b)

                    c = R_amp[0:pl2]
                    d = R_amp[pl2:ll]

                    R_amp = np.append(c, R_amp_temp)
                    R_amp = np.append(R_amp, d)
                else:
                    a = R_t[0:pl2 + 1]
                    b = R_t[pl2 + 1:ll]

                    R_t = np.append(a, R_t_temp)
                    R_t = np.append(R_t, b)

                    c = R_amp[0:pl2 + 1]
                    d = R_amp[pl2 + 1:ll]

                    R_amp = np.append(c, R_amp_temp)
                    R_amp = np.append(R_amp, d)

                draw1()

        else:
            if (warnings_on):
                messagebox.showwarning("Warning",
                                       "Edit mode not activated. \n\nCannot edit while both RRI plots are active, please select either the RR_Predictions or RR_Annotations and re-activate edit mode.")
            edit_toggle()

    elif ((event.button == 2) and (delete_flag == 0)) or ((event.button == 1) and (delete_flag == 1)):

        if (plot_ann == 1 and plot_pred == 0):  # Checks to see if editing loaded annotations or predicted annotations
            if (loaded_ann == 0 & warnings_on):
                messagebox.showwarning("Warning",
                                       "Cannot edit 'imported annotations' \n\nPlease note: Switch set the RRI plot to RR_Predicitions"
                                       + " or load in previously determined annotations.")
            elif loaded_ann:
                if (invert_flag == 0):
                    pl = np.argmin(np.abs(True_R_t - (event.xdata * Fs)))
                else:
                    pl = np.argmax(np.abs(True_R_t - (event.xdata * Fs)))

                leng = np.size(True_R_t)

                a = True_R_t[0:pl]
                b = True_R_t[pl + 1:leng]

                True_R_t = np.append(a, b)

                c = True_R_amp[0:pl]
                d = True_R_amp[pl + 1:leng]

                True_R_amp = np.append(c, d)

                draw1()

        elif (plot_pred == 1 and plot_ann == 0):

            if (len(R_t) == 0 & warnings_on):
                messagebox.showwarning("Warning",
                                       "Cannot edit empty annotations \n\nPlease generate annotations using the prediction functionality.")
            elif (len(R_t) > 0):

                if (invert_flag == 0):
                    pl = np.argmin(np.abs(R_t - (event.xdata * Fs)))
                else:
                    pl = np.argmax(np.abs(R_t - (event.xdata * Fs)))

                leng = np.size(R_t)

                a = R_t[0:pl]
                b = R_t[pl + 1:leng]

                R_t = np.append(a, b)

                c = R_amp[0:pl]
                d = R_amp[pl + 1:leng]

                R_amp = np.append(c, d)

                draw1()
        else:
            if (warnings_on):
                messagebox.showwarning("Warning",
                                       "Edit mode not activated. \n\nCannot edit while both RRI plots are active, please select either the RR_Predictions or RR_Annotations and re-activate edit mode.")
            edit_toggle()

    elif event.button == 3:
        if (plot_ann == 1 and plot_pred == 0):  # Checks to see if editing loaded annotations or predicted annotations
            if (loaded_ann == 0 & warnings_on):
                messagebox.showwarning("Warning",
                                       "Cannot edit 'imported annotations' \n\nPlease note: Switch set the RRI plot to RR_Predicitions"
                                       + " or load in previously determined annotations.")
            elif loaded_ann:  # Checks to see if editing loaded annotations or predicted annotations
                xx = int(event.xdata * Fs)
                ll = np.size(True_R_t)

                if (invert_flag == 0):
                    R_amp_temp = np.min(dat[xx - bound1:xx + bound1])
                    pl = np.argmin(dat[xx - bound1:xx + bound1])

                else:
                    R_amp_temp = np.max(dat[xx - bound1:xx + bound1])
                    pl = np.argmax(dat[xx - bound1:xx + bound1])

                R_t_temp = xx - (bound1 + 1) + pl

                pl2 = np.argmin(np.abs(True_R_t - R_t_temp))

                if (R_t_temp < True_R_t[pl2]):
                    a = True_R_t[0:pl2]
                    b = True_R_t[pl2:ll]

                    True_R_t = np.append(a, R_t_temp)
                    True_R_t = np.append(True_R_t, b)

                    c = True_R_amp[0:pl2]
                    d = True_R_amp[pl2:ll]

                    True_R_amp = np.append(c, R_amp_temp)
                    True_R_amp = np.append(True_R_amp, d)
                else:
                    a = True_R_t[0:pl2 + 1]
                    b = True_R_t[pl2 + 1:ll]

                    True_R_t = np.append(a, R_t_temp)
                    True_R_t = np.append(True_R_t, b)

                    c = True_R_amp[0:pl2 + 1]
                    d = True_R_amp[pl2 + 1:ll]

                    True_R_amp = np.append(c, R_amp_temp)
                    True_R_amp = np.append(True_R_amp, d)

                draw1()


        elif (plot_pred == 1 and plot_ann == 0):

            if (len(R_t) == 0 & warnings_on):
                messagebox.showwarning("Warning",
                                       "Cannot edit empty annotations \n\nPlease generate annotations using the prediction functionality.")
            elif (len(R_t) > 0):
                xx = int(event.xdata * Fs)
                ll = np.size(R_t)

                if (invert_flag == 0):
                    R_amp_temp = np.min(dat[xx - bound1:xx + bound1])
                    pl = np.argmin(dat[xx - bound1:xx + bound1])

                else:
                    R_amp_temp = np.max(dat[xx - bound1:xx + bound1])
                    pl = np.argmax(dat[xx - bound1:xx + bound1])

                R_t_temp = xx - (bound1 + 1) + pl

                pl2 = np.argmin(np.abs(R_t - R_t_temp))

                if (R_t_temp < R_t[pl2]):
                    a = R_t[0:pl2]
                    b = R_t[pl2:ll]

                    R_t = np.append(a, R_t_temp)
                    R_t = np.append(R_t, b)

                    c = R_amp[0:pl2]
                    d = R_amp[pl2:ll]

                    R_amp = np.append(c, R_amp_temp)
                    R_amp = np.append(R_amp, d)
                else:
                    a = R_t[0:pl2 + 1]
                    b = R_t[pl2 + 1:ll]

                    R_t = np.append(a, R_t_temp)
                    R_t = np.append(R_t, b)

                    c = R_amp[0:pl2 + 1]
                    d = R_amp[pl2 + 1:ll]

                    R_amp = np.append(c, R_amp_temp)
                    R_amp = np.append(R_amp, d)

                draw1()

        else:
            if (warnings_on):
                messagebox.showwarning("Warning",
                                       "Edit mode not activated. \n\nCannot edit while both RRI plots are active, please select either the RR_Predictions or RR_Annotations and re-activate edit mode.")
            edit_toggle()


def onclick2(event):
    global xminn
    global Slider
    global disp_length
    global dat
    if event.button == 1:
        ss = int(event.xdata)
        xminn = ((ss - 0.5 * disp_length) * Fs)  # Xminn is in samples not seconds
        if xminn < 0:
            xminn = 0
            Slider.set(0)
        elif xminn > (len(dat) - disp_length * Fs):
            xminn = len(dat) - disp_length * Fs
            Slider.set(100)
        else:
            Slider.set(ss / (len(dat) / Fs) * 100)


def edit_toggle():
    global cid
    global fig
    global edit_btn
    global root
    global edit_flag

    #    if edit_btn.config('relief')[-1] == 'sunken':
    if edit_flag == 1:
        #        edit_btn.config(relief = "raised")
        edit_flag ^= 1
        root.style.configure('B.TButton', background='light grey', padding=0, relief='raised', width=0)
        fig.canvas.mpl_disconnect(cid)

    else:
        #        edit_btn.config(relief="sunken")
        edit_flag ^= 1
        root.style.configure('B.TButton', background='light grey', padding=0, relief='sunken', width=0)
        cid = fig.canvas.mpl_connect('button_press_event', onclick)


# ~~~~~~~~~~~~~~ OTHER FUNCTIONS ~~~~~~~~~~~~~~~~~~~#
def replace_line(file_name, line_number, new_text):
    lines = open(file_name, 'r').readlines()
    lines[line_number] = new_text
    with open(file_name, "w") as file:
        file.writelines(lines)


# def toggle_RRI_plot():
#    global loaded_ann
#    global can_plot_ogann
#    global b9
#
#
#    if ((loaded_ann == 1) & (can_plot_ogann == 0)):
#        can_plot_ogann = 1
#        b9.config(relief="sunken")
#
#    else:
#        can_plot_ogann = 0
#
#        b9.config(relief="raised")
#
#    draw1()

def draw1():
    global x
    global xminn
    global xmaxx
    global Fs
    global dat
    global plot_fig
    global graphCanvas
    global fig
    global R_t
    global R_amp
    global loaded_ann
    global labelled_flag
    global True_R_t
    global True_R_amp

    xmaxx = int(xminn + disp_length * Fs)
    yminn = np.min(dat[int(xminn):xmaxx]) - 0.1
    ymaxx = np.max(dat[int(xminn):xmaxx]) + 0.1

    # Top Figure
    plot_fig.clear()
    plot_fig.plot(x / Fs, dat, color='k', linewidth=1)
    if labelled_flag & plot_pred:
        plot_fig.plot(R_t / Fs, R_amp, 'b*', linewidth=1, markersize=7)
    if loaded_ann & plot_ann:
        plot_fig.plot(True_R_t / Fs, True_R_amp, 'ro', linewidth=1, markersize=5, fillstyle='none')
    plot_fig.axis([xminn / Fs, xmaxx / Fs, yminn, ymaxx])  # ([xminn,xmaxx,yminn,ymaxx])
    plot_fig.set_xlabel('Time (sec)')
    plot_fig.set_ylabel('ECG Amplitude (mV)')
    fig.tight_layout()
    graphCanvas.draw()

    draw2()


def draw2():
    global R_t
    global R_amp
    global True_R_t
    global True_R_amp
    global dat
    global x
    global fig2
    global plot_fig2
    global RR_interval_Canvas
    global lab
    global labelled_flag
    global xminn
    global xmaxx
    global plot_ann
    global plot_pred

    if plot_ann ^ plot_pred:
        if plot_pred:
            plotx = R_t
        else:
            plotx = True_R_t

        plotx = np.reshape(plotx, (len(plotx), 1))

        x2_max = np.max(plotx) / Fs
        sRt = np.size(plotx)

        y_diff = (np.diff(plotx[:, 0]) / Fs) * 1000

        pl = xminn / Fs
        pl2 = xmaxx / Fs

        y_minn = np.min(y_diff) - 10
        y_maxx = np.max(y_diff) + 10

        plot_fig2.clear()
        x2 = plotx[1:sRt] / Fs
        x2 = np.reshape(x2, (len(x2),))

        if plot_pred:
            plot_fig2.plot(x2, y_diff, 'b*', label='Predicted beats')
        else:
            plot_fig2.plot(x2, y_diff, 'ro', label='Annotated beats', markersize=5, fillstyle='none')

        plot_fig2.plot([pl + 0.05, pl + 0.05], [y_minn, y_maxx], 'k')
        plot_fig2.plot([pl2, pl2], [y_minn, y_maxx], 'k')
        plot_fig2.axis([0, x2_max, y_minn, y_maxx])
        plot_fig2.set_xlabel('Time (sec)')
        plot_fig2.set_ylabel('R-R Interval (ms)')
        plot_fig2.fill_between(x2, y_minn, y_maxx, where=(x2) <= pl2, facecolor='gainsboro')
        plot_fig2.fill_between(x2, y_minn, y_maxx, where=(x2) <= pl, facecolor='white')
        plot_fig2.legend()
        fig2.tight_layout()
        RR_interval_Canvas.draw()

    elif plot_ann & plot_pred:
        plot_fig2.clear()
        y_minn = np.zeros(2)
        y_maxx = np.zeros(2)
        x2_maxx = np.zeros(2)

        for N in range(2):
            if N == 0:
                plotx = R_t
            else:
                plotx = True_R_t
            sRt = np.size(plotx)
            y_diff = (np.diff(plotx[:, 0]) / Fs) * 1000
            y_minn[N] = np.min(y_diff) - 10
            y_maxx[N] = np.max(y_diff) + 10
            x2_maxx[N] = np.max(plotx) / Fs

            if N == 0:
                plot_fig2.plot(plotx[1:sRt] / Fs, y_diff, 'b*', label='Predicted beats')
            else:
                plot_fig2.plot(plotx[1:sRt] / Fs, y_diff, 'ro', label='Annotated beats', markersize=5, fillstyle='none')

        plot_fig2.set_xlabel('Time (sec)')
        plot_fig2.set_ylabel('R-R Interval (ms)')
        if y_minn[0] < y_minn[1]:
            plot_min = y_minn[0]
        else:
            plot_min = y_minn[1]
        if y_maxx[0] > y_maxx[1]:
            plot_max = y_maxx[0]
        else:
            plot_max = y_maxx[1]
        if x2_maxx[0] < x2_maxx[1]:
            plot_max_x2 = x2_maxx[0]
        else:
            plot_max_x2 = x2_maxx[1]
        plot_fig2.plot([(xminn / Fs) + 0.5, (xminn / Fs) + 0.5], [y_minn, y_maxx], 'k')
        plot_fig2.plot([(xmaxx / Fs), (xmaxx / Fs)], [y_minn, y_maxx], 'k')
        plot_fig2.axis([0, plot_max_x2, plot_min, plot_max])
        plot_fig2.legend()
        fig2.tight_layout()
        RR_interval_Canvas.draw()


def draw3():
    global R_t
    global fig2
    global plot_fig2
    global RR_interval_Canvas
    global lab
    global labelled_flag
    global plot_ann
    global plot_pred

    plotx = np.reshape(R_t, (len(R_t), 1))
    x2_max = np.max(plotx) / Fs
    sRt = np.size(plotx)
    y_diff = (np.diff(plotx[:, 0]) / Fs) * 1000
    y_minn = np.min(y_diff) - 10
    y_maxx = np.max(y_diff) + 10
    plot_fig2.clear()
    x2 = plotx[1:sRt] / Fs
    x2 = np.reshape(x2, (len(x2),))
    plot_fig2.plot(x2, y_diff, 'b*', label='R-peaks')
    plot_fig2.axis([0, x2_max, y_minn, y_maxx])
    plot_fig2.set_xlabel('Time (sec)')
    plot_fig2.set_ylabel('R-R Interval (ms)')
    plot_fig2.legend()
    fig2.tight_layout()
    RR_interval_Canvas.draw()


def freq_plot(METHOD, DECIMALS, subplot_, m=1200, o=600, btval=10, omax=500, Ord=100):
    global draw_figure
    global R_t
    global graphCanvas2
    Rpeakss = np.reshape(R_t, (len(R_t),))
    Rpeak_input = Rpeakss / Fs
    fig_holder = Figure(dpi=150)
    f, P2 = Freq_Analysis(Rpeak_input, meth=METHOD, decim=DECIMALS, m=m, o=o, bt_val=btval, omega_max=omax, order=Ord,
                          figure=True)
    title_list = ['Welch\'s', 'Blackman-Tukey\'s', 'LombScargle\'s', 'Auto Regression']
    subplot_.clear()
    subplot_.xaxis.tick_bottom()
    subplot_.plot(f, P2, 'black', linewidth=0.75, label='PSD')  # ,
    subplot_.set_xlim(0, 0.5)
    subplot_.set_ylim(ymin=0)
    subplot_.set_title("PSD Estimation using " + title_list[METHOD - 1] + " method")
    subplot_.set_xlabel("Frequency (Hz)")
    subplot_.set_ylabel("Power Spectral Density (s$^{2}$/Hz)", labelpad=10)
    subplot_.fill_between(f, 0, P2, where=f <= 0.4, facecolor='red', label='HF')
    subplot_.fill_between(f, 0, P2, where=f <= 0.15, facecolor='blue', label='LF')
    subplot_.fill_between(f, 0, P2, where=f < 0.04, facecolor='green', label='VLF')
    subplot_.legend()
    draw_figure.tight_layout()
    graphCanvas2.draw()


def DFA_plot(subplot_, Min=4, Max=64, Inc=1, COP=12):
    global draw_figure
    global R_t
    global graphCanvas2
    Rpeakss = np.reshape(R_t, (len(R_t),))
    Rpeak_input = Rpeakss / Fs
    RRI = np.diff(Rpeak_input)
    (x_vals1, y_vals1, y_new1, x_vals2, y_vals2, y_new2, a1, a2) = DFA(RRI, min_box=Min, max_box=Max, inc=Inc,
                                                                       cop=COP, decim=3, figure=True)
    subplot_.clear()
    subplot_.plot(x_vals1, y_vals1, '*')
    subplot_.plot(x_vals1, y_new1, 'r--')
    txt = "\u03B1$_1$ = {:.3f}".format(a1)
    subplot_.text(0.8, np.min(y_vals1) + 0.1, txt)
    subplot_.plot(x_vals2, y_vals2, '*')
    subplot_.plot(x_vals2, y_new2, 'k--')
    txt = "\u03B1$_2$ = {:.3f}".format(a2)
    subplot_.text(1.5, np.mean(y_vals2) * 1.1, txt)
    subplot_.set_title('Detrended Fluctuation Analysis of the RRI series')
    subplot_.set_xlabel('log$_{10}$ n (beats)')
    subplot_.set_ylabel('log$_{10}$ F(n) (beats)')
    draw_figure.tight_layout()
    graphCanvas2.draw()


def Poincare_plot(subplot_):
    global draw_figure
    global R_t
    global graphCanvas2
    Rpeakss = np.reshape(R_t, (len(R_t),))
    Rpeak_input = Rpeakss / Fs
    RRI = np.diff(Rpeak_input)
    sd1, sd2, c1, c2 = Poincare(RRI, decim=3, Fig=0, fig_return=0)
    lenx = np.size(RRI)
    xp = RRI[0:lenx - 1]
    yp = RRI[1:lenx]
    A = sd2 * np.cos(np.pi / 4)
    B = sd1 * np.sin(np.pi / 4)
    ellipse = patch.Ellipse((c1, c2), sd2 * 2, sd1 * 2, 45, facecolor="none", edgecolor="b", linewidth=2, zorder=5)
    subplot_.clear()
    subplot_.plot(xp, yp, 'ko', markersize=3, zorder=0)  # ,'MarkerFaceColor', 'k', 'MarkerSize',4)
    subplot_.add_patch(ellipse)
    subplot_.set_title('Poincare Plot')
    subplot_.set_xlabel('RRI$_{n}$ (s)')
    subplot_.set_ylabel('RRI$_{n+1}$ (s)')
    subplot_.plot([c1, c1 + A], [c2, c2 + A], 'm', label="SD1", zorder=10)
    subplot_.plot([c1 - 4 * A, c1 + 4 * A], [c2 - 4 * A, c2 + 4 * A], 'b', dashes=[6, 2])
    subplot_.plot([c1, c1 - B], [c2, c2 + B], 'c', label="SD2", zorder=10)
    subplot_.plot([c1 + B * 4, c1 - B * 4], [c2 - 4 * B, c2 + 4 * B], 'b', dashes=[4, 2, 10, 2])
    subplot_.legend()
    draw_figure.tight_layout()
    graphCanvas2.draw()


def RQA_plott(subplot_, Mval=10, Lval=1):
    global draw_figure
    global R_t
    global graphCanvas2
    Rpeakss = np.reshape(R_t, (len(R_t),))
    Rpeak_input = Rpeakss / Fs
    RRI = np.diff(Rpeak_input)
    Matrix, N = RQA_plot_fig(RRI, m=Mval, L=Lval, decim=2)
    xplot = np.zeros((N, N))
    yplot = np.zeros((N, N))
    subplot_.clear()
    for i in range(0, len(Matrix)):
        yplot[:, i] = np.arange(1, len(Matrix) + 1) * Matrix[:, i]
        xplot[:, i] = np.ones(len(Matrix)) * (i + 1)
    subplot_.scatter(xplot, yplot, c='k', s=0.5)
    subplot_.set_title('Recurrence Plot')
    subplot_.set_xlabel('Heart beat (sample number)')
    subplot_.set_ylabel('Heart beat (sample number)')
    draw_figure.tight_layout()
    graphCanvas2.draw()


def Prediction_mode(mode_type, thr_ratio=1.25, SBL=5, MAG_LIM=0.10, ENG_LIM=0.05, MIN_L=0.3):
    global dat, R_t, R_amp, xminn, xmaxx, labelled_flag

    labelled_flag = 1
    if mode_type == 1:
        # ========================== Set Values =========================#
        fs = Fs
        fpass = 0.5
        fstop = 45
        # ====================== Conduct Predictions =======================#
        R_t = MHTD(dat, fs, fpass, fstop, thr_ratio=thr_ratio, sbl=SBL, mag_lim=MAG_LIM, eng_lim=ENG_LIM, min_L=MIN_L)

    elif mode_type == 2:
        # ========================== Set Values =========================#
        data_input = np.reshape(dat, (len(dat),))
        # ====================== Conduct Predictions =======================#
        R_amp, R_t, delay = pan_tompkin(data_input, Fs, 0)

    elif mode_type == 3:
        # ====================== Conduct Predictions =======================#
        R_t = ECG_processing(dat)

    elif mode_type == 4:
        # ====================== Conduct Predictions =======================#
        R_t, R_amp = my_function(dat)

    if 1 <= mode_type <= 4:
        if mode_type == 1 or mode_type == 3:
            siz = np.size(R_t)
            R_t = np.reshape(R_t, [siz, 1])
            R_amp = np.zeros(siz)
            for i in range(siz):
                R_amp[i] = dat[int(R_t[i])]
            if len(R_amp) == 1:
                R_amp = np.transpose(R_amp)

        draw1()
    else:
        messagebox.showwarning("Warning",
                               "Mode_Type selected is unavaliable' \n\nPlease select a valid mode.")


def Prediction_no_plot(ECGdata, mode_type, fs, thr_ratio=1.25, SBL=5, MAG_LIM=0.10, ENG_LIM=0.05, MIN_L=0.3):
    """

    :param fs:
    :param ECGdata:
    :param mode_type:
    :param thr_ratio:
    :param SBL:
    :param MAG_LIM:
    :param ENG_LIM:
    :param MIN_L:
    :return:
    """
    # ========================== Set Values =========================#
    fpass = 5
    fstop = 30
    # ====================== Conduct Predictions =======================#
    if mode_type == 1:
        predictions = MHTD(dat, fs, fpass, fstop, thr_ratio, SBL, MAG_LIM, ENG_LIM, MIN_L)
    elif mode_type == 2:
        data_input = np.reshape(ECGdata, (len(ECGdata),))
        _, predictions, _ = pan_tompkin(data_input, fs, 0)
    elif mode_type == 3:
        predictions = ECG_processing(ECGdata)
    else:
        return 0
    siz = np.size(predictions)
    predictions = np.reshape(predictions, [siz, ])
    return predictions


def savefigure():
    global draw_figure
    if linux_compile:
        path_for_save = filedialog.asksaveasfilename(title="Select file", filetypes=(
            ("eps", "*.eps"), ("png", "*.png"), ("svg", "*.svg"), ("all files", "*.*")))
    if windows_compile:
        path_for_save = filedialog.asksaveasfilename(title="Select file", defaultextension=".*", filetypes=(
            ("eps", "*.eps"), ("png", "*.png"), ("svg", "*.svg"), ("all files", "*.*")))

    fname, file_extension = os.path.splitext(path_for_save)

    if file_extension == '.png':
        draw_figure.savefig(path_for_save, format='png', dpi=300)
    elif file_extension == '.svg':
        draw_figure.savefig(path_for_save, format='svg', dpi=300)
    else:
        draw_figure.savefig(path_for_save, format='eps', dpi=300)


def savemetrics():
    global R_amp
    global R_t
    global True_R_amp
    global True_R_t
    global loaded_ann
    global labelled_flag
    global tt
    global Fs

    if loaded_ann == 1:
        TP, FP, FN = test2(R_t, True_R_t, tt)
        Se, PP, ACC, DER = acc2(TP, FP, FN)

    Rpeakss = np.reshape(R_t, (len(R_t),))

    file = open("Preferences.txt", 'r')
    Preferences = file.read().split()
    file.close()
    welchL = float(Preferences[5])
    welchO = float(Preferences[6])
    btval_input = int(Preferences[7])  # 10
    omax_input = int(Preferences[8])  # 500
    order = int(Preferences[9])  # 10

    # Time-domain Statistics
    SDNN, SDANN, MeanRR, RMSSD, pNN50 = Calculate_Features(Rpeakss, Fs)

    # Frequency-domain Statistics
    Rpeak_input = Rpeakss / Fs
    powVLF, powLF, powHF, perpowVLF, perpowLF, perpowHF, peak_freq_VLF, peak_freq_LF, peak_freq_HF, LF_HF_ratio = Freq_Analysis(
        Rpeak_input, meth=1, decim=3, m=welchL, o=welchO, bt_val=btval_input, omega_max=omax_input, order=order)
    powVLF2, powLF2, powHF2, perpowVLF2, perpowLF2, perpowHF2, peak_freq_VLF2, peak_freq_LF2, peak_freq_HF2, LF_HF_ratio2 = Freq_Analysis(
        Rpeak_input, meth=2, decim=3, m=welchL, o=welchO, bt_val=btval_input, omega_max=omax_input, order=order)
    powVLF3, powLF3, powHF3, perpowVLF3, perpowLF3, perpowHF3, peak_freq_VLF3, peak_freq_LF3, peak_freq_HF3, LF_HF_ratio3 = Freq_Analysis(
        Rpeak_input, meth=3, decim=3, m=welchL, o=welchO, bt_val=btval_input, omega_max=omax_input, order=order)
    powVLF4, powLF4, powHF4, perpowVLF4, perpowLF4, perpowHF4, peak_freq_VLF4, peak_freq_LF4, peak_freq_HF4, LF_HF_ratio4 = Freq_Analysis(
        Rpeak_input, meth=4, decim=3, m=welchL, o=welchO, bt_val=btval_input, omega_max=omax_input, order=order)

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
            if (labelled_flag == 1) & (loaded_ann == 1):
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


def setupperbutton(opts):
    global root
    global rhs_ecg_frame
    global plt_options
    global pltmenu
    global lower_RHS
    if lower_RHS != None:
        lower_RHS.destroy()

    lower_RHS = Frame(rhs_ecg_frame, bg='white smoke')
    lower_RHS.pack(side='bottom', fill=BOTH, expand=False)

    root.plot_num = StringVar(lower_RHS)
    plt_options = opts
    pltmenu = OptionMenu(lower_RHS, root.plot_num, plt_options[0], *plt_options)
    pltmenu.config(width=4)
    pltmenu.pack(side='bottom')
    root.plot_num.trace('w', change_dropdown1)
    Label2(lower_RHS, text='Input', style='Text2.TLabel').pack(side='bottom', anchor='w')


def change_dropdown1(*args):
    global enable2
    global plt_options
    global load_dat
    global dat
    global x
    global xminn
    global tcol
    arg = root.plot_num.get()

    enable2 ^= 1
    if enable2:
        for counter in range(len(plt_options)):
            if arg == plt_options[counter]:
                if tcol == 1:
                    counter = counter + 1
                dat = load_dat[:, counter]
                dat = np.reshape(dat, [len(dat), 1])
                x = np.arange(len(dat))
                #                xminn = 0
                Prediction_mode(1)
                draw1()
                break


def Link_hover_popup_tips(widget, text):
    if button_help_on:
        toolTip = Pop_up(widget)

        def enter(event):
            toolTip.showpopup(text)

        def leave(event):
            toolTip.hidepopup()

        widget.bind('<Enter>', enter)
        widget.bind('<Leave>', leave)


def Link_hover_popup_tips2(widget, text):
    if button_help_on:
        toolTip = Pop_up2(widget)

        def enter(event):
            toolTip.showpopup(text)

        def leave(event):
            toolTip.hidepopup()

        widget.bind('<Enter>', enter)
        widget.bind('<Leave>', leave)


# FILTER PARAMETERS
def multRUN(path, savefilename, FS, ext):
    # GET VARIABLES SET IN PROGRAM
    global Preferences

    messagebox.showinfo("Runtime Info", "Multiple ECG processing is underway. Please do not touch RR-APET Program.")

    FreqMeth = 1  # Set 1 for Welch, 2 for Blackman-Tukey, 3 for Lombscargle, or 4 for AutoRegression
    Dec = 2  # Number of decimal places for metrics
    RpeakMeth = int(Preferences[
                        3])  # Set 1 for MHTD, 2 for Pan-Tompkins, 3 for original HTD, 4 for K-means, or 5 for your own method
    # MHTD settings
    ML = float(Preferences[
                   18])  # Magnitude limit - peaks within specified proxmity must meet this criteria to both remain in predictions
    EL = float(Preferences[
                   19])  # Energy limit - peaks within specified proxmity must meet this criteria to both remain in predictions
    Lmin = float(Preferences[20])  # Minimum length - time proximity (in seconds) for ML and EL
    Ratio_vthr = float(Preferences[16])  # Variable threshold ratio between max values and height of thr
    sbl = int(Preferences[17])  # Search back length - time (in seconds) for vthr
    #    chunk = 30          #Chunking factor (in seconds) for Hilbert transform
    savefilename2 = savefilename + ext
    saveroot = path + '/' + savefilename2

    file_names = [f for f in listdir(path) if isfile(join(path, f))]

    if ext == '.txt':
        dtyp = 1
    elif ext == '.h5':
        dtyp = 2
    elif ext == '.mat':
        dtyp = 3
    else:
        dtyp = 0

    if dtyp == 1:
        with open(saveroot, "w") as text_file:
            print("file\tSDNN\tSDANN\tMRRI\tRMSSD\tpNN50\tpowULF\tpowLF\tpowHF\t%powULF\t%powLF\t%powHF\tfULF\t" +
                  "fLF\tfHF\tLFHF\tREC\tDET\tLAM\tLmean\tLmax\tVmean\tVmax\tSD1\tSD2\talp1\talp2", file=text_file)
        for i in range(len(file_names)):
            doNotContinue = 0
            fn = file_names[i]
            patientfile = path + '/' + fn
            fname, ext = splitext(patientfile)

            if ext != '.txt':
                print(file_names[
                          i] + ' - cannot import this file type! Please keep consistent file types when using RR-APET for multiple ECG analysis')

            elif fn == savefilename2:
                print('Ignore Self')

            else:
                file = open(patientfile, 'r')
                if Preferences[24] == '0':
                    temp = file.read().split()
                elif Preferences[24] == '1':
                    temp = file.read().split(":")
                else:
                    temp = file.read().split(";")
                var1 = len(temp)
                ECG = np.zeros(np.size(temp))
                try:
                    for i in range(var1):
                        ECG[i] = float(temp[i].rstrip('\n'))
                except:
                    doNotContinue = 1

                if doNotContinue != 1:
                    file.seek(0)
                    temp2 = file.readlines()
                    var2 = len(temp2)
                    columns = var1 / var2
                    ECG = np.reshape(ECG, [len(temp2), int(columns)])
                    if columns > var2:
                        ECG = np.transpose(ECG)

                    if columns > 1:
                        if (np.diff(ECG[:, 0]) > 0).all():
                            dat1 = ECG[:, 1]
                            dat1 = np.reshape(dat1, [len(dat1), 1])
                        else:
                            dat1 = load_dat[:, 0]
                            dat1 = np.reshape(dat1, [len(dat1), 1])
                    else:
                        dat1 = load_dat[:, 0]
                        dat1 = np.reshape(dat1, [len(dat1), 1])

                    if len(ECG) > 0:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            R_peaks = Prediction_no_plot(ECGdata=dat1, mode_type=RpeakMeth, fs=FS, thr_ratio=Ratio_vthr,
                                                         SBL=sbl, MAG_LIM=ML, ENG_LIM=EL, MIN_L=Lmin)
                            # ===================Time-domain Statistics====================#
                            SDNN, SDANN, MeanRR, RMSSD, pNN50 = Calculate_Features(R_peaks, FS, decim=Dec)
                            # ===================Frequency-domain Statistics====================#
                            Rpeak_input = R_peaks / FS
                            powULF, powLF, powHF, perpowULF, perpowLF, perpowHF, peak_freq_ULF, peak_freq_LF, peak_freq_HF, LF_HF_ratio = Freq_Analysis(
                                Rpeak_input, meth=FreqMeth, decim=Dec)
                            # ===================Nonlinear statistics====================#
                            RRI = np.diff(Rpeak_input)
                            REC, DET, LAM, Lmean, Lmax, Vmean, Vmax = RQA_plot(RRI, decim=Dec)
                            SD1, SD2, c1, c2 = Poincare(RRI, decim=Dec)
                            alp1, alp2, F = DFA(RRI, decim=Dec)

                        with open(saveroot, "a") as text_file:
                            print(f"{fn}\t{SDNN}\t{SDANN}\t{MeanRR}\t{RMSSD}\t{pNN50}\t{powULF}\t{powLF}\t" +
                                  f"{powHF}\t{perpowULF}\t{perpowLF}\t{perpowHF}\t{peak_freq_HF}\t{peak_freq_LF}\t" +
                                  f"{peak_freq_HF}\t{LF_HF_ratio}\t{REC}\t{DET}\t{LAM}\t{Lmean}\t{Lmax}\t{Vmean}\t" +
                                  f"{Vmax}\t{SD1}\t{SD2}\t{alp1}\t{alp2}", file=text_file)


    elif dtyp == 2:
        with open(saveroot, "w") as text_file:
            print("file\tSDNN\tSDANN\tMRRI\tRMSSD\tpNN50\tpowULF\tpowLF\tpowHF\t%powULF\t%powLF\t%powHF\tfULF\t" +
                  "fLF\tfHF\tLFHF\tREC\tDET\tLAM\tLmean\tLmax\tVmean\tVmax\tSD1\tSD2\talp1\talp2", file=text_file)
        subdir = input("What is the name of the directory within the HDF5 file, where the ECG data exists? ")
        for i in range(len(file_names)):
            fn = file_names[i]
            patientfile = path + '/' + fn
            fname, ext = splitext(patientfile)

            if ext != '.h5':
                print(file_names[
                          i] + ' - cannot import this file type! Please keep consistent file types when using RR-APET for multiple ECG analysis')

            else:
                file = h5py.File(patientfile, 'r')
                pats = list(file.keys())

                for j in range(len(pats)):
                    f_name = pats[j] + '/' + subdir
                    ECG = file[f_name]
                    ECG = ECG[:]

                    r, c = np.shape(ECG)
                    if c > r:
                        ECG = np.transpose(ECG)

                    if c > 1:
                        if (np.diff(ECG[:, 0]) > 0).all():
                            dat1 = ECG[:, 1]
                            dat1 = np.reshape(dat1, [len(dat1), 1])
                        else:
                            dat1 = load_dat[:, 0]
                            dat1 = np.reshape(dat1, [len(dat1), 1])
                    else:
                        dat1 = load_dat[:, 0]
                        dat1 = np.reshape(dat1, [len(dat1), 1])

                    if len(ECG) > 0:
                        R_peaks = Prediction_no_plot(ECGdata=dat1, mode_type=RpeakMeth, fs=FS, thr_ratio=Ratio_vthr,
                                                     SBL=sbl, MAG_LIM=ML, ENG_LIM=EL, MIN_L=Lmin)

                        #                        R_peaks = MHTD(ECG, fs, fpass = pass_freq, fstop = stop_freq, MAG_LIM=ML, ENG_LIM = EL, MIN_L = Lmin, viewfilter = view, vthr= Ratio_vthr, SBL = sbl, chunking = chunk)    #Gives back in samples
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            # ===================Time-domain Statistics====================#
                            SDNN, SDANN, MeanRR, RMSSD, pNN50 = Calculate_Features(R_peaks, FS, decim=Dec)
                            # ===================Frequency-domain Statistics====================#
                            Rpeak_input = R_peaks / FS
                            powULF, powLF, powHF, perpowULF, perpowLF, perpowHF, peak_freq_ULF, peak_freq_LF, peak_freq_HF, LF_HF_ratio = Freq_Analysis(
                                Rpeak_input, meth=FreqMeth, decim=Dec)
                            # ===================Nonlinear statistics====================#
                            RRI = np.diff(Rpeak_input)
                            REC, DET, LAM, Lmean, Lmax, Vmean, Vmax = RQA_plot(RRI, decim=Dec)
                            SD1, SD2 = Poincare(RRI, decim=Dec)
                            alp1, alp2, F = DFA(RRI, decim=Dec)
                        with open(saveroot, "a") as text_file:
                            savename = fn + '->' + f_name
                            print(f"{savename}\t{SDNN}\t{SDANN}\t{MeanRR}\t{RMSSD}\t{pNN50}\t{powULF}\t{powLF}\t" +
                                  f"{powHF}\t{perpowULF}\t{perpowLF}\t{perpowHF}\t{peak_freq_HF}\t{peak_freq_LF}\t" +
                                  f"{peak_freq_HF}\t{LF_HF_ratio}\t{REC}\t{DET}\t{LAM}\t{Lmean}\t{Lmax}\t{Vmean}\t" +
                                  f"{Vmax}\t{SD1}\t{SD2}\t{alp1}\t{alp2}", file=text_file)


    elif dtyp == 3:
        with open(saveroot, "w") as text_file:
            print("file\tSDNN\tSDANN\tMRRI\tRMSSD\tpNN50\tpowULF\tpowLF\tpowHF\t%powULF\t%powLF\t%powHF\tfULF\t" +
                  "fLF\tfHF\tLFHF\tREC\tDET\tLAM\tLmean\tLmax\tVmean\tVmax\tSD1\tSD2\talp1\talp2", file=text_file)

        data_call = input("Please enter complete directory path to ECG data with MAT files: ")

        for i in range(len(file_names)):
            fn = file_names[i]
            patientfile = path + '/' + file_names[i]
            fname, ext = splitext(patientfile)

            if ext != '.mat':
                print(file_names[
                          i] + ' - cannot import this file type! Please keep consistent file types when using RR-APET for multiple ECG analysis')
            else:
                with h5py.File(patientfile, 'r') as hrv:
                    ECG = hrv[data_call][:]

                r, c = np.shape(ECG)
                if c > r:
                    ECG = np.transpose(ECG)

                if c > 1:
                    if (np.diff(ECG[:, 0]) > 0).all():
                        dat1 = ECG[:, 1]
                        dat1 = np.reshape(dat1, [len(dat1), 1])
                    else:
                        dat1 = load_dat[:, 0]
                        dat1 = np.reshape(dat1, [len(dat1), 1])
                else:
                    dat1 = load_dat[:, 0]
                    dat1 = np.reshape(dat1, [len(dat1), 1])

                if len(ECG) > 0:
                    R_peaks = Prediction_no_plot(ECGdata=dat1, mode_type=RpeakMeth, fs=FS, thr_ratio=Ratio_vthr,
                                                 SBL=sbl, MAG_LIM=ML, ENG_LIM=EL, MIN_L=Lmin)

                    #                    R_peaks = MHTD(ECG, fs, fpass = pass_freq, fstop = stop_freq, MAG_LIM=ML, ENG_LIM = EL, MIN_L = Lmin, viewfilter = view, vthr= Ratio_vthr, SBL = sbl, chunking = chunk)    #Gives back in samples
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        # ===================Time-domain Statistics====================#
                        SDNN, SDANN, MeanRR, RMSSD, pNN50 = Calculate_Features(R_peaks, FS, decim=Dec)
                        # ===================Frequency-domain Statistics====================#
                        Rpeak_input = R_peaks / FS
                        powULF, powLF, powHF, perpowULF, perpowLF, perpowHF, peak_freq_ULF, peak_freq_LF, peak_freq_HF, LF_HF_ratio = Freq_Analysis(
                            Rpeak_input, meth=FreqMeth, decim=Dec)
                        # ===================Nonlinear statistics====================#
                        RRI = np.diff(Rpeak_input)
                        REC, DET, LAM, Lmean, Lmax, Vmean, Vmax = RQA_plot(RRI, decim=Dec)
                        SD1, SD2 = Poincare(RRI, decim=Dec)
                        alp1, alp2, F = DFA(RRI, decim=Dec)
                    with open(saveroot, "a") as text_file:
                        savename = fn + '->' + fname
                        print(f"{savename}\t{SDNN}\t{SDANN}\t{MeanRR}\t{RMSSD}\t{pNN50}\t{powULF}\t{powLF}\t" +
                              f"{powHF}\t{perpowULF}\t{perpowLF}\t{perpowHF}\t{peak_freq_HF}\t{peak_freq_LF}\t" +
                              f"{peak_freq_HF}\t{LF_HF_ratio}\t{REC}\t{DET}\t{LAM}\t{Lmean}\t{Lmax}\t{Vmean}\t" +
                              f"{Vmax}\t{SD1}\t{SD2}\t{alp1}\t{alp2}", file=text_file)

    else:
        print('Cannot import this file type! Use *.txt, *.mat, or *.h5')
    print("Complete.")


def multRUN2(path, savefilename, FS, ext):
    # GET VARIABLES SET IN PROGRAM
    messagebox.showinfo("Runtime Info", "Multiple ECG processing is underway. Please do not touch RR-APET Program.")

    FILE = open("Preferences.txt", 'r')
    Preferences = FILE.read().split()
    FILE.close()

    FreqMeth = 1  # Set 1 for Welch, 2 for Blackman-Tukey, 3 for Lombscargle, or 4 for AutoRegression
    Dec = 2  # Number of decimal places for metrics

    #    chunk = 30          #Chunking factor (in seconds) for Hilbert transform
    savefilename2 = savefilename + ext
    saveroot = path + '/' + savefilename2

    file_names = [f for f in listdir(path) if isfile(join(path, f))]

    if ext == '.txt':
        dtyp = 1
    elif ext == '.h5':
        dtyp = 2
    elif ext == '.mat':
        dtyp = 3
    else:
        dtyp = 0

    if (dtyp == 1):
        with open(saveroot, "w") as text_file:
            print("file\tSDNN\tSDANN\tMRRI\tRMSSD\tpNN50\tpowULF\tpowLF\tpowHF\t%powULF\t%powLF\t%powHF\tfULF\t" +
                  "fLF\tfHF\tLFHF\tREC\tDET\tLAM\tLmean\tLmax\tVmean\tVmax\tSD1\tSD2\talp1\talp2", file=text_file)
        for i in range(len(file_names)):
            fn = file_names[i]
            patientfile = path + '/' + fn
            fname, ext = splitext(patientfile)

            if ext != '.txt':
                print(file_names[
                          i] + ' - cannot import this file type! Please keep consistent file types when using RR-APET for multiple ECG analysis')

            elif fn == savefilename2:
                print('Ignore Self')

            else:
                file = open(patientfile, 'r')
                if (Preferences[24] == '0'):
                    temp = file.read().split()
                elif (Preferences[24] == '1'):
                    temp = file.read().split(":")
                else:
                    temp = file.read().split(";")
                var1 = len(temp)
                og_ann = np.zeros(np.size(temp))
                for i in range(len(temp)):
                    og_ann[i] = float(temp[i].rstrip('\n'))

                file.seek(0)
                temp2 = file.readlines()
                var2 = len(temp2)
                columns = var1 / var2
                og_ann = np.reshape(og_ann, [len(temp2), int(columns)])
                if (columns > var2):
                    og_ann = np.transpose(og_ann)

                file.close()
                R_peaks = og_ann[:, 0]
                if (Preferences[23] == '1'):
                    R_peaks = R_peaks / 1e3
                if np.mean(
                        np.diff(
                            R_peaks)) < 6:  # Average time interval between heart beats wouldn't be less than 10bpm, so a gap of over 6 seconds on average or greater means loaded in as samples
                    R_peaks = R_peaks * Fs  # Measured in time but need samples
                R_peaks = R_peaks[R_peaks != 0]
                R_peaks = np.reshape(R_peaks, [len(R_peaks), ])
                if (np.diff(R_peaks[:]) > 0).all() == False:
                    # They aren't all greater than the previous - therefore RRI series not time-stamps
                    tmp = np.zeros(np.size(R_peaks))
                    tmp[0] = True_R_t[0]

                    for i in range(1, np.size(R_peaks)):
                        tmp[i] = tmp[i - 1] + R_peaks[i]

                R_peaks = np.reshape(R_peaks, [len(R_peaks), 1])

                if (len(R_peaks) > 0):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)

                        # ===================Time-domain Statistics====================#
                        SDNN, SDANN, MeanRR, RMSSD, pNN50 = Calculate_Features(R_peaks, FS, decim=Dec)
                        # ===================Frequency-domain Statistics====================#
                        Rpeak_input = R_peaks / FS
                        powULF, powLF, powHF, perpowULF, perpowLF, perpowHF, peak_freq_ULF, peak_freq_LF, peak_freq_HF, LF_HF_ratio = Freq_Analysis(
                            Rpeak_input, meth=FreqMeth, decim=Dec)
                        # ===================Nonlinear statistics====================#
                        RRI = np.diff(Rpeak_input)
                        REC, DET, LAM, Lmean, Lmax, Vmean, Vmax = RQA_plot(RRI, decim=Dec)
                        SD1, SD2, c1, c2 = Poincare(RRI, decim=Dec)
                        alp1, alp2, F = DFA(RRI, decim=Dec)

                    with open(saveroot, "a") as text_file:
                        print(f"{fn}\t{SDNN}\t{SDANN}\t{MeanRR}\t{RMSSD}\t{pNN50}\t{powULF}\t{powLF}\t" +
                              f"{powHF}\t{perpowULF}\t{perpowLF}\t{perpowHF}\t{peak_freq_HF}\t{peak_freq_LF}\t" +
                              f"{peak_freq_HF}\t{LF_HF_ratio}\t{REC}\t{DET}\t{LAM}\t{Lmean}\t{Lmax}\t{Vmean}\t" +
                              f"{Vmax}\t{SD1}\t{SD2}\t{alp1}\t{alp2}", file=text_file)


    elif (dtyp == 2):
        with open(saveroot, "w") as text_file:
            print("file\tSDNN\tSDANN\tMRRI\tRMSSD\tpNN50\tpowULF\tpowLF\tpowHF\t%powULF\t%powLF\t%powHF\tfULF\t" +
                  "fLF\tfHF\tLFHF\tREC\tDET\tLAM\tLmean\tLmax\tVmean\tVmax\tSD1\tSD2\talp1\talp2", file=text_file)
        subdir = input("What is the name of the directory within the HDF5 file, where the ECG data exists? ")
        for i in range(len(file_names)):
            fn = file_names[i]
            patientfile = path + '/' + fn
            fname, ext = splitext(patientfile)

            if ext != '.h5':
                print(file_names[
                          i] + ' - cannot import this file type! Please keep consistent file types when using RR-APET for multiple ECG analysis')

            else:
                file = h5py.File(patientfile, 'r')
                pats = list(file.keys())

                for j in range(len(pats)):
                    f_name = pats[j] + '/' + subdir
                    ECG = file[f_name]
                    ECG = ECG[:]

                    r, c = np.shape(ECG)
                    if c > r:
                        ECG = np.transpose(ECG)
                    R_peaks = ECG[:, 0]
                    R_peaks = R_peaks[R_peaks != 0]
                    R_peaks = np.reshape(R_peaks, [len(R_peaks), ])

                    if Preferences[23] == '1':
                        R_peaks = R_peaks / 1e3
                    if np.mean(
                            np.diff(
                                R_peaks)) < 6:  # Average time interval between heart beats wouldn't be less than 10bpm, so a gap of over 6 seconds on average or greater means loaded in as samples
                        R_peaks = R_peaks * Fs  # Measured in time but need samples

                    if not (np.diff(R_peaks[:]) > 0).all():
                        # They aren't all greater than the previous - therefore RRI series not time-stamps
                        tmp = np.zeros(np.size(R_peaks))
                        tmp[0] = R_peaks[0]

                        for i in range(1, np.size(R_peaks)):
                            tmp[i] = tmp[i - 1] + R_peaks[i]

                    R_peaks = np.reshape(tmp, [len(tmp), 1]) * Fs

                    if len(R_peaks) > 0:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            # ===================Time-domain Statistics====================#
                            SDNN, SDANN, MeanRR, RMSSD, pNN50 = Calculate_Features(R_peaks, FS, decim=Dec)
                            # ===================Frequency-domain Statistics====================#
                            Rpeak_input = R_peaks / FS
                            powULF, powLF, powHF, perpowULF, perpowLF, perpowHF, peak_freq_ULF, peak_freq_LF, peak_freq_HF, LF_HF_ratio = Freq_Analysis(
                                Rpeak_input, meth=FreqMeth, decim=Dec)
                            # ===================Nonlinear statistics====================#
                            RRI = np.diff(Rpeak_input)
                            REC, DET, LAM, Lmean, Lmax, Vmean, Vmax = RQA_plot(RRI, decim=Dec)
                            SD1, SD2 = Poincare(RRI, decim=Dec)
                            alp1, alp2, F = DFA(RRI, decim=Dec)
                        with open(saveroot, "a") as text_file:
                            savename = fn + '->' + f_name
                            print(f"{savename}\t{SDNN}\t{SDANN}\t{MeanRR}\t{RMSSD}\t{pNN50}\t{powULF}\t{powLF}\t" +
                                  f"{powHF}\t{perpowULF}\t{perpowLF}\t{perpowHF}\t{peak_freq_HF}\t{peak_freq_LF}\t" +
                                  f"{peak_freq_HF}\t{LF_HF_ratio}\t{REC}\t{DET}\t{LAM}\t{Lmean}\t{Lmax}\t{Vmean}\t" +
                                  f"{Vmax}\t{SD1}\t{SD2}\t{alp1}\t{alp2}", file=text_file)


    elif dtyp == 3:
        with open(saveroot, "w") as text_file:
            print("file\tSDNN\tSDANN\tMRRI\tRMSSD\tpNN50\tpowULF\tpowLF\tpowHF\t%powULF\t%powLF\t%powHF\tfULF\t" +
                  "fLF\tfHF\tLFHF\tREC\tDET\tLAM\tLmean\tLmax\tVmean\tVmax\tSD1\tSD2\talp1\talp2", file=text_file)

        data_call = input("Please enter complete directory path to ECG data with MAT files: ")

        for i in range(len(file_names)):
            fn = file_names[i]
            patientfile = path + '/' + file_names[i]
            fname, ext = splitext(patientfile)

            if ext != '.mat':
                print(file_names[
                          i] + ' - cannot import this file type! Please keep consistent file types when using RR-APET for multiple ECG analysis')
            else:
                with h5py.File(patientfile, 'r') as hrv:
                    ECG = hrv[data_call][:]
                    r, c = np.shape(ECG)
                    if c > r:
                        ECG = np.transpose(ECG)
                    R_peaks = ECG[:, 0]
                    R_peaks = R_peaks[R_peaks != 0]
                    R_peaks = np.reshape(R_peaks, [len(R_peaks), ])
                    if Preferences[23] == '1':
                        R_peaks = R_peaks / 1e3
                    if np.mean(
                            np.diff(
                                R_peaks)) < 6:  # Average time interval between heart beats wouldn't be less than 10bpm, so a gap of over 6 seconds on average or greater means loaded in as samples
                        R_peaks = R_peaks * Fs  # Measured in time but need samples

                    if not (np.diff(R_peaks[:]) > 0).all():
                        # They aren't all greater than the previous - therefore RRI series not time-stamps
                        tmp = np.zeros(np.size(R_peaks))
                        tmp[0] = R_peaks[0]

                        for i in range(1, np.size(R_peaks)):
                            tmp[i] = tmp[i - 1] + R_peaks[i]

                    R_peaks = np.reshape(tmp, [len(tmp), 1]) * Fs

                    if len(R_peaks) > 0:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            # ===================Time-domain Statistics====================#
                            SDNN, SDANN, MeanRR, RMSSD, pNN50 = Calculate_Features(R_peaks, FS, decim=Dec)
                            # ===================Frequency-domain Statistics====================#
                            Rpeak_input = R_peaks / FS
                            powULF, powLF, powHF, perpowULF, perpowLF, perpowHF, peak_freq_ULF, peak_freq_LF, peak_freq_HF, LF_HF_ratio = Freq_Analysis(
                                Rpeak_input, meth=FreqMeth, decim=Dec)
                            # ===================Nonlinear statistics====================#
                            RRI = np.diff(Rpeak_input)
                            REC, DET, LAM, Lmean, Lmax, Vmean, Vmax = RQA_plot(RRI, decim=Dec)
                            SD1, SD2 = Poincare(RRI, decim=Dec)
                            alp1, alp2, F = DFA(RRI, decim=Dec)
                        with open(saveroot, "a") as text_file:
                            savename = fn + '->' + fname
                            print(f"{savename}\t{SDNN}\t{SDANN}\t{MeanRR}\t{RMSSD}\t{pNN50}\t{powULF}\t{powLF}\t" +
                                  f"{powHF}\t{perpowULF}\t{perpowLF}\t{perpowHF}\t{peak_freq_HF}\t{peak_freq_LF}\t" +
                                  f"{peak_freq_HF}\t{LF_HF_ratio}\t{REC}\t{DET}\t{LAM}\t{Lmean}\t{Lmax}\t{Vmean}\t" +
                                  f"{Vmax}\t{SD1}\t{SD2}\t{alp1}\t{alp2}", file=text_file)

    else:
        print('Cannot import this file type! Use *.txt, *.mat, or *.h5')
    print("Complete.")


# ~~~~~~~~~~~~~~ WINDOWS - MAIN RR-APET PROGRAM ~~~~~~~~~~~~~~~~~~~#

class RRAPET(Frame):

    def __init__(self, parent):
        Frame.__init__(self, parent)
        self._job = None
        self.parent = parent
        # IMPORT IMAGES
        self.openicon = PhotoImage(master=self, file='./Pics/open24.png')
        self.saveicon = PhotoImage(master=self, file='./Pics/savemet24.png')
        self.saveicon2 = PhotoImage(master=self, file='./Pics/saveRT24.png')
        self.updateicon = PhotoImage(master=self, file='./Pics/update24.png')
        self.settingsicon = PhotoImage(master=self, file='./Pics/settings24.png')
        self.closeicon = PhotoImage(master=self, file='./Pics/close24.png')
        self.calcicon = PhotoImage(master=self, file='./Pics/calc24.png')
        self.helpicon = PhotoImage(master=self, file='./Pics/help24.png')
        #        self.printicon = PhotoImage(master=self, file='./Pics/pdf24.png')
        self.editicon = PhotoImage(master=self, file='./Pics/edit24.png')
        self.photo = PhotoImage(master=self, file='./Pics/darrow16.png')
        self.dacontrols = PhotoImage(master=self, file='./Pics/settings24.png')
        #        self.dacontrols = PhotoImage(master=self, file='./Pics/DAcontrols24.png')
        self.batch = PhotoImage(master=self, file='./Pics/batch24.png')
        self.initUI()
        pad = 0
        parent.geometry("{0}x{1}+0+0".format(parent.winfo_screenwidth() - pad, parent.winfo_screenheight() - pad))

    def initUI(self):
        self.parent.bind('<Control-a>', self.onMetrics)
        self.parent.bind('<Control-f>', self.onPref)
        self.parent.bind('<Control-o>', self.onOpen)
        self.parent.bind('<Control-l>', self.onLoad)
        self.parent.bind('<Control-s>', self.cntrls)
        self.parent.bind('<Control-b>', self.cntrlb)
        self.parent.bind('<Control-u>', self.cntrlu)
        self.parent.bind('<Control-m>', savemetrics)
        self.parent.bind('<Control-q>', self.onClose)
        self.parent.bind('<Control-i>', Invert)
        self.parent.bind('<Control-d>', InvertDelete)

        self.parent.bind('<Control-p>', self.fakeCommand)  # PRINT
        self.parent.bind('<Control-h>', self.fakeCommand)  # HELP

        self.parent.title("RR APET")
        self.pack(fill=BOTH, expand=1)

        # SET UP MENUBAR#
        menubar = Menu(self.parent, font=cust_header)
        self.parent.config(menu=menubar)

        fileMenu = Menu(menubar, font=cust_subheadernb, tearoff=False)
        fileMenu.add_command(label="Open", command=self.onOpen)  # ctrl+o
        fileMenu.add_command(label="Load Annotations", command=self.onLoad)  # ctrl+l
        fileMenu.add_command(label="Update Annotations", command=lambda: self.onSave(1))  # ctrl+u
        fileMenu.add_command(label="Save Annotations", command=lambda: self.onSave(2))  # ctrl+a
        fileMenu.add_command(label="Save HRV Metrics", command=savemetrics)  # ctrl+s
        fileMenu.add_command(label="Close", command=self.onClose)  # ctrl+q
        fileMenu.add_command(label="Quit", command=shut2)  # esc
        menubar.add_cascade(label="File", menu=fileMenu, font=cust_header)

        toolMenu = Menu(menubar, font=cust_subheadernb, tearoff=False)
        toolMenu.add_command(label="Preferences", command=self.onPref)  # ctrl+f
        toolMenu.add_command(label="Generate HRV metrics", command=self.onMetrics)  # ctrl+m
        toolMenu.add_command(label="Batch Save", command=lambda: self.onSave(3))
        #        toolMenu.add_command(label="Convert to PDF", command=self.fakeCommand) #ctrl+m
        menubar.add_cascade(label="Tools", menu=toolMenu, font=cust_header)

        helpMenu = Menu(menubar, font=cust_subheadernb, tearoff=False)
        helpMenu.add_command(label="RR-APET User Guide", command=lambda: self.fakeCommand)
        helpMenu.add_command(label="Contact Us", command=lambda: self.fakeCommand)
        menubar.add_cascade(label="Help", menu=helpMenu, font=cust_header)

        # GET THIS AS A LOAD IN VALUE FROM PREFRENCCES AND CONNECT TO PREFERENCES WINDOW
        # CHECK FOR ECG OR RRI!!!!

        if (ECG_pref_on):
            self.LAUNCH_ECG()
        else:
            self.LAUNCH_RRI()

    def LAUNCH_RRI(self):
        global graphCanvas
        global fig
        global t
        global fig2
        global RR_interval_Canvas
        global Slider
        global edit_btn
        global screen_height
        global screen_width
        global upper
        global labelled_flag
        global TOTAL_FRAME
        global plot_fig
        global plot_fig2

        labelled_flag = 0

        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        if TOTAL_FRAME != None:
            TOTAL_FRAME.destroy()

        # SET UP SUB FRAMES FOR STORAGE OF EACH COMPONENT#
        TOTAL_FRAME = Frame(self, bg='light grey')
        TOTAL_FRAME.pack(side=TOP, fill=BOTH, expand=True)

        tsbh = Frame(TOTAL_FRAME, bg='light grey', relief='raised', height=10)  # tsbh = top of screen button housing :)
        tsbh.pack(side=TOP, fill=BOTH, expand=False)
        MF = Frame(TOTAL_FRAME, bg='white smoke')
        MF.pack(side=TOP, fill=BOTH, expand=True)

        T1 = Frame(MF, bg='white smoke')
        T1.pack(side=TOP, fill=BOTH, expand=True)
        RHS_butts = Frame(T1, bg='white smoke')
        RHS_butts.pack(side='right', fill=BOTH, expand=False)

        RRI_plot_housing = Frame(T1, bg='white smoke')
        RRI_plot_housing.pack(side='right', fill='both', expand=True)

        editbut = Button2(RHS_butts, image=self.editicon, compound='center', command=edit_toggle, style='B.TButton',
                          takefocus=False)
        editbut.pack(side='top', anchor='w')
        Link_hover_popup_tips2(editbut,
                               text='Edit annotations. \nLeft click to add positive peak.\nRight click to add negative peak.\nMouse scroll wheel click to \nremove closest peak.')

        # TOP of screen control buttons
        openbutton = Button2(tsbh, image=self.openicon, compound='center', command=self.onOpen, style='C.TButton',
                             takefocus=False)
        openbutton.pack(side='left')
        Link_hover_popup_tips(openbutton, text='Open file (ctrl+O)')

        save1 = Button2(tsbh, image=self.saveicon2, compound='center', command=lambda: self.onSave(2),
                        style='C.TButton', takefocus=False)
        save1.pack(side='left')
        Link_hover_popup_tips(save1, text='Save annotations (ctrl+S)')

        save2 = Button2(tsbh, image=self.updateicon, compound='center', command=lambda: self.onSave(1),
                        style='C.TButton', takefocus=False)
        save2.pack(side='left')
        Link_hover_popup_tips(save2, text='Update annotations (ctrl+U)')

        save4 = Button2(tsbh, image=self.batch, compound='center', command=lambda: self.onSave(3), style='C.TButton',
                        takefocus=False)
        save4.pack(side='left')
        Link_hover_popup_tips(save4, text='Batch Metrics (ctrl+B)')

        #        printer=Button2(tsbh, image=self.printicon, compound='center', command=lambda:self.fakeCommand, style='C.TButton', takefocus=False)
        #        printer.pack(side='left')
        #        Link_hover_popup_tips(printer, text = 'Convert to PDF (ctrl+P)')

        Label(tsbh, text='', width=1, bg='light grey', takefocus=False).pack(side='left')
        genmets = Button2(tsbh, image=self.calcicon, compound='center', command=self.onMetrics, style='C.TButton',
                          takefocus=False)
        genmets.pack(side='left')
        Link_hover_popup_tips(genmets, text='Calculate HRV metrics (ctrl+A)')

        save3 = Button2(tsbh, image=self.saveicon, compound='center', command=savemetrics, style='C.TButton',
                        takefocus=False)
        save3.pack(side='left')
        Link_hover_popup_tips(save3, text='Save metrics (ctrl+M)')

        prefs = Button2(tsbh, image=self.settingsicon, compound='center', command=self.onPref, style='C.TButton',
                        takefocus=False)
        prefs.pack(side='left')
        Link_hover_popup_tips(prefs, text='Preferences (ctrl+F)')

        helps = Button2(tsbh, image=self.helpicon, compound='center', command=lambda: self.fakeCommand,
                        style='C.TButton', takefocus=False)
        helps.pack(side='left')
        Link_hover_popup_tips(helps, text='Help (ctrl+H)')

        Label(tsbh, text='', width=1, bg='light grey', takefocus=False).pack(side='left')
        closebutton = Button2(tsbh, image=self.closeicon, compound='center', command=self.onClose, style='C.TButton',
                              takefocus=False)
        closebutton.pack(side='left')
        Link_hover_popup_tips(closebutton, text='Close file (ctrl+Q)')

        fig2 = Figure(tight_layout=1, facecolor='#f5f5f5')
        plot_fig2 = fig2.add_subplot(111)

        RR_interval_Canvas = FigureCanvasTkAgg(fig2, master=RRI_plot_housing)
        RR_interval_Canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)
        RR_interval_Canvas._tkcanvas

    #        RR_interval_Canvas.mpl_connect('button_press_event', onclick2)
    #
    #        Slider = Scale(Slider_housing, from_=0, to=100, resolution=0.001, showvalue=0, orient=TKc.HORIZONTAL, bg='white', command=self.getSlideValue, length=screen_width-0.05*screen_width)
    #        Slider.pack(side=BOTTOM)
    #
    #        t= Entry(Slider_housing, width = 10, readonlybackground='white')
    #        t.pack(side=RIGHT, anchor='e')
    #        t.insert(0, Preferences[2])
    #        t.bind("<FocusIn>", self.callback)
    #        t.bind("<FocusOut>", self.updateRange)

    #        Label2(Slider_housing, text="Range (s)", style='Text2.TLabel').pack(side='right', anchor='e')

    def LAUNCH_ECG(self):
        global graphCanvas
        global fig
        global t
        global fig2
        global RR_interval_Canvas
        global Slider
        global edit_btn
        global screen_height
        global screen_width
        global upper
        global labelled_flag
        global TOTAL_FRAME
        global plot_fig
        global plot_fig2
        global rhs_ecg_frame

        labelled_flag = 0

        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        if TOTAL_FRAME != None:
            TOTAL_FRAME.destroy()

        # SET UP SUB FRAMES FOR STORAGE OF EACH COMPONENT#
        TOTAL_FRAME = Frame(self, bg='light grey')
        TOTAL_FRAME.pack(side=TOP, fill=BOTH, expand=True)

        tsbh = Frame(TOTAL_FRAME, bg='light grey', relief='raised', height=10)  # tsbh = top of screen button housing :)
        tsbh.pack(side=TOP, fill=BOTH, expand=False)
        MF = Frame(TOTAL_FRAME, bg='white smoke')
        MF.pack(side=TOP, fill=BOTH, expand=True)

        T1 = Frame(MF, bg='white smoke')
        T1.pack(side=TOP, fill=BOTH, expand=True)
        rhs_ecg_frame = Frame(T1, bg='white smoke')
        rhs_ecg_frame.pack(side='right', fill=BOTH, expand=False)
        upper = Frame(rhs_ecg_frame, bg='white smoke')
        upper.pack(side='top', fill=BOTH, expand=False)
        Label2(upper, text='', style='Text2.TLabel').pack(side='top')

        ECG_plot_housing = Frame(T1, bg='white smoke')
        ECG_plot_housing.pack(side='right', fill='both', expand=True)

        T2 = Frame(MF, bg='white smoke')
        T2.pack(side=TOP, fill=BOTH, expand=False)
        mid = Frame(T2, bg='white smoke')
        mid.pack(side='right', fill=BOTH, expand=False)
        Label2(mid, text='', style='Text2.TLabel').pack(side='top')
        Slider_housing = Frame(T2, bg='white smoke')
        Slider_housing.pack(side='right', fill='x', expand=False)

        T3 = Frame(MF, bg='white smoke')
        T3.pack(side=TOP, fill=BOTH, expand=True)
        lower = Frame(T3, bg='white smoke')
        lower.pack(side='right', fill=BOTH, expand=False)
        Label2(lower, text='', style='Text2.TLabel').pack(side='top')
        RRI_housing = Frame(T3, bg='white')
        RRI_housing.pack(side='right', fill='both', expand=True)

        # TOP of screen control buttons
        openbutton = Button2(tsbh, image=self.openicon, compound='center', command=self.onOpen, style='C.TButton',
                             takefocus=False)
        openbutton.pack(side='left')
        Link_hover_popup_tips(openbutton, text='Open file (ctrl+O)')

        save1 = Button2(tsbh, image=self.saveicon2, compound='center', command=lambda: self.onSave(2),
                        style='C.TButton', takefocus=False)
        save1.pack(side='left')
        Link_hover_popup_tips(save1, text='Save annotations (ctrl+S)')

        save2 = Button2(tsbh, image=self.updateicon, compound='center', command=lambda: self.onSave(1),
                        style='C.TButton', takefocus=False)
        save2.pack(side='left')
        Link_hover_popup_tips(save2, text='Update annotations (ctrl+U)')

        save4 = Button2(tsbh, image=self.batch, compound='center', command=lambda: self.onSave(3), style='C.TButton',
                        takefocus=False)
        save4.pack(side='left')
        Link_hover_popup_tips(save4, text='Batch Metrics (ctrl+B)')

        #        printer=Button2(tsbh, image=self.printicon, compound='center', command=lambda:self.fakeCommand, style='C.TButton', takefocus=False)
        #        printer.pack(side='left')
        #        Link_hover_popup_tips(printer, text = 'Convert to PDF (ctrl+P)')

        Label(tsbh, text='', width=1, bg='light grey', takefocus=False).pack(side='left')
        genmets = Button2(tsbh, image=self.calcicon, compound='center', command=self.onMetrics, style='C.TButton',
                          takefocus=False)
        genmets.pack(side='left')
        Link_hover_popup_tips(genmets, text='Calculate HRV metrics (ctrl+A)')

        save3 = Button2(tsbh, image=self.saveicon, compound='center', command=savemetrics, style='C.TButton',
                        takefocus=False)
        save3.pack(side='left')
        Link_hover_popup_tips(save3, text='Save metrics (ctrl+M)')

        prefs = Button2(tsbh, image=self.settingsicon, compound='center', command=self.onPref, style='C.TButton',
                        takefocus=False)
        prefs.pack(side='left')
        Link_hover_popup_tips(prefs, text='Preferences (ctrl+F)')

        helps = Button2(tsbh, image=self.helpicon, compound='center', command=lambda: self.fakeCommand,
                        style='C.TButton', takefocus=False)
        helps.pack(side='left')
        Link_hover_popup_tips(helps, text='Help (ctrl+H)')

        Label(tsbh, text='', width=1, bg='light grey', takefocus=False).pack(side='left')
        closebutton = Button2(tsbh, image=self.closeicon, compound='center', command=self.onClose, style='C.TButton',
                              takefocus=False)
        closebutton.pack(side='left')
        Link_hover_popup_tips(closebutton, text='Close file (ctrl+Q)')

        # ECG related buttons
        editbut = Button2(upper, image=self.editicon, compound='center', command=edit_toggle, style='B.TButton',
                          takefocus=False)
        editbut.pack(side='top', anchor='w')
        Link_hover_popup_tips2(editbut,
                               text='Edit annotations. \nLeft click to add positive peak.\nRight click to add negative peak.\nMouse scroll wheel click to \nremove closest peak.')

        #        Menu for RRI plot
        self.RRI_variable = StringVar(lower)
        self.options = ['RR_P', 'RR_A', 'RR_B']
        RRImenu = OptionMenu(lower, self.RRI_variable, self.options[0], *self.options)
        RRImenu.config(width=5)
        #        RRImenu.configure(compound='right',image=self.photo)
        RRImenu.pack(side='top')
        self.RRI_variable.trace('w', self.change_dropdown)

        fig = Figure(tight_layout=1,
                     facecolor='#f5f5f5')  # Configuration of the Figure to be plotted on the first canvas
        plot_fig = fig.add_subplot(111)  # Adding a subplot which can be updated for viewing of ECG and R-peaks

        graphCanvas = FigureCanvasTkAgg(fig,
                                        master=ECG_plot_housing)  # Using TkAgg as a backend to bind the figure to the canvas
        graphCanvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)  # Positioning of figure within canvas window
        graphCanvas._tkcanvas  # .pack(side=TOP, fill=BOTH, expand=True)                   #Ensuring that "graphCanvas" can be bound to mouse input later on

        fig2 = Figure(tight_layout=1, facecolor='#f5f5f5')
        plot_fig2 = fig2.add_subplot(111)

        RR_interval_Canvas = FigureCanvasTkAgg(fig2, master=RRI_housing)
        RR_interval_Canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)
        RR_interval_Canvas._tkcanvas
        RR_interval_Canvas.mpl_connect('button_press_event', onclick2)

        Slider = Scale(Slider_housing, from_=0, to=100, resolution=0.001, showvalue=0, orient=TKc.HORIZONTAL,
                       bg='white', command=self.getSlideValue, length=screen_width - 0.05 * screen_width)
        Slider.pack(side=BOTTOM)

        t = Entry(Slider_housing, width=10, readonlybackground='white')
        t.pack(side=RIGHT, anchor='e')
        t.insert(0, Preferences[2])
        t.bind("<FocusIn>", self.callback)
        t.bind("<FocusOut>", self.updateRange)

        Label2(Slider_housing, text="Range (s)", style='Text2.TLabel').pack(side='right', anchor='e')

    #

    def callbackFunc(self, event):
        print("New Element Selected")

    def change_dropdown(self, *args):
        global enable
        global plot_pred
        global plot_ann
        global loaded_ann
        arg = self.RRI_variable.get()

        enable ^= 1
        if enable:
            if (arg == 'RR_P'):
                plot_pred = 1
                plot_ann = 0
                draw1()
            elif loaded_ann == 1:
                if (arg == 'RR_A'):
                    plot_pred = 0
                    plot_ann = 1
                    draw1()
                elif (arg == 'RR_B'):
                    plot_pred = 1
                    plot_ann = 1
                    draw1()
            else:
                messagebox.showwarning("Warning", "External annotations have not been loaded into the program. \n\n" +
                                       "RR_Annotations and RR_Both are invalid selections - please load annotations before selecting these options.")
                plot_pred = 1
                plot_ann = 0
                self.RRI_variable.set(self.options[0])

    def callback(self, event):
        global t
        t.config(insertontime=600, state='normal')
        t.bind("<Return>", self.updateRange)
        t.bind("<KP_Enter>", self.updateRange)

    def updateRange(self, event):
        global t
        global disp_length
        t.config(insertontime=0, state='readonly')
        t.unbind("<Return>")
        t.unbind("<KP_Enter>")
        newdisp = int(t.get())
        if (newdisp != disp_length):
            disp_length = newdisp
            draw1()

    def getSlideValue(self, event):
        if self._job:
            self.after_cancel(self._job)
        self._job = self.after(100, self.onSlide)

    def onSlide(self):
        global x
        global dat
        global xminn
        global Fs
        global plot_fig
        global graphCanvas
        global Slider
        global disp_length
        self._job = None
        tmp = int(((len(dat) / Fs) * Slider.get() / 100) - 0.5 * disp_length)
        if tmp < 0:
            tmp = 0

        xminn = tmp * Fs
        draw1()

    def onClose(self, event=1):
        global dat
        global x
        global xminn
        global Fs
        global tcol
        global tt
        global loaded_ann
        global invert_flag
        global enable
        global enable2
        global plot_ann
        global plot_pred
        global R_t
        global mets
        global plot_wind
        global plot_fig
        global plot_fig2
        global graphCanvas
        global RR_interval_Canvas
        global pltmenu
        global ECG_pref_on

        tt = 4
        loaded_ann = 0
        invert_flag = 0
        tcol = 0
        enable = 0
        enable2 = 0
        plot_pred = 1
        plot_ann = 0
        R_t = []
        x = []
        xminn = []
        dat = []

        try:
            pltmenu.destroy()
        except:
            pass

        try:
            mets.withdraw()
            mets = None
        except:
            mets = None
        try:
            plot_wind.withdraw()
            plot_wind = None
        except:
            plot_wind = None

        file = open("Preferences.txt", 'r')
        P = file.read().split()
        file.close()
        ECG_pref_on = int(P[21])

        if (ECG_pref_on):
            self.LAUNCH_ECG()
        else:
            self.LAUNCH_RRI()

    def onOpen(self, event=1):
        global dat
        global x
        global xminn
        global Fs
        global tcol
        global load_dat
        global pref
        global plot_wind
        global ECG_pref_on
        global R_t
        global labelled_flag

        file = open("Preferences.txt", 'r')
        Preferences = file.read().split()
        file.close()

        if ECG_pref_on:
            ftypes = [('All files', '*'), ('Text files', '*.txt'), ('HDF5 files', '*.h5'), ('MAT files', '*.mat'),
                      ('WFDB files', '*.dat')]
            #        ftypes = [('HDF5 files', '*.h5'), ('Text files','*.txt'), ('MAT files', '*.mat'), ('All files', '*')]
            input_file = filedialog.Open(self, filetypes=ftypes)
            F = input_file.show()
            filename, file_extension = os.path.splitext(F)
            go_ahead = 1

            if pref != None:
                pref.withdraw()

            if plot_wind is not None:
                plot_wind.destroy()

            if file_extension == '.txt':
                file = open(F, 'r')
                if (len(F) == 0 & warnings_on):
                    messagebox.showwarning("Warning",
                                           "The file you have selected is unreadable. Please ensure that the file interest is saved in "
                                           + "the correct format. For further information, refer to the 'Help' module")
                    go_ahead = 0
                else:
                    if (Preferences[24] == '0'):
                        print('mm')
                        temp = file.read().split()
                    elif (Preferences[24] == '1'):
                        temp = file.read().split(":")
                    else:
                        temp = file.read().split(";")
                    var1 = len(temp)
                    load_dat = np.zeros(np.size(temp))
                    for i in range(var1):  # for i in range(len(temp)):
                        load_dat[i] = float(temp[i].rstrip('\n'))
                    file.seek(0)
                    temp2 = file.readlines()
                    var2 = len(temp2)
                    columns = int(var1 / var2)
                    load_dat = np.reshape(load_dat, [len(temp2), int(columns)])
                    if (columns > var2):
                        load_dat = np.transpose(load_dat)
                        columns = var2

                file.close()

            elif ((file_extension == '.h5') or (file_extension == '.mat')):
                global h5window
                h5window = Toplevel()
                H5_selector(h5window, (F, 'data'))
                if windows_compile:
                    h5window.bind('<Escape>', close_h5win)
                if linux_compile:
                    h5window.bind('<Control-Escape>', close_h5win)
                go_ahead = 0

            elif ((file_extension == '.dat') or (file_extension == '.hea')):
                load_dat, fields = wfdb.rdsamp(filename)
                rub, columns = np.shape(load_dat)

            elif (warnings_on):
                messagebox.showwarning("Warning",
                                       "The file type you have selected in not compatible. Please select a *.txt, *.h5, *.mat, or *.dat file only.")
                go_ahead = 0

            if go_ahead:

                # DETERMINE IF DATA INCLUDES TIME STAMPS OR NOT
                if (np.diff(load_dat[:, 0]) > 0).all() == True:
                    tcol = 1
                    opts = []
                    for ix in range(1, columns):
                        opts = np.append(opts, str(ix))
                    setupperbutton(opts)
                    # They are all increasing therefore time series - use next column
                    if (Preferences[23] == '0'):
                        Fs = int(1 / (np.mean(np.diff(load_dat[:, 0]))))
                    else:  # if ms divide to get s
                        Fs = int(1 / (np.mean(np.diff(load_dat[:, 0] / 1e3))))
                    dat = load_dat[:, 1]
                    if (Preferences[22] == '1'):  # if Volts multiple to get mV
                        dat = dat * 1e3
                    dat = np.reshape(dat, [len(dat), 1])
                    x = np.arange(len(dat))
                    xminn = 0
                    Prediction_mode(1)
                    draw1()

                else:
                    # They vary in magintude and are therefore the ECG of interest
                    tcol = 0
                    opts = []
                    for ix in range(columns):
                        opts = np.append(opts, str(ix))
                    setupperbutton(opts)
                    dat = load_dat[:, 0]
                    if (Preferences[22] == '1'):  # if Volts multiple to get mV
                        dat = dat * 1e3
                    dat = np.reshape(dat, [len(dat), 1])
                    x = np.arange(len(dat))
                    xminn = 0
                    onNoFSdata()

        else:
            ftypes = [('All files', '*'), ('Text files', '*.txt'), ('HDF5 files', '*.h5'), ('MAT files', '*.mat'),
                      ('WFDB files', '*.atr')]
            #            ftypes = [('Text files','*.txt'), ('All files', '*')]
            input_file = filedialog.Open(self, filetypes=ftypes)
            F = input_file.show()
            filename, file_extension = os.path.splitext(F)
            go_ahead = 1

            if pref != None:
                pref.withdraw()

            if plot_wind is not None:
                plot_wind.destroy()

            if file_extension == '.txt':
                column_number = 1
                file = open(F, 'r')
                if Preferences[24] == '0':
                    temp = file.read().split()
                elif Preferences[24] == '1':
                    temp = file.read().split(":")
                else:
                    temp = file.read().split(";")
                var1 = len(temp)
                og_ann = np.zeros(np.size(temp))
                for i in range(len(temp)):
                    og_ann[i] = float(temp[i].rstrip('\n'))

                file.seek(0)
                temp2 = file.readlines()
                var2 = len(temp2)
                columns = var1 / var2
                og_ann = np.reshape(og_ann, [len(temp2), int(columns)])
                if columns > var2:
                    og_ann = np.transpose(og_ann)

                file.close()
                R_t = []
                R_t = og_ann[:, (column_number - 1)]


            elif ((file_extension == '.h5') or (file_extension == '.mat')):
                global h5window2
                h5window2 = Toplevel()
                H5_selector(h5window2, (F, 'data'))
                if windows_compile:
                    h5window2.bind('<Escape>', close_h5win2)
                if linux_compile:
                    h5window2.bind('<Control-Escape>', close_h5win2)
                go_ahead = 0

            elif file_extension == '.atr':
                ann = wfdb.rdann(filename, 'atr')
                stamps = ann.sample
                syms = ann.symbol

                R_t = np.zeros(len(stamps), dtype='int64')
                for i in range(len(stamps)):
                    # Include only beat annotations
                    if ((syms[i] == 'N') or (syms[i] == 'L') or (syms[i] == 'R') or (syms[i] == 'B') or (
                            syms[i] == 'A') or (syms[i] == 'a') or (syms[i] == 'J') or (syms[i] == 'S') or (
                            syms[i] == 'V') or (syms[i] == 'r') or (syms[i] == 'F') or (syms[i] == 'e') or (
                            syms[i] == 'j') or (syms[i] == 'n') or (syms[i] == 'E') or (syms[i] == '/') or (
                            syms[i] == 'f') or (syms[i] == 'Q') or (syms[i] == '?')):
                        R_t[i] = stamps[i]

            elif (warnings_on):
                messagebox.showwarning("Warning",
                                       "The file type you have selected in not compatible. Please select a *.txt, *.h5, *.mat, or *.atr file only.")
                go_ahead = 0

            if go_ahead:
                # DETERMINE IF DATA INCLUDES TIME STAMPS OR NOT
                R_t = R_t[R_t != 0]
                R_t = np.reshape(R_t, [len(R_t), ])

                if (np.diff(R_t[:]) > 0).all() == False:
                    # They aren't all greater than the previous - therefore RRI series not time-stamps
                    tmp = np.zeros(np.size(R_t))
                    tmp[0] = R_t[0]

                    for i in range(1, np.size(R_t)):
                        tmp[i] = tmp[i - 1] + R_t[i]

                if (Preferences[23] == '1'):
                    R_t = R_t / 1e3
                R_t = np.reshape(R_t, [len(R_t), 1])
                labelled_flag = 1
                onNoFSdata()

    def onLoad(self, event=1):
        global dat
        global xminn
        global Fs
        global loaded_ann
        global True_R_t
        global True_R_amp

        file = open("Preferences.txt", 'r')
        Preferences = file.read().split()
        file.close()

        if ECG_pref_on:
            ftypes = [('Text files', '*.txt'), ('Mat files', '*.mat'), ('HDF5 files', '*.h5'), ('WFDB files', '*.atr'),
                      ('All files', '*')]

            input_file = filedialog.Open(self, filetypes=ftypes)
            annfile = input_file.show()
            filename, file_extension = os.path.splitext(annfile)
            # ~~~~~~~~~~~~~~ ASSUMPTIONS HAVE BEEN MADE about column number and annotation type  ~~~~~~~~~~~~~~~~~~~#

            go_ahead = 1
            loaded_ann = 1
            column_number = 1
            if file_extension == '.txt':
                file = open(annfile, 'r')
                if (Preferences[24] == '0'):
                    temp = file.read().split()
                elif (Preferences[24] == '1'):
                    temp = file.read().split(":")
                else:
                    temp = file.read().split(";")
                var1 = len(temp)
                og_ann = np.zeros(np.size(temp))
                for i in range(len(temp)):
                    og_ann[i] = float(temp[i].rstrip('\n'))

                file.seek(0)
                temp2 = file.readlines()
                var2 = len(temp2)
                columns = var1 / var2
                og_ann = np.reshape(og_ann, [len(temp2), int(columns)])
                if (columns > var2):
                    og_ann = np.transpose(og_ann)

                file.close()

                True_R_t = []
                True_R_t = og_ann[:, (column_number - 1)]

                if (Preferences[23] == '1'):
                    True_R_t = True_R_t / 1e3

                if np.mean(
                        np.diff(
                            True_R_t)) < 6:  # Average time interval between heart beats wouldn't be less than 10bpm, so a gap of over 6 seconds on average or greater means loaded in as samples
                    True_R_t = True_R_t * Fs  # Measured in time but need samples


            elif ((file_extension == '.h5') or (file_extension == '.mat')):
                global h5window
                h5window = Toplevel()
                H5_selector(h5window, (annfile, 'ann'))
                if windows_compile:
                    h5window.bind('<Escape>', close_h5win)
                if linux_compile:
                    h5window.bind('<Control-Escape>', close_h5win)
                go_ahead = 0

            elif (file_extension == '.atr'):
                ann = wfdb.rdann(filename, 'atr')
                stamps = ann.sample
                syms = ann.symbol

                True_R_t = np.zeros(len(stamps), dtype='int64')
                for i in range(len(stamps)):
                    # Include only beat annotations
                    if ((syms[i] == 'N') or (syms[i] == 'L') or (syms[i] == 'R') or (syms[i] == 'B') or (
                            syms[i] == 'A') or (syms[i] == 'a') or (syms[i] == 'J') or (syms[i] == 'S') or (
                            syms[i] == 'V') or (syms[i] == 'r') or (syms[i] == 'F') or (syms[i] == 'e') or (
                            syms[i] == 'j') or (syms[i] == 'n') or (syms[i] == 'E') or (syms[i] == '/') or (
                            syms[i] == 'f') or (syms[i] == 'Q') or (syms[i] == '?')):
                        True_R_t[i] = stamps[i]

            else:
                loaded_ann = 0
                messagebox.showwarning("Warning",
                                       "The file type you have selected in not compatible. Please select a *.txt, *.h5, *.mat, or *.atr file only.")
                go_ahead = 0
            #
            if go_ahead:
                # DETERMINE IF DATA INCLUDES TIME STAMPS OR NOT
                True_R_t = True_R_t[True_R_t != 0]
                True_R_t = np.reshape(True_R_t, [len(True_R_t), ])

                if (np.diff(True_R_t[:]) > 0).all() == False:
                    # They aren't all greater than the previous - therefore RRI series not time-stamps
                    tmp = np.zeros(np.size(True_R_t))
                    tmp[0] = True_R_t[0]

                    for i in range(1, np.size(True_R_t)):
                        tmp[i] = tmp[i - 1] + True_R_t[i]

                True_R_t = np.reshape(True_R_t, [len(True_R_t), 1])
                True_R_amp = np.zeros(np.size(True_R_t))
                for i in range(0, np.size(True_R_t)):
                    True_R_amp[i] = dat[int(True_R_t[i])]

        else:
            messagebox.showwarning("Warning", "When operating in RRI mode, use 'Open' not 'Load Annotations'...")

    def cntrls(self, event=1):
        self.onSave(2)

    def cntrlu(self, event=1):
        self.onSave(1)

    def cntrlb(self, event=1):
        self.onSave(3)

    def onSave(self, stype):
        global R_amp
        global R_t
        global True_R_amp
        global True_R_t
        global ECG_pref_on

        if ECG_pref_on:
            if stype == 1:
                stype = 2

        if (stype == 1):  # UPDATES CHANGES TO A LOADED ANNOTATION FILE
            if windows_compile:
                saveroot = filedialog.asksaveasfilename(title="Select file", defaultextension=".*",
                                                        filetypes=(("text files", "*.txt"), ("all files", "*.*")))
            if linux_compile:
                saveroot = filedialog.asksaveasfilename(title="Select file",
                                                        filetypes=(("text files", "*.txt"), ("all files", "*.*")))
            fname, file_extension = os.path.splitext(saveroot)
            if (loaded_ann) == 1:
                if file_extension == '.h5':
                    f = h5py.File(saveroot, 'w')
                    name1 = '/amp'
                    name2 = '/time'
                    f.create_dataset(name1, data=True_R_amp)
                    f.create_dataset(name2, data=True_R_t / Fs)

                elif file_extension == '.txt':
                    np.savetxt(saveroot, True_R_t / Fs, fmt='%.5e')

                elif file_extension == '.mat':
                    sio.savemat(saveroot, True_R_t / Fs)

                elif warnings_on:
                    messagebox.showwarning("Warning",
                                           "Annotations were not updated! \n\nPlease Note: Currently RR-APET cannot export the file type selected" +
                                           "")
            elif warnings_on:
                messagebox.showwarning("Warning",
                                       "Annotations were not updated! \n\nPlease Note: Updating annotations requires an imported annotation file"
                                       + " (red data-points on RRI plot) that has been edited. The 'Save Annotations' option allows annotations that were generated "
                                       + "within the program (blue data-points on RRI plot) to be saved.")

        elif (stype == 2):
            if windows_compile:
                saveroot = filedialog.asksaveasfilename(title="Select file", defaultextension=".*",
                                                        filetypes=(("text files", "*.txt"), ("all files", "*.*")))
            if linux_compile:
                saveroot = filedialog.asksaveasfilename(title="Select file",
                                                        filetypes=(("text files", "*.txt"), ("all files", "*.*")))
            fname, file_extension = os.path.splitext(saveroot)
            if file_extension == '.h5':
                f = h5py.File(saveroot, 'w')
                name1 = '/amp'
                name2 = '/time'
                if ECG_pref_on:
                    f.create_dataset(name1, data=R_amp)
                f.create_dataset(name2, data=R_t / Fs)

            elif file_extension == '.txt':
                np.savetxt(saveroot, R_t / Fs, fmt='%.5e')

            elif file_extension == '.mat':
                sio.savemat(saveroot, R_t / Fs)

            else:
                print('Cannot export this file type')


        elif (stype == 3):
            global PATH
            global fs_frame_2
            input_file = filedialog.askdirectory()
            PATH = input_file
            fs_frame_2 = Toplevel()
            File_type_and_Fs(fs_frame_2)
            if windows_compile:
                fs_frame_2.bind('<Escape>', close_fs2)
            if linux_compile:
                fs_frame_2.bind('<Control-Escape>', close_fs2)

    def onPref(self, event=1):
        global pref
        pref = Toplevel()
        UserPreferences(pref)
        if windows_compile:
            pref.bind('<Escape>', close_pref)
        if linux_compile:
            pref.bind('<Control-Escape>', close_pref)

    def onMetrics(self, event=1):
        global mets
        mets = Toplevel()
        HRVstatics(mets)
        if windows_compile:
            mets.bind('<Escape>', close_mets)
        if linux_compile:
            mets.bind('<Control-Escape>', close_mets)

    def readFile(self, filename):

        f = open(filename, "r")
        text = f.read()
        return text

    def fakeCommand(self, HA):
        print(HA)


# ~~~~~~~~~~~~~~ WINDOWS - USER PREFERENCES WINDOW ~~~~~~~~~~~~~~~~~~~#
class UserPreferences(Frame):

    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.initUI_pref()

    def initUI_pref(self):
        global test_txt
        global Outerframe
        global F
        global F2
        global big_frame
        global btn
        global trybtn
        global resetbtn
        global okbtn

        self.contructionImage = PhotoImage(master=self, file='./Pics/construction.png')
        self.contructionImage = self.contructionImage.subsample(3, 3)

        file = open("Preferences.txt", 'r')
        self.pref = file.read().split()
        file.close()

        Outerframe = None
        F = None
        F2 = None
        test_txt = None
        btn = None
        trybtn = None
        resetbtn = None
        okbtn = None

        self.parent.title("Preferences")
        self.parent.resizable(width=FALSE, height=FALSE)
        self.parent.configure(highlightthickness=1, highlightbackground='grey')

        big_frame = Frame(self.parent, bg='white smoke', borderwidth=20)
        big_frame.pack(side='top')
        #        button_frame.grid(row=0,column=0,rowspan=5)
        button_frame = Frame(big_frame, bg='white', borderwidth=10, highlightcolor='grey', highlightthickness=1,
                             highlightbackground='black')
        button_frame.pack(side='left', fill='y')
        #        Button(button_frame, text = "Text Options", command=self.text_op, relief='flat', width = 25, anchor='w', font=Cust, bg = 'white').pack()
        #        Button(button_frame, text = "ECG View Options", command=self.ecg_op, relief='flat', width = 25, anchor='w', font=Cust, bg = 'white').pack()
        #        Button(button_frame, text = "Prediction Mode Settings", command=self.prediction_op, relief='flat', width = 25, anchor='w', font=Cust, bg = 'white').pack()
        #        Button(button_frame, text = "HRV Metric Settings Options", command=self.metric_op, relief='flat', width = 25, anchor='w', font=Cust, bg = 'white').pack()
        #        Button(button_frame, text = "Other Options", command=self.fakeCommand, relief='flat', width = 25, anchor='w', font=Cust, bg = 'white').pack()

        self.btn1 = Button2(button_frame, text="Text Options", command=self.text_op, style='UserPref.TButton')
        self.btn1.pack()
        #        self.btn2 = Button2(button_frame, text = "ECG View Options", command=self.ecg_op, style='UserPref.TButton')
        #        self.btn2.pack()
        self.btn3 = Button2(button_frame, text="Prediction Mode Settings", command=self.prediction_op,
                            style='UserPref.TButton')
        self.btn3.pack()
        self.btn4 = Button2(button_frame, text="HRV Analysis Settings", command=self.metric_op,
                            style='UserPref.TButton')
        self.btn4.pack()
        self.btn5 = Button2(button_frame, text="General Settings", command=self.text_analysis_op,
                            style='UserPref.TButton')
        self.btn5.pack()

        self.text_op()

    def text_op(self):
        global Outerframe
        global big_frame
        global test_txt
        global btn
        global trybtn

        self.btn1.config(style='SelectUserPref.TButton', takefocus=False)
        #        self.btn2.config(style='UserPref.TButton', takefocus=False)
        self.btn3.config(style='UserPref.TButton', takefocus=False)
        self.btn4.config(style='UserPref.TButton', takefocus=False)
        self.btn5.config(style='UserPref.TButton', takefocus=False)

        if Outerframe is not None:
            Outerframe.destroy()
            btn.destroy()

        if test_txt is not None:
            test_txt.destroy()

        if trybtn is not None:
            trybtn.destroy()

        if resetbtn is not None:
            resetbtn.destroy()
            okbtn.destroy()

        Outerframe = Frame(big_frame, bg='white', borderwidth=10, highlightcolor='grey', highlightthickness=1,
                           highlightbackground='black')
        Outerframe.pack(side='left', fill='y')
        F = Frame(Outerframe)
        F.pack(side='left')
        F2 = Frame(Outerframe)
        F2.pack(side='left')

        Label(F, text="Style", font=cust_subheadernb).pack(side="top")
        font_nodes = Listbox(F, font=cust_text, exportselection=0, width=20)
        font_nodes.pack(side="left", fill="y")
        fonts = ['Helvetica', 'Courier', 'FreeSans', 'FreeSerif', 'Times', 'Verdana']
        for i in range(len(fonts)):
            font_nodes.insert(END, fonts[i])

        Label(F2, text="Size", font=cust_subheadernb).pack(side="top")
        text_size_nodes = Listbox(F2, font=cust_text, exportselection=0, width=5)
        text_size_nodes.pack(side="left", fill="y")
        scrollbar_tsz = Scrollbar(F2, orient="vertical")
        scrollbar_tsz.config(command=text_size_nodes.yview)
        scrollbar_tsz.pack(side="right", fill="y")
        text_size_nodes.config(yscrollcommand=scrollbar_tsz.set)
        sizes = ['8', '9', '10', '11', '12', '13', '14', '16', '18', '20', '22', '24', '28', '32']
        for i in range(len(sizes)):
            text_size_nodes.insert(END, sizes[i])

        btn = Button(self.parent, text="Ok",
                     command=lambda: self.update_font(0, font_nodes.get(ANCHOR), text_size_nodes.get(ANCHOR)),
                     font=cust_text)
        btn.pack(side='right', anchor='e')

        trybtn = Button(self.parent, text="Try",
                        command=lambda: self.update_font(1, font_nodes.get(ANCHOR), text_size_nodes.get(ANCHOR)),
                        font=cust_text)
        trybtn.pack(side='right', anchor='e')

        test_txt = Label(self.parent, text="Sample Text", font=cust_text)
        test_txt.pack()

    #    def ecg_op(self):
    #        global Outerframe
    #        global big_frame
    #        global test_txt
    #        global btn
    #        self.btn1.config(style='UserPref.TButton', takefocus=False)
    #        self.btn2.config(style='SelectUserPref.TButton', takefocus=False)
    #        self.btn3.config(style='UserPref.TButton', takefocus=False)
    #        self.btn4.config(style='UserPref.TButton', takefocus=False)
    #        self.btn5.config(style='UserPref.TButton', takefocus=False)
    #
    #        file = open("Preferences.txt", 'r')
    #        Preferences = file.read().split()
    #
    #        if Outerframe is not None:
    #            Outerframe.destroy()
    #            btn.destroy()
    #
    #        if resetbtn is not None:
    #            resetbtn.destroy()
    #            okbtn.destroy()
    #
    #        if test_txt is not None:
    #            test_txt.destroy()
    #
    #        if trybtn is not None:
    #            trybtn.destroy()
    #
    #        Outerframe = Frame(big_frame, bg = 'white', borderwidth=10, highlightcolor='grey', highlightthickness=1, highlightbackground='black')
    #        Outerframe.pack(side='left', fill='y')
    #        F = Frame(Outerframe)
    #        F.pack(side='left')
    #        F2 = Frame(Outerframe)
    #        F2.pack(side='left')
    #
    #        Label(F, text="Range", font=cust_text).pack()
    #
    #        t= Entry(F2, width = 10)
    #        t.pack()
    #        t.insert(0, Preferences[2])
    #
    #        btn = Button(self.parent, text = "Ok", command = lambda: self.update_ecg_settings(t.get()), font=cust_text)
    #        btn.pack(side='right', anchor='e')

    def prediction_op(self):

        global Meth
        global Outerframe
        global big_frame
        global test_txt
        global btn
        global entry

        if Outerframe is not None:
            Outerframe.destroy()
            btn.destroy()

        if test_txt is not None:
            test_txt.destroy()

        if trybtn is not None:
            trybtn.destroy()

        Outerframe = Frame(big_frame, bg='white', borderwidth=10, highlightcolor='grey', highlightthickness=1,
                           highlightbackground='black')
        Outerframe.pack(side='left', fill='y')

        file = open("Preferences.txt", 'r')
        txtfile = file.read().split()
        file.close()

        self.btn1.config(style='UserPref.TButton', takefocus=False)
        #        self.btn2.config(style='UserPref.TButton', takefocus=False)
        self.btn3.config(style='SelectUserPref.TButton', takefocus=False)
        self.btn4.config(style='UserPref.TButton', takefocus=False)
        self.btn5.config(style='UserPref.TButton', takefocus=False)

        Outerframe = Frame(big_frame, bg='white', borderwidth=10, highlightcolor='grey', highlightthickness=1,
                           highlightbackground='black')
        Outerframe.pack(side='left', fill='y')
        F = Frame(Outerframe, bg='white')
        F.pack(side='top', fill='y')

        Label2(F, text='Detection Algorithm Parameters', style='Header.TLabel').grid(row=2, column=0, columnspan=3,
                                                                                     sticky='w')
        Label(F, text='Detection Method', bg='white', font=cust_text, width=22, anchor='w').grid(row=3, column=0,
                                                                                                 sticky='w')
        Meth = StringVar(F)
        Opts_meth = ['MHTD', 'Pan-Tompkins', 'K-means', 'Own Method/Code']
        Method_menu = OptionMenu(F, Meth, Opts_meth[int(txtfile[3]) - 1], *Opts_meth)
        Method_menu.config(style='Text.TMenubutton')
        Method_menu.grid(row=3, column=1)
        Meth.trace('w', self.F2vals)

        self.F2vals()

    def F2vals(self, *args):
        global F2
        global Meth
        global btn
        global resetbtn
        global okbtn
        global Outerframe

        if F2 is not None:
            F2.destroy()
        if btn is not None:
            btn.destroy()
        if resetbtn is not None:
            resetbtn.destroy()
            okbtn.destroy()

        file = open("Preferences.txt", 'r')
        self.vals = file.read().split()
        file.close()

        F2 = Frame(Outerframe, bg='white')
        F2.pack(side='top', fill='both')

        val = Meth.get()
        if (val == 'MHTD'):
            Label(F2, text='Threshold Ratio', bg='white', font=cust_text, width=22, anchor='w').grid(row=1, column=0,
                                                                                                     sticky='w')
            self.thr = Entry(F2, font=cust_text, width=10)
            self.thr.grid(row=1, column=1, sticky='e')
            self.thr.insert(0, self.vals[16])
            Label(F2, text='Search Back Length (s)', bg='white', font=cust_text, width=22, anchor='w').grid(row=2,
                                                                                                            column=0,
                                                                                                            sticky='w')
            self.sbl = Entry(F2, font=cust_text, width=10)
            self.sbl.grid(row=2, column=1)
            self.sbl.insert(0, self.vals[17])
            Label(F2, text='Magnitude Limit (%)', bg='white', font=cust_text, width=22, anchor='w').grid(row=3,
                                                                                                         column=0,
                                                                                                         sticky='w')
            self.magL = Entry(F2, font=cust_text, width=10)
            self.magL.grid(row=3, column=1)
            self.magL.insert(0, self.vals[18])
            Label(F2, text='Energy Limit (%)', bg='white', font=cust_text, width=22, anchor='w').grid(row=4, column=0,
                                                                                                      sticky='w')
            self.engL = Entry(F2, font=cust_text, width=10)
            self.engL.grid(row=4, column=1)
            self.engL.insert(0, self.vals[19])
            Label(F2, text='Min. RR time (s)', bg='white', font=cust_text, width=22, anchor='w').grid(row=5, column=0,
                                                                                                      sticky='w')
            self.minL = Entry(F2, font=cust_text, width=10)
            self.minL.grid(row=5, column=1)
            self.minL.insert(0, self.vals[20])

            okbtn = Button(self.parent, text="Ok", command=lambda: self.repredict(1, 1),
                           font=cust_text)  # thr.get(),sbl.get(),magL.get(),engL.get(),minL.get()
            okbtn.pack(side='right', anchor='e')
            btn = Button(self.parent, text="Test", command=lambda: self.repredict(1, 0),
                         font=cust_text)  # thr.get(),sbl.get(),magL.get(),engL.get(),minL.get()
            btn.pack(side='right', anchor='e')
            resetbtn = Button(self.parent, text="Reset", command=lambda: self.reset(1), font=cust_text)
            resetbtn.pack(side='right', anchor='e')

        elif (val == 'Pan-Tompkins'):
            txt = "Pan-Tompkins currently only avaliable in RR-APET's implementation of it's original form (i.e. unable to alter parameters; however, this function will be avaliable in later releases)."
            Label(F2, text=txt, bg='white', font=cust_text, width=35, anchor='w', wraplength=330).grid(row=3, column=0,
                                                                                                       sticky='w')
            #            Label(F2, text='Under construction', bg='white', font = cust_text, width=22, anchor='w').grid(row=3, column=0, sticky='w')

            imag = Label(F2, image=self.contructionImage, compound='center', takefocus=False, bg='white')
            imag.grid(row=4, column=0, columnspan=3, sticky='e' + 'w')

            okbtn = Button(self.parent, text="Ok", command=lambda: self.repredict(2, 1),
                           font=cust_text)  # thr.get(),sbl.get(),magL.get(),engL.get(),minL.get()
            okbtn.pack(side='right', anchor='e')
            btn = Button(self.parent, text="Test", command=lambda: self.repredict(2, 0), font=cust_text)
            btn.pack(side='right', anchor='e')
            resetbtn = Button(self.parent, text="Reset", command=lambda: self.reset(2), font=cust_text)
            resetbtn.pack(side='right', anchor='e')


        elif (val == 'K-means'):
            txt = "K-means currently only avaliable in RR-APETs implementation of it's original form (i.e. unable to alter parameters; however, this function will be avaliable in later releases)."
            Label(F2, text=txt, bg='white', font=cust_text, width=35, anchor='w', wraplength=330).grid(row=3, column=0,
                                                                                                       sticky='w')
            imag = Label(F2, image=self.contructionImage, compound='center', takefocus=False, bg='white')
            imag.grid(row=4, column=0, columnspan=3, sticky='e' + 'w')
            okbtn = Button(self.parent, text="Ok", command=lambda: self.repredict(3, 1),
                           font=cust_text)  # thr.get(),sbl.get(),magL.get(),engL.get(),minL.get()
            okbtn.pack(side='right', anchor='e')
            btn = Button(self.parent, text="Test", command=lambda: self.repredict(3, 0), font=cust_text)
            btn.pack(side='right', anchor='e')
            resetbtn = Button(self.parent, text="Reset", command=lambda: self.reset(3), font=cust_text)
            resetbtn.pack(side='right', anchor='e')


        elif (val == 'Own Method/Code'):
            txt = "Change the settings of your own method in the provided python script; or use the provided code and insert your options below..."
            Label(F2, text=txt, bg='white', font=cust_text, width=35, anchor='w', wraplength=330).grid(row=3, column=0,
                                                                                                       columnspan=3,
                                                                                                       sticky='ew')

            okbtn = Button(self.parent, text="Ok", command=lambda: self.repredict(4, 1),
                           font=cust_text)  # thr.get(),sbl.get(),magL.get(),engL.get(),minL.get()
            okbtn.pack(side='right', anchor='e')
            btn = Button(self.parent, text="Test", command=lambda: self.repredict(4, 0), font=cust_text)
            btn.pack(side='right', anchor='e')
            resetbtn = Button(self.parent, text="Reset", command=lambda: self.reset(4), font=cust_text)
            resetbtn.pack(side='right', anchor='e')

    def repredict(self, val, closewind):

        if val == 1:
            try:
                TR = float(self.thr.get())
                SB = int(self.sbl.get())
                ML = float(self.magL.get())
                EL = float(self.engL.get())
                mL = float(self.minL.get())
                Prediction_mode(1, thr_ratio=TR, SBL=SB, MAG_LIM=ML, ENG_LIM=EL, MIN_L=mL)
                if closewind == 1:
                    pref.withdraw()
                    replace_line("Preferences.txt", 3, str(1) + '\n')
                    replace_line("Preferences.txt", 16, str(TR) + '\n')
                    replace_line("Preferences.txt", 17, str(SB) + '\n')
                    replace_line("Preferences.txt", 18, str(ML) + '\n')
                    replace_line("Preferences.txt", 19, str(EL) + '\n')
                    replace_line("Preferences.txt", 20, str(mL) + '\n')
            except:
                messagebox.showwarning("Warning",
                                       "Incorrect data-type detected. Please ensure you are using correct format for each MHTD variable.")
                self.F2vals()

        if val == 2:
            try:
                #               PASS THE VALUES THAT RELATE TO PAN-TOMPKIN!! WHEN THIS IS AVALIABLE
                Prediction_mode(2)
                if closewind == 1:
                    pref.withdraw()
                    replace_line("Preferences.txt", 3, str(2) + '\n')
            except:
                messagebox.showwarning("Warning",
                                       "Pan-tompkins unavaliable for this signal. Please report error to makers of RR-APET for further support.")
                self.F2vals()

        if val == 3:
            try:
                #               PASS THE VALUES THAT RELATE TO PAN-TOMPKIN!! WHEN THIS IS AVALIABLE
                Prediction_mode(3)
                if closewind == 1:
                    pref.withdraw()
                    replace_line("Preferences.txt", 3, str(3) + '\n')
            except:
                messagebox.showwarning("Warning",
                                       "K-means unavaliable for this signal. Please report error to makers of RR-APET for further support.")
                self.F2vals()

        if val == 4:
            try:
                #               PASS THE VALUES THAT RELATE TO PAN-TOMPKIN!! WHEN THIS IS AVALIABLE
                Prediction_mode(4)
                if closewind == 1:
                    pref.withdraw()
                    replace_line("Preferences.txt", 3, str(4) + '\n')
            except:
                messagebox.showwarning("Warning",
                                       "Own method did not work for prediction. Please check your code or contact makers of RR-APET for further support.")
                self.F2vals()

    def reset(self, method):

        file = open("Original_Preferences.txt", 'r')
        val = file.read().split()
        file.close()

        if method == 1:
            # Check if different!
            if ((self.thr.get() != val[16]) or (self.sbl.get() != val[16]) or (self.magL.get() != val[16]) or (
                    self.engL.get() != val[16]) or (self.minL.get() != val[16])):
                # Reset MHTD using OG READ IN
                replace_line("Preferences.txt", 16, str(val[16]) + '\n')
                replace_line("Preferences.txt", 17, str(val[17]) + '\n')
                replace_line("Preferences.txt", 18, str(val[18]) + '\n')
                replace_line("Preferences.txt", 19, str(val[19]) + '\n')
                replace_line("Preferences.txt", 20, str(val[20]) + '\n')
                self.repredict(1, 0)
                self.F2vals()

        if method == 2:  # PAN-TOMPKINS
            self.repredict(2, 0)
            self.F2vals()
            # Reset Pan-Tompkins using OG READ IN
        #            replace_line("Preferences.txt", 16, str(val[16]) + '\n')

        if method == 3:  # K-MEANS
            self.repredict(3, 0)
            self.F2vals()

        if method == 4:  # OWN METHOD
            self.repredict(4, 0)
            self.F2vals()

    def metric_op(self):
        global Outerframe
        global big_frame
        global test_txt
        global btn

        self.btn1.config(style='UserPref.TButton', takefocus=False)
        #        self.btn2.config(style='UserPref.TButton', takefocus=False)
        self.btn3.config(style='UserPref.TButton', takefocus=False)
        self.btn4.config(style='SelectUserPref.TButton', takefocus=False)
        self.btn5.config(style='UserPref.TButton', takefocus=False)

        if trybtn is not None:
            trybtn.destroy()

        if test_txt is not None:
            test_txt.destroy()

        if Outerframe is not None:
            Outerframe.destroy()
            btn.destroy()

        if resetbtn is not None:
            resetbtn.destroy()
            okbtn.destroy()

        Outerframe = Frame(big_frame, bg='white', borderwidth=10, highlightcolor='grey', highlightthickness=1,
                           highlightbackground='black')
        Outerframe.pack(side='left', fill='y')
        F = Frame(Outerframe, bg='white', borderwidth=0, highlightcolor='grey', highlightthickness=1,
                  highlightbackground='grey')
        F.pack(side='top', fill='both')
        F2 = Frame(Outerframe, bg='white')
        F2.pack(side='top', fill='both')

        Label2(F, text="Frequency-Domain", style='Header.TLabel').grid(row=0, column=0,
                                                                       columnspan=2)  # , font=Cust, bg='white'

        Label2(F, text="Welch PSD", style='SubHeader.TLabel').grid(row=1, column=0, sticky='w')
        Label2(F, text="Segment Length, L (%)", style='Text.TLabel').grid(row=2, column=0, sticky='w')
        Wel_M = Entry(F, width=10)
        Wel_M.grid(row=2, column=1)
        Wel_M.insert(0, self.pref[5])
        Label2(F, text="Overlap length, O (%)", style='Text.TLabel').grid(row=3, column=0, sticky='w')
        Wel_O = Entry(F, width=10)
        Wel_O.grid(row=3, column=1)
        Wel_O.insert(0, self.pref[6])

        Label2(F, text="Blackman-Tukey PSD", style='SubHeader.TLabel').grid(row=4, column=0, sticky='w')
        Label2(F, text="N-bins, where K=N/10", style='Text.TLabel').grid(row=5, column=0, sticky='w')
        BT = Entry(F, width=10)
        BT.grid(row=5, column=1)
        BT.insert(0, self.pref[7])

        Label2(F, text="Lombscargle PSD", style='SubHeader.TLabel').grid(row=6, column=0, sticky='w')
        Label2(F, text="Omega max", style='Text.TLabel').grid(row=7, column=0, sticky='w')
        LS = Entry(F, width=10)
        LS.grid(row=7, column=1)
        LS.insert(0, self.pref[8])

        Label2(F, text="Auto Regression", style='SubHeader.TLabel').grid(row=8, column=0, sticky='w')
        Label2(F, text="Order", style='Text.TLabel').grid(row=9, column=0, sticky='w')
        AR = Entry(F, width=10)
        AR.grid(row=9, column=1)
        AR.insert(0, self.pref[9])

        Label(F, text="", bg='white', font=(Preferences[0], 6)).grid(row=11, column=0, columnspan=2)

        Label2(F, text="Nonlinear", style='Header.TLabel').grid(row=12, column=0, columnspan=2)

        Label2(F, text="Detrended Fluctuation Analysis", style='SubHeader.TLabel').grid(row=13, column=0, sticky='w')
        Label2(F, text="Minimum box length", style='Text.TLabel').grid(row=14, column=0, sticky='w')
        DFA1 = Entry(F, width=10)
        DFA1.grid(row=14, column=1)
        DFA1.insert(0, self.pref[10])
        Label2(F, text="Crosover point", style='Text.TLabel').grid(row=15, column=0, sticky='w')
        DFA2 = Entry(F, width=10)
        DFA2.grid(row=15, column=1)
        DFA2.insert(0, self.pref[11])
        Label2(F, text="Maximum box length", style='Text.TLabel').grid(row=16, column=0, sticky='w')
        DFA3 = Entry(F, width=10)
        DFA3.grid(row=16, column=1)
        DFA3.insert(0, self.pref[12])
        Label2(F, text="Step-size", style='Text.TLabel').grid(row=17, column=0, sticky='w')
        DFA4 = Entry(F, width=10)
        DFA4.grid(row=17, column=1)
        DFA4.insert(0, self.pref[13])

        Label2(F, text="Recurrence Quantification Analysis", style='SubHeader.TLabel').grid(row=18, column=0,
                                                                                            sticky='w')
        Label2(F, text="Embedding dimension, M", style='Text.TLabel').grid(row=19, column=0, sticky='w')
        RQA1 = Entry(F, width=10)
        RQA1.grid(row=19, column=1)
        RQA1.insert(0, Preferences[14])
        Label2(F, text="Lag, L", style='Text.TLabel').grid(row=20, column=0, sticky='w')
        RQA2 = Entry(F, width=10)
        RQA2.grid(row=20, column=1)
        RQA2.insert(0, self.pref[15])

        btn = Button(self.parent, text="Ok",
                     command=lambda: self.update_mets(Wel_M.get(), Wel_O.get(), BT.get(), LS.get(), AR.get(),
                                                      DFA1.get(), DFA2.get(), DFA3.get(), DFA4.get(), RQA1.get(),
                                                      RQA2.get()), font=cust_text)
        btn.pack(side='right', anchor='e')

    def text_analysis_op(self):
        self.btn1.config(style='UserPref.TButton', takefocus=False)
        #        self.btn2.config(style='UserPref.TButton', takefocus=False)
        self.btn3.config(style='UserPref.TButton', takefocus=False)
        self.btn4.config(style='UserPref.TButton', takefocus=False)
        self.btn5.config(style='SelectUserPref.TButton', takefocus=False)

        global Outerframe
        global big_frame
        global test_txt
        global btn
        global entry

        file = open("Preferences.txt", 'r')
        P = file.read().split()
        file.close()

        #        ECG_pref_on =

        if resetbtn is not None:
            resetbtn.destroy()
            okbtn.destroy()

        if Outerframe is not None:
            Outerframe.destroy()
            btn.destroy()

        if test_txt is not None:
            test_txt.destroy()

        if trybtn is not None:
            trybtn.destroy()

        Outerframe = Frame(big_frame, bg='white', borderwidth=10, highlightcolor='grey', highlightthickness=1,
                           highlightbackground='black')
        Outerframe.pack(side='left', fill='y')
        F = Frame(Outerframe, bg='white', borderwidth=0, highlightcolor='grey', highlightthickness=1,
                  highlightbackground='grey')
        F.pack(side='left')
        F2 = Frame(Outerframe, bg='white')
        F2.pack(side='left')

        Label(F, text="General Import Settings ", font=cust_header, bg='white').grid(row=0, column=0, columnspan=2)

        Label(F, text="Import data type: ", font=cust_text, bg='white').grid(row=2, column=0, sticky='w')
        self.dtype = StringVar(F)
        options = ['RRI', 'ECG']
        dtypemenu = OptionMenu(F, self.dtype, options[int(P[21])], *options, style='Text.TMenubutton')
        dtypemenu.grid(row=2, column=1, sticky='w')

        Label(F, text="Data Units: ", font=cust_text, bg='white').grid(row=3, column=0, sticky='w')
        self.dunits = StringVar(F)
        options = ['mV', 'V']
        dunitsmenu = OptionMenu(F, self.dunits, options[int(P[22])], *options, style='Text.TMenubutton')
        dunitsmenu.grid(row=3, column=1, sticky='w')

        Label(F, text="Time Units: ", font=cust_text, bg='white').grid(row=4, column=0, sticky='w')
        self.tunits = StringVar(F)
        options = ['s', 'ms']
        tunitsmenu = OptionMenu(F, self.tunits, options[int(P[23])], *options, style='Text.TMenubutton')
        tunitsmenu.grid(row=4, column=1, sticky='w')

        Label(F, text=" ", font=cust_text, bg='white').grid(row=5, column=0, columnspan=2)

        Label(F, text="Custom Text-File Settings ", font=cust_header, bg='white').grid(row=6, column=0, columnspan=2)

        Label(F, text="Column Seperator: ", font=cust_text, bg='white').grid(row=7, column=0, sticky='w')
        self.separator = StringVar(F)
        options = ['Space/Tab', 'Colon (:)', 'Semi-colon (;)']
        separatormenu = OptionMenu(F, self.separator, options[int(P[24])], *options,
                                   style='Text.TMenubutton')  # options[0] can be updated in preferences
        separatormenu.grid(row=7, column=1, sticky='w')

        btn = Button(self.parent, text="Ok",
                     command=lambda: self.update_cust_text(self.dtype.get(), self.dunits.get(), self.tunits.get(),
                                                           self.separator.get()), font=cust_text)
        btn.pack(side='right', anchor='e')

    # command=lambda:self.updateprec(but_wtd))

    def update_cust_text(self, dt, du, tu, sep):
        pref.withdraw()

        if dt == 'ECG':
            replace_line("Preferences.txt", 21, str(1) + '\n')
        elif dt == 'RRI':
            replace_line("Preferences.txt", 21, str(0) + '\n')

        if du == 'mV':
            replace_line("Preferences.txt", 22, str(0) + '\n')
        elif du == 'V':
            replace_line("Preferences.txt", 22, str(1) + '\n')

        if tu == 's':
            replace_line("Preferences.txt", 23, str(0) + '\n')
        elif tu == 'ms':
            replace_line("Preferences.txt", 23, str(1) + '\n')

        if sep == 'Space/Tab':
            replace_line("Preferences.txt", 24, str(0) + '\n')
        elif sep == 'Colon (:)':
            replace_line("Preferences.txt", 24, str(1) + '\n')
        elif sep == 'Semi-colon (;)':
            replace_line("Preferences.txt", 24, str(2) + '\n')

        #            replace_line("Preferences.txt", 11, str(copbox_temp) + '\n')

    #            replace_line("Preferences.txt", 12, str(maxbox_temp) + '\n')
    #            replace_line("Preferences.txt", 13, str(increm_temp) + '\n')

    def update_mets(self, welm, welo, bt, ls, ar, dfa1, dfa2, dfa3, dfa4, rqa1, rqa2):
        pref.withdraw()
        try:
            val = int(welm)
            replace_line("Preferences.txt", 5, str(val) + '\n')
        except:
            messagebox.showwarning("Warning",
                                   "Welch PSD segment length must be an integer. \n\nPlease note: Value NOT updated.")

        try:
            val = int(welo)
            replace_line("Preferences.txt", 6, str(val) + '\n')
        except:
            messagebox.showwarning("Warning",
                                   "Welch PSD overlap length must be an integer. \n\nPlease note: Value NOT updated.")

        try:
            val = int(bt)
            replace_line("Preferences.txt", 7, str(val) + '\n')
        except:
            messagebox.showwarning("Warning",
                                   "Blackman-Tukey PSD N-bins value must be an integer. \n\nPlease note: Value NOT updated.")

        try:
            val = int(ls)
            replace_line("Preferences.txt", 8, str(val) + '\n')
        except:
            messagebox.showwarning("Warning",
                                   "Lombscargle Omega max value must be an integer. \n\nPlease note: Value NOT updated.")

        try:
            val = int(ar)
            replace_line("Preferences.txt", 9, str(val) + '\n')
        except:
            messagebox.showwarning("Warning",
                                   "Auto-Regression Order value must be an integer. \n\nPlease note: Value NOT updated.")

        try:
            minbox_temp = int(dfa1)
            copbox_temp = int(dfa2)
            maxbox_temp = int(dfa3)
            increm_temp = int(dfa4)

            if ((minbox_temp < copbox_temp) & (copbox_temp < maxbox_temp) & (
                    increm_temp < ((copbox_temp - minbox_temp) / 2)) & (
                    increm_temp < ((maxbox_temp - copbox_temp) / 2))):
                replace_line("Preferences.txt", 10, str(minbox_temp) + '\n')
                replace_line("Preferences.txt", 11, str(copbox_temp) + '\n')
                replace_line("Preferences.txt", 12, str(maxbox_temp) + '\n')
                replace_line("Preferences.txt", 13, str(increm_temp) + '\n')
            else:
                if ((minbox_temp > copbox_temp) or (copbox_temp > maxbox_temp)):
                    messagebox.showwarning("Warning",
                                           "The values you have selected are incompatiable.\n\nThe minimum box length must be less than the crossover point and the crossover point must be less than the maximum box length.")
                else:
                    messagebox.showwarning("Warning",
                                           "The values you have selected are incompatiable.\n\nThere must be at least two points per gradient. Hint: Try lowering the step size value.")
        except:
            messagebox.showwarning("Warning",
                                   "All DFA parameters value must be integers. \n\nPlease note: Values NOT updated.")

        try:
            val = int(rqa1)
            replace_line("Preferences.txt", 14, str(val) + '\n')
        except:
            messagebox.showwarning("Warning",
                                   "RQA embedding dimension parameter must be an integer. \n\nPlease note: Value NOT updated.")

        try:
            val = int(rqa2)
            replace_line("Preferences.txt", 15, str(val) + '\n')
        except:
            messagebox.showwarning("Warning",
                                   "RQA lag parameter must be an integer. \n\nPlease note: Value NOT updated.")

    def fakeCommand(self):
        print('Under-Construction')

    def update_font(self, trial, new_font, new_size):

        global test_txt
        global root
        if new_font != '':
            replace_line("Preferences.txt", 0, new_font + '\n')
        if new_size != '':
            replace_line("Preferences.txt", 1, new_size + '\n')

        file = open("Preferences.txt", 'r')
        Preferences = file.read().split()

        self.new_font = font.Font(family=Preferences[0], size=int(Preferences[1]))

        test_txt.destroy()
        test_txt = Label(self.parent, text="Sample Text", font=self.new_font)
        test_txt.pack()

        if trial == 0:
            pref.withdraw()
            headerStyles()


# ~~~~~~~~~~~~~~ WINDOWS - HRV analysis WINDOW ~~~~~~~~~~~~~~~~~~~#
class HRVstatics(Frame):

    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.initUI_mets()

    def initUI_mets(self):

        self.parent.title("Algorithm and HRV Metrics")
        self.parent.resizable(width=FALSE, height=FALSE)
        self.parent.configure(highlightthickness=1, highlightbackground='grey')
        self.Stats()

    def Stats(self):
        global R_t
        global True_R_t
        global loaded_ann
        global labelled_flag
        global stats
        global E1
        global tt
        global Fs
        global figure1
        global frequency_figure
        global plot_pred
        global plot_view_fig
        global True_R_amp
        global R_amp
        self.but_wtd = but_wtd = 20

        if (len(R_t) <= 1 & warnings_on):
            messagebox.showwarning("Warning",
                                   "Cannot calculate HRV metrics \n\nPlease note: Annotations must be present for HRV metrics to be calculated.")
        else:

            if ECG_pref_on:
                # Removes any accidental double-ups created during editing and sets metrics to be calculated based on which plot is present
                if plot_pred == 1:
                    R_t = np.reshape(R_t, [np.size(R_t), ])
                    R_amp = np.reshape(R_amp, [np.size(R_amp), ])
                    temp = np.diff([R_t])
                    temp = np.append(temp, 1)
                    Rpeakss = R_t[temp != 0]
                    R_amp = R_amp[temp != 0]
                    R_t = np.reshape(Rpeakss, [np.size(Rpeakss), 1])
                    R_amp = np.reshape(R_amp, [np.size(R_amp), 1])

                else:
                    True_R_t = np.reshape(True_R_t, [np.size(True_R_t), ])
                    True_R_amp = np.reshape(True_R_amp, [np.size(True_R_amp), ])
                    temp = np.diff([True_R_t])
                    temp = np.append(temp, 1)
                    Rpeakss = True_R_t[temp != 0]
                    True_R_amp = True_R_amp[temp != 0]
                    True_R_t = np.reshape(Rpeakss, [np.size(Rpeakss), 1])
                    True_R_amp = np.reshape(True_R_amp, [np.size(True_R_amp), 1])

                # REMOVE any double-ups
                draw1()

            else:
                R_t = np.reshape(R_t, [np.size(R_t), ])
                temp = np.diff([R_t])
                temp = np.append(temp, 1)
                Rpeakss = R_t[temp != 0]
                R_t = np.reshape(Rpeakss, [np.size(Rpeakss), 1])

            file = open("Preferences.txt", 'r')
            Preferences = file.read().split()
            file.close()
            self.welchL = float(Preferences[5])
            self.welchO = float(Preferences[6])
            self.btval_input = int(Preferences[7])  # 10
            self.omax_input = int(Preferences[8])  # 500
            self.order = int(Preferences[9])  # 10
            # Time-domain Statistics
            SDNN, SDANN, MeanRR, RMSSD, pNN50 = Calculate_Features(Rpeakss, Fs)

            # Frequency-domain Statistics
            self.Rpeak_input = Rpeakss / Fs
            powVLF, powLF, powHF, perpowVLF, perpowLF, perpowHF, peak_freq_VLF, peak_freq_LF, peak_freq_HF, LF_HF_ratio = Freq_Analysis(
                self.Rpeak_input, meth=1, decim=3, m=self.welchL, o=self.welchO, bt_val=self.btval_input,
                omega_max=self.omax_input, order=self.order)

            # Nonlinear statistics
            RRI = np.diff(self.Rpeak_input)

            REC, DET, LAM, Lmean, Lmax, Vmean, Vmax = RQA_plot(RRI)
            SD1, SD2, c1, c2 = Poincare(RRI)
            alp1, alp2, F = DFA(RRI)

            data_frame = Frame(master=self.parent)
            data_frame.pack()

            time_dat = Frame(master=data_frame)
            time_dat.pack(side='left', anchor='n')

            self.Outer_freq_dat = Frame(master=data_frame)
            self.Outer_freq_dat.pack(side='left', anchor='n')

            freq_dat_up = Frame(self.Outer_freq_dat)
            freq_dat_up.pack(side='top')

            self.freq_dat_low = Frame(self.Outer_freq_dat)
            self.freq_dat_low.pack(side='top')

            non_dat = Frame(master=data_frame)
            non_dat.pack(side='left', anchor='n')

            self.Outer_DA_dat = Frame(master=data_frame)
            self.Outer_DA_dat.pack(side='left', anchor='n')

            self.DA_dat = Frame(master=self.Outer_DA_dat)
            self.DA_dat.pack(side='top', anchor='n')

            # TIME-DOMAIN Parameters
            Label(time_dat, text="Time-Domain Parameters", anchor=TKc.W, font=cust_text).grid(row=0, column=0,
                                                                                              columnspan=2)
            Label(time_dat, text="SDNN (ms)", anchor=TKc.W, width=but_wtd).grid(row=1, column=0)
            Label(time_dat, text=SDNN, anchor=TKc.W, width=int(but_wtd / 2)).grid(row=1, column=1)
            Label(time_dat, text="SDANN (ms)", anchor=TKc.W, width=but_wtd).grid(row=2, column=0)
            Label(time_dat, text=SDANN, anchor=TKc.W, width=int(but_wtd / 2)).grid(row=2, column=1)
            Label(time_dat, text="Mean RR interval (ms)", anchor=TKc.W, width=but_wtd).grid(row=3, column=0)
            Label(time_dat, text=MeanRR, anchor=TKc.W, width=int(but_wtd / 2)).grid(row=3, column=1)
            Label(time_dat, text="RMSSD (ms)", anchor=TKc.W, width=but_wtd).grid(row=4, column=0)
            Label(time_dat, text=RMSSD, anchor=TKc.W, width=int(but_wtd / 2)).grid(row=4, column=1)
            Label(time_dat, text="pNN50 (%)", anchor=TKc.W, width=but_wtd).grid(row=5, column=0)
            Label(time_dat, text=pNN50, anchor=TKc.W, width=int(but_wtd / 2)).grid(row=5, column=1)
            Label(time_dat, text="", anchor=TKc.W, width=int(but_wtd / 4)).grid(row=5, column=2)  # SPACER

            # FREQUENCY-DOMAIN Parameters

            Label(freq_dat_up, text="Frequency-Domain Parameters", anchor=TKc.W, font=cust_text).pack(side='top',
                                                                                                      anchor='center')

            # MENU FOR CHOICE OF ANALYSIS     title_list =
            self.frqmeth = StringVar(freq_dat_up)
            options = ['Welch', 'Blackman-Tukey', 'LombScargle', 'Auto Regression']
            RRImenu = OptionMenu(freq_dat_up, self.frqmeth, options[0], *options)
            RRImenu.config(width=16)
            #        RRImenu.configure(compound='right',image=self.photo)
            RRImenu.pack(side='top')

            self.frqmeth.trace('w', self.change_dropdown_HRV)
            self.updatefreqstats(but_wtd, method=1)

            # NONLINEAR Parameters
            Label(non_dat, text="Nonlinear Parameters", anchor=TKc.W, font=cust_text).grid(row=0, column=0,
                                                                                           columnspan=4)
            Label(non_dat, text="Recurrence Analysis", anchor=TKc.W, width=but_wtd, font='Helvetica 10 bold').grid(
                row=1, column=0, columnspan=2)
            Label(non_dat, text="REC (%)", anchor=TKc.W, width=int(but_wtd / 2)).grid(row=2, column=0)
            Label(non_dat, text=REC, anchor=TKc.W, width=int(but_wtd / 2)).grid(row=2, column=1)
            Label(non_dat, text="DET (%)", anchor=TKc.W, width=int(but_wtd / 2)).grid(row=3, column=0)
            Label(non_dat, text=DET, anchor=TKc.W, width=int(but_wtd / 2)).grid(row=3, column=1)
            Label(non_dat, text="LAM (%)", anchor=TKc.W, width=int(but_wtd / 2)).grid(row=4, column=0)
            Label(non_dat, text=LAM, anchor=TKc.W, width=int(but_wtd / 2)).grid(row=4, column=1)
            Label(non_dat, text="Lmean (bts)", anchor=TKc.W, width=int(but_wtd / 2)).grid(row=5, column=0)
            Label(non_dat, text=Lmean, anchor=TKc.W, width=int(but_wtd / 2)).grid(row=5, column=1)
            Label(non_dat, text="Lmax (bts)", anchor=TKc.W, width=int(but_wtd / 2)).grid(row=6, column=0)
            Label(non_dat, text=Lmax, anchor=TKc.W, width=int(but_wtd / 2)).grid(row=6, column=1)
            Label(non_dat, text="Vmean (bts)", anchor=TKc.W, width=int(but_wtd / 2)).grid(row=7, column=0)
            Label(non_dat, text=Vmean, anchor=TKc.W, width=int(but_wtd / 2)).grid(row=7, column=1)
            Label(non_dat, text="Vmax (bts)", anchor=TKc.W, width=int(but_wtd / 2)).grid(row=8, column=0)
            Label(non_dat, text=Vmax, anchor=TKc.W, width=int(but_wtd / 2)).grid(row=8, column=1)

            Label(non_dat, text="Poincare Analysis", anchor=TKc.W, width=but_wtd, font='Helvetica 10 bold').grid(row=1,
                                                                                                                 column=2,
                                                                                                                 columnspan=2)
            Label(non_dat, text="SD1 (ms)", anchor=TKc.W, width=int(but_wtd / 2)).grid(row=2, column=2)
            Label(non_dat, text=SD1, anchor=TKc.W, width=int(but_wtd / 2)).grid(row=2, column=3)
            Label(non_dat, text="SD2 (ms)", anchor=TKc.W, width=int(but_wtd / 2)).grid(row=3, column=2)
            Label(non_dat, text=SD2, anchor=TKc.W, width=int(but_wtd / 2)).grid(row=3, column=3)

            Label(non_dat, text="DFA", anchor=TKc.W, width=but_wtd, font='Helvetica 10 bold').grid(row=5, column=2,
                                                                                                   columnspan=2)
            Label(non_dat, text="\u03B1 1", anchor=TKc.W, width=int(but_wtd / 2)).grid(row=6, column=2)
            Label(non_dat, text=alp1, anchor=TKc.W, width=int(but_wtd / 2)).grid(row=6, column=3)
            Label(non_dat, text="\u03B1 2", anchor=TKc.W, width=int(but_wtd / 2)).grid(row=7, column=2)
            Label(non_dat, text=alp2, anchor=TKc.W, width=int(but_wtd / 2)).grid(row=7, column=3)

            if loaded_ann == 1:
                self.prec = None
                TP, FP, FN = test2(R_t, True_R_t, tt)
                Se, PP, ACC, DER = acc2(TP, FP, FN)
                Label(self.DA_dat, text="Detection Algorithm Metrics", anchor=TKc.W, font=cust_text).grid(row=3,
                                                                                                          column=13,
                                                                                                          columnspan=4)
                Label(self.DA_dat, text="Sensitivity (%)", anchor=TKc.W, width=but_wtd).grid(row=4, column=13)
                Label(self.DA_dat, text=Se, anchor=TKc.W, width=int(but_wtd / 2)).grid(row=4, column=14)
                Label(self.DA_dat, text="Positive Predictability (%)", anchor=TKc.W, width=but_wtd).grid(row=5,
                                                                                                         column=13)
                Label(self.DA_dat, text=PP, anchor=TKc.W, width=int(but_wtd / 2)).grid(row=5, column=14)
                Label(self.DA_dat, text="Accuracy (%)", anchor=TKc.W, width=but_wtd).grid(row=6, column=13)
                Label(self.DA_dat, text=ACC, anchor=TKc.W, width=int(but_wtd / 2)).grid(row=6, column=14)
                Label(self.DA_dat, text="Detection Error Rate (%)", anchor=TKc.W, width=but_wtd).grid(row=7, column=13)
                Label(self.DA_dat, text=DER, anchor=TKc.W, width=int(but_wtd / 2)).grid(row=7, column=14)
                Label(self.DA_dat, text="Precision Window (ms)", anchor=TKc.W).grid(row=8, column=13)
                self.prec = Entry(self.DA_dat, width=int(but_wtd / 2))
                self.prec.grid(row=8, column=14)
                time = tt / Fs * 1000
                self.prec.insert(0, '{:.2f}'.format(time))
                Button(self.DA_dat, text="Update", anchor=TKc.W, width=int(but_wtd / 2),
                       command=lambda: self.updateprec(but_wtd)).grid(row=8, column=15)

            Button(self.parent, text="Save", width=int(but_wtd / 2), height=2, command=savemetrics,
                   font='Helvetica 12 bold').pack(side='bottom', anchor='e')

            open_plot()

    def updateprec(self, but_wtd):
        global R_t
        global True_R_t
        global tt
        global Fs

        tt = round(float(self.prec.get()) * Fs / 1000)
        time = tt / Fs * 1000
        TP, FP, FN = test2(R_t, True_R_t, tt)
        Se, PP, ACC, DER = acc2(TP, FP, FN)

        self.DA_dat.destroy()
        self.DA_dat = Frame(master=self.Outer_DA_dat)
        self.DA_dat.pack(side='top')

        # Detection Algorithm Metrics
        Label(self.DA_dat, text="Detection Algorithm Metrics", anchor=TKc.W, font=cust_text).grid(row=3, column=13,
                                                                                                  columnspan=4)
        Label(self.DA_dat, text="Sensitivity (%)", anchor=TKc.W, width=but_wtd).grid(row=4, column=13)
        Label(self.DA_dat, text=Se, anchor=TKc.W, width=int(but_wtd / 2)).grid(row=4, column=14)
        Label(self.DA_dat, text="Positive Predictability (%)", anchor=TKc.W, width=but_wtd).grid(row=5, column=13)
        Label(self.DA_dat, text=PP, anchor=TKc.W, width=int(but_wtd / 2)).grid(row=5, column=14)
        Label(self.DA_dat, text="Accuracy (%)", anchor=TKc.W, width=but_wtd).grid(row=6, column=13)
        Label(self.DA_dat, text=ACC, anchor=TKc.W, width=int(but_wtd / 2)).grid(row=6, column=14)
        Label(self.DA_dat, text="Detection Error Rate (%)", anchor=TKc.W, width=but_wtd).grid(row=7, column=13)
        Label(self.DA_dat, text=DER, anchor=TKc.W, width=int(but_wtd / 2)).grid(row=7, column=14)
        Label(self.DA_dat, text="Precision Window (ms)", anchor=TKc.W).grid(row=8, column=13)
        self.prec = Entry(self.DA_dat, width=int(but_wtd / 2))
        self.prec.grid(row=8, column=14)
        self.prec.insert(0, '{:.2f}'.format(time))
        Button(self.DA_dat, text="Update", anchor=TKc.W, width=int(but_wtd / 2),
               command=lambda: self.updateprec(but_wtd)).grid(row=8, column=15)

    def change_dropdown_HRV(self, *args):
        methods = self.frqmeth.get()

        if (methods == 'Welch'):
            METH = 1
        elif (methods == 'Blackman-Tukey'):
            METH = 2
        elif (methods == 'LombScargle'):
            METH = 3
        else:
            METH = 4

        self.updatefreqstats(but_wtd=20, method=METH)

    def updatefreqstats(self, but_wtd, method):
        self.freq_dat_low.destroy()
        self.freq_dat_low = Frame(master=self.Outer_freq_dat)
        self.freq_dat_low.pack(side='top')
        # GET METHOD FROM DROP-DOWN MENU INPUT:
        powVLF, powLF, powHF, perpowVLF, perpowLF, perpowHF, peak_freq_VLF, peak_freq_LF, peak_freq_HF, LF_HF_ratio = Freq_Analysis(
            self.Rpeak_input, meth=1, decim=3, m=self.welchL, o=self.welchO, bt_val=self.btval_input,
            omega_max=self.omax_input, order=self.order)
        Label(self.freq_dat_low, text="Peak Frequency", anchor=TKc.W, width=int(but_wtd / 4 * 3)).grid(row=1, column=0)
        Label(self.freq_dat_low, text="VLF (Hz)", anchor=TKc.W, width=int(but_wtd / 2)).grid(row=1, column=1)
        Label(self.freq_dat_low, text=peak_freq_VLF, anchor=TKc.W, width=int(but_wtd / 2)).grid(row=2, column=1)
        Label(self.freq_dat_low, text="LF (Hz)", anchor=TKc.W, width=int(but_wtd / 2)).grid(row=1, column=2)
        Label(self.freq_dat_low, text=peak_freq_LF, anchor=TKc.W, width=int(but_wtd / 2)).grid(row=2, column=2)
        Label(self.freq_dat_low, text="HF (Hz)", anchor=TKc.W, width=int(but_wtd / 2)).grid(row=1, column=3)
        Label(self.freq_dat_low, text=peak_freq_HF, anchor=TKc.W, width=int(but_wtd / 2)).grid(row=2, column=3)
        Label(self.freq_dat_low, text="Percentage Power", anchor=TKc.W, width=int(but_wtd / 4 * 3)).grid(row=3,
                                                                                                         column=0)
        Label(self.freq_dat_low, text="VLF (%)", anchor=TKc.W, width=int(but_wtd / 2)).grid(row=3, column=1)
        Label(self.freq_dat_low, text=perpowVLF, anchor=TKc.W, width=int(but_wtd / 2)).grid(row=4, column=1)
        Label(self.freq_dat_low, text="LF (%)", anchor=TKc.W, width=int(but_wtd / 2)).grid(row=3, column=2)
        Label(self.freq_dat_low, text=perpowLF, anchor=TKc.W, width=int(but_wtd / 2)).grid(row=4, column=2)
        Label(self.freq_dat_low, text="HF (%)", anchor=TKc.W, width=int(but_wtd / 2)).grid(row=3, column=3)
        Label(self.freq_dat_low, text=perpowHF, anchor=TKc.W, width=int(but_wtd / 2)).grid(row=4, column=3)
        Label(self.freq_dat_low, text="Absolute Power", anchor=TKc.W, width=int(but_wtd / 4 * 3)).grid(row=5, column=0)
        Label(self.freq_dat_low, text="VLF (ms^2)", anchor=TKc.W, width=int(but_wtd / 2)).grid(row=5, column=1)
        Label(self.freq_dat_low, text=powVLF, anchor=TKc.W, width=int(but_wtd / 2)).grid(row=6, column=1)
        Label(self.freq_dat_low, text="LF (ms^2)", anchor=TKc.W, width=int(but_wtd / 2)).grid(row=5, column=2)
        Label(self.freq_dat_low, text=powLF, anchor=TKc.W, width=int(but_wtd / 2)).grid(row=6, column=2)
        Label(self.freq_dat_low, text="HF (ms^2)", anchor=TKc.W, width=int(but_wtd / 2)).grid(row=5, column=3)
        Label(self.freq_dat_low, text=powHF, anchor=TKc.W, width=int(but_wtd / 2)).grid(row=6, column=3)
        Label(self.freq_dat_low, text="LF/HF Ratio", anchor=TKc.W, width=int(but_wtd / 4 * 3)).grid(row=7, column=0)
        Label(self.freq_dat_low, text=LF_HF_ratio, anchor=TKc.W, width=int(but_wtd / 2)).grid(row=7, column=1)
        Label(self.freq_dat_low, text="", anchor=TKc.W, width=int(but_wtd / 4)).grid(row=5, column=4)


def close_plot():
    plot_wind.withdraw()


class plot_viewer(Frame):

    def __init__(self, parent):
        global Preferences
        Frame.__init__(self, parent)
        self.welch_int_M = None
        self.results_frame = None
        self.Slider_housing = None
        self.welch_int_M = float(Preferences[5])
        self.welch_int_O = int(Preferences[6])
        self.btval_input = int(Preferences[7])
        self.omax_input = int(Preferences[8])
        self.order = int(Preferences[9])
        self.m_input = self.welch_int_M  # int(self.sig_len*self.welch_int_M/100)
        self.o_input = self.welch_int_O  # int(self.m_input*self.welch_int_O/100)

        # INITIAL VALUES FOR DFA PLOT
        self.minbox = int(Preferences[10])  # 1
        self.copbox = int(Preferences[11])  # 15
        self.maxbox = int(Preferences[12])  # 64
        self.increm = int(Preferences[13])  # 1

        # INITIAL VALUES FOR RQA PLOT
        self.M = int(Preferences[14])  # 10
        self.L = int(Preferences[15])  # 1
        self.parent = parent
        self.initUI_plotting()

    def initUI_plotting(self):
        global freq_wind
        global frequency_figure
        global RQA_figure
        global R_t
        global Fs
        global graphCanvas2
        global draw_figure
        global pvp

        self.sig_len = len(R_t)
        self.parent.title("Plot Viewer")
        self.parent.configure(highlightthickness=1, highlightbackground='black')
        T1 = Frame(self.parent)
        T1.pack(side='left', fill=BOTH, expand=True)
        picture_frame = Frame(T1)
        picture_frame.pack(side='top', anchor='n', fill=BOTH, expand=True)

        T2 = Frame(self.parent, bg='white smoke')
        T2.pack(side='left', fill=BOTH, expand=False)
        self.results_frame = Frame(T2, bg='white smoke')
        self.results_frame.pack(side='top', fill=BOTH, expand=True)
        self.Slider_housing = Frame(T2, bg='white smoke')
        self.Slider_housing.pack(side='top', fill=BOTH, expand=False)

        #        self.buttonhouse = Frame(T2, bg='white smoke')
        #        self.buttonhouse.pack(side='top', fill=BOTH, expand=True)

        draw_figure = Figure(tight_layout=True)
        pvp = draw_figure.add_subplot(111)

        graphCanvas2 = FigureCanvasTkAgg(draw_figure, master=picture_frame)
        graphCanvas2.get_tk_widget().pack(side='top', fill=BOTH, expand=True)

        # ~~~~~~~~~~~~ Dropdown Menu for Prediction Mode ~~~~~~~~~~~~~~~~#

        # SET UP MENUBAR#
        menubar = Menu(self.parent, font=cust_text)
        self.parent.config(menu=menubar)

        fileMenu = Menu(menubar, font=cust_text, tearoff=False)

        fileMenu.add_command(label="Welch\'s Periodogram", command=lambda: self.reselect_graph(0))
        fileMenu.add_command(label="Blackman-Tukey\'s Periodogram", command=lambda: self.reselect_graph(1))
        fileMenu.add_command(label="Lomb-Scargle\'s Periodogram", command=lambda: self.reselect_graph(2))
        fileMenu.add_command(label="Autoregression Periodogram", command=lambda: self.reselect_graph(3))
        fileMenu.add_command(label="Poincare Plot", command=lambda: self.reselect_graph(4))
        fileMenu.add_command(label="DFA Plot", command=lambda: self.reselect_graph(5))
        fileMenu.add_command(label="RQA Plot", command=lambda: self.reselect_graph(6))
        fileMenu.add_command(label="Show all", command=lambda: self.reselect_graph(7))
        menubar.add_cascade(label="Select Plot", menu=fileMenu, font=cust_subheadernb)

        toolMenu = Menu(menubar, font=cust_text, tearoff=False)
        toolMenu.add_command(label="Save", command=savefigure)
        toolMenu.add_command(label="Quit", command=close_plot)
        menubar.add_cascade(label="Options", menu=toolMenu, font=cust_subheadernb)

        self.reselect_graph(0)

    def onRefresh(self, method):
        if (method > 0) & (method <= 4):
            if method == 1:
                self.m_input = self.M_slide.get()  # int(self.sig_len*self.M_slide.get()/100)
                self.o_input = self.O_slide.get()  # int(self.m_input*self.O_slide.get()/100)
            elif method == 2:
                self.btval_input = int(self.BT_slide.get())
            elif method == 3:
                self.omax_input = int(self.omeg_slide.get())
            elif method == 4:
                self.order = int(self.AR_slide.get())
            freq_plot(method, 3, pvp, m=self.m_input, o=self.o_input, btval=self.btval_input, omax=self.omax_input,
                      Ord=self.order)
            self.printParameters(method)
        else:
            if method == 5:
                messagebox.showwarning("Warning", "Feature not operational yet.")
            elif method == 6:
                minbox_temp = int(self.ent[0].get())
                copbox_temp = int(self.ent[1].get())
                maxbox_temp = int(self.ent[2].get())
                increm_temp = int(self.ent[3].get())

                if ((minbox_temp < copbox_temp) & (copbox_temp < maxbox_temp) & (
                        increm_temp < ((copbox_temp - minbox_temp) / 2)) & (
                        increm_temp < ((maxbox_temp - copbox_temp) / 2))):
                    self.minbox = int(self.ent[0].get())
                    self.copbox = int(self.ent[1].get())
                    self.maxbox = int(self.ent[2].get())
                    self.increm = int(self.ent[3].get())
                    DFA_plot(pvp, Min=self.minbox, Max=self.maxbox, Inc=self.increm, COP=self.copbox)
                    self.printParameters(method)
                else:
                    if (minbox_temp > copbox_temp) or (copbox_temp > maxbox_temp):
                        messagebox.showwarning("Warning", "The values you have selected are incompatiable.\n\n" +
                                               "The minimum box length must be less than the crossover point and the crossover point must be less than the maximum box length.")
                    else:
                        messagebox.showwarning("Warning", "The values you have selected are incompatiable.\n\n" +
                                               "There must be at least two points per gradient. Hint: Try lowering the step size value.")

            elif method == 7:
                self.M = int(self.M2_slide.get())
                self.L = int(self.L_slide.get())
                RQA_plott(pvp, Mval=self.M, Lval=self.L)
                self.printParameters(method)

    def printParameters(self, method):
        global graphCanvas2
        global draw_figure
        global showallwindow
        global pvp
        global R_t

        for widget in self.results_frame.winfo_children():
            widget.destroy()
        local_R_t = np.reshape(R_t, [np.size(R_t), ])
        local_RRI_freq = local_R_t / Fs
        local_RRI_nonlinear = np.diff(local_RRI_freq)
        if (method > 0) & (method <= 4):
            res = np.zeros(10)

            labels = ['Power (ms^2/Hz)', 'Power (%)', 'Peak frequency (Hz)']
            labels2 = ['VLF', 'LF', 'HF']
            Label(self.results_frame, text='Frequency Analysis', bg='white smoke', font=cust_subheader).grid(row=0,
                                                                                                             column=0,
                                                                                                             columnspan=4,
                                                                                                             sticky=TKc.E + TKc.W)
            Label(self.results_frame, text='Variable', bg='white smoke', anchor=TKc.W, borderwidth=1, relief='solid',
                  font=cust_text).grid(row=1, column=0, sticky=TKc.E + TKc.W)
            res[:] = Freq_Analysis(local_RRI_freq, meth=method, decim=3, m=self.m_input, o=self.o_input,
                                   bt_val=self.btval_input, omega_max=self.omax_input, order=self.order)
            plh = 0
            for itr in range(3):
                Label(self.results_frame, text=labels2[itr], anchor=TKc.W, bg='white smoke', borderwidth=1,
                      relief='solid', font=cust_text).grid(row=1, column=itr + 1, sticky=TKc.E + TKc.W)
                Label(self.results_frame, text=labels[itr], anchor=TKc.W, bg='white smoke', borderwidth=1,
                      relief='solid', font=cust_text).grid(row=itr + 2, column=0, sticky=TKc.E + TKc.W)
                for itr2 in range(3):
                    Label(self.results_frame, text=res[plh], anchor=TKc.E, bg='white smoke', borderwidth=1,
                          relief='solid', font=cust_text).grid(row=itr + 2, column=itr2 + 1, sticky=TKc.E + TKc.W)
                    plh = plh + 1
            Label(self.results_frame, text='', bg='white smoke', anchor=TKc.W, font=cust_text).grid(row=5, column=0,
                                                                                                    sticky=TKc.E + TKc.W)
            Label(self.results_frame, text='LF/HF ratio', bg='white smoke', anchor=TKc.W, font=cust_text).grid(row=6,
                                                                                                               column=0,
                                                                                                               sticky=TKc.E + TKc.W)
            Label(self.results_frame, text=res[9], bg='white smoke', anchor=TKc.E, font=cust_text).grid(row=6, column=1,
                                                                                                        sticky=TKc.E + TKc.W)

        elif method == 5:  # Poincare parameters
            sd1, sd2, c1, c2 = Poincare(local_RRI_nonlinear)
            Label(self.results_frame, text='Nonlinear Analysis', bg='white smoke', font=cust_subheader).grid(row=0,
                                                                                                             column=0,
                                                                                                             columnspan=4,
                                                                                                             sticky=TKc.E + TKc.W)
            labels = ['Variable', 'SD1 (ms)', 'SD2 (ms)']
            res = ['Value', sd1, sd2]

            for itr in range(3):
                Label(self.results_frame, text=labels[itr], anchor=TKc.W, bg='white smoke', borderwidth=1,
                      relief='solid', font=cust_text).grid(row=itr + 1, column=0, sticky=TKc.E + TKc.W)
                Label(self.results_frame, text=res[itr], anchor=TKc.E, bg='white smoke', borderwidth=1, relief='solid',
                      font=cust_text).grid(row=itr + 1, column=1, sticky=TKc.E + TKc.W)

        elif method == 6:  # DFA parameters
            alp1, alp2, F = DFA(local_RRI_nonlinear, min_box=self.minbox, max_box=self.maxbox, inc=self.increm,
                                cop=self.copbox)
            Label(self.results_frame, text='Nonlinear Analysis', bg='white smoke', font=cust_subheader).grid(row=0,
                                                                                                             column=0,
                                                                                                             columnspan=4,
                                                                                                             sticky=TKc.E + TKc.W)
            labels = ['Variable', 'alpha 1 (ms)', 'alpha 2 (ms)']
            res = ['Value', alp1, alp2]

            for itr in range(3):
                Label(self.results_frame, text=labels[itr], anchor=TKc.W, bg='white smoke', borderwidth=1,
                      relief='solid', font=cust_text).grid(row=itr + 1, column=0, sticky=TKc.E + TKc.W)
                Label(self.results_frame, text=res[itr], anchor=TKc.E, bg='white smoke', borderwidth=1, relief='solid',
                      font=cust_text).grid(row=itr + 1, column=1, sticky=TKc.E + TKc.W)

        elif method == 7:  # RQA parameters
            REC, DET, LAM, Lmean, Lmax, Vmean, Vmax = RQA_plot(local_RRI_nonlinear, m=self.M, l=self.L)
            Label(self.results_frame, text='Nonlinear Analysis', bg='white smoke', font=cust_subheader).grid(row=0,
                                                                                                             column=0,
                                                                                                             columnspan=4,
                                                                                                             sticky=TKc.E + TKc.W)
            labels = ['Variable', 'REC', 'DET', 'LAM', 'Lmean', 'Lmax', 'Vmean', 'Vmax']
            res = ['Value', REC, DET, LAM, Lmean, Lmax, Vmean, Vmax]
            for itr in range(8):
                Label(self.results_frame, text=labels[itr], anchor=TKc.W, bg='white smoke', borderwidth=1,
                      relief='solid', font=cust_text).grid(row=itr + 1, column=0, sticky=TKc.E + TKc.W)
                Label(self.results_frame, text=res[itr], anchor=TKc.E, bg='white smoke', borderwidth=1, relief='solid',
                      font=cust_text).grid(row=itr + 1, column=1, sticky=TKc.E + TKc.W)

    def multi_ent_box(self, frme, a):
        empty = Entry(master=frme, width=5)
        empty.grid(row=a, column=1)
        return empty

    def reselect_graph(self, sf):
        global graphCanvas2
        global draw_figure
        global showallwindow
        global pvp
        global R_t
        for widget in self.Slider_housing.winfo_children():
            widget.destroy()

        if (sf >= 0) & (sf < 7):
            mini_frame3 = Frame(self.Slider_housing, bg='white smoke')
            mini_frame3.pack(side='bottom', fill=BOTH)
            mini_frame2 = Frame(self.Slider_housing, bg='white smoke')
            mini_frame2.pack(side='bottom', fill=BOTH)
            mini_frame1 = Frame(self.Slider_housing, bg='white smoke')
            mini_frame1.pack(side='bottom', fill=BOTH)

            if sf < 4:
                freq_plot((sf + 1), 3, pvp, m=self.m_input, o=self.o_input, btval=self.btval_input,
                          omax=self.omax_input, Ord=self.order)
                if sf == 0:
                    Label(self.Slider_housing, text="Welch's Plotting Parameters", bg='white smoke',
                          font=cust_subheader).pack(side='top')
                    self.M_slide = Scale(mini_frame1, label='L (%)', from_=0.1, to=99.9, resolution=0.1,
                                         orient=TKc.HORIZONTAL, bg='white smoke', borderwidth=0, highlightthickness=0)
                    self.M_slide.pack(side='left', fill='x', expand=True)
                    self.M_slide.set(self.welch_int_M)
                    self.O_slide = Scale(mini_frame2, label='O (%)', from_=0, to=99, orient=TKc.HORIZONTAL,
                                         bg='white smoke', borderwidth=0, highlightthickness=0)
                    self.O_slide.pack(side='left', fill='x', expand=True)
                    self.O_slide.set(self.welch_int_O)

                elif sf == 1:
                    Label(self.Slider_housing, text="Blackman-Tukey's Plotting Parameters", bg='white smoke',
                          font=cust_subheader).pack(side='top')
                    self.BT_slide = Scale(mini_frame1, label='N-bins, where K=N/10', from_=1, to=30,
                                          orient=TKc.HORIZONTAL, bg='white smoke', borderwidth=0, highlightthickness=0)
                    self.BT_slide.pack(side='left', fill='x', expand=True)
                    self.BT_slide.set(self.btval_input)

                elif (sf == 2):
                    Label(self.Slider_housing, text="LombScargle's Plotting Parameters", bg='white smoke',
                          font=cust_subheader).pack(side='top')
                    self.omeg_slide = Scale(mini_frame1, label='Omega max', from_=100, to=1000, resolution=10,
                                            orient=TKc.HORIZONTAL, bg='white smoke', borderwidth=0,
                                            highlightthickness=0)
                    self.omeg_slide.pack(side='left', fill='x', expand=True)
                    self.omeg_slide.set(self.omax_input)

                elif sf == 3:
                    Label(self.Slider_housing, text="Autoregression Plotting Parameters", bg='white smoke',
                          font=cust_subheader).pack(side='top')
                    self.AR_slide = Scale(mini_frame1, label='Order', from_=1, to=200, orient=TKc.HORIZONTAL,
                                          bg='white smoke', borderwidth=0, highlightthickness=0)
                    self.AR_slide.pack(side='left', fill='x', expand=True)
                    self.AR_slide.set(self.order)

            elif sf == 4:
                Poincare_plot(pvp)

            elif sf == 5:
                DFA_plot(pvp, Min=self.minbox, Max=self.maxbox, Inc=self.increm, COP=self.copbox)
                Label(self.Slider_housing, text="DFA Parameters", bg='white smoke', font=cust_subheader).pack(
                    side='top')
                labels = ['Minimum: ', 'COP: ', 'Maximum: ', 'Step size: ']
                vals = [self.minbox, self.copbox, self.maxbox, self.increm]
                self.ent = [self.multi_ent_box(mini_frame1, idx) for idx in range(4)]
                for itr in range(4):
                    Label(mini_frame1, text=labels[itr], anchor=TKc.W, bg='white smoke').grid(row=itr, column=0,
                                                                                              sticky=TKc.E + TKc.W)
                    self.ent[itr].insert(0, vals[itr])

            elif sf == 6:
                RQA_plott(pvp, Mval=self.M, Lval=self.L)

                Label(self.Slider_housing, text="RQA Parameters", bg='white smoke', font=cust_subheader).pack(
                    side='top')
                self.M2_slide = Scale(mini_frame1, label='M ', from_=1, to=99, orient=TKc.HORIZONTAL, bg='white smoke',
                                      borderwidth=0, highlightthickness=0)
                self.M2_slide.pack(side='left', fill='x', expand=True)
                self.M2_slide.set(self.M)
                self.L_slide = Scale(mini_frame2, label='L ', from_=0, to=99, orient=TKc.HORIZONTAL, bg='white smoke',
                                     borderwidth=0, highlightthickness=0)
                self.L_slide.pack(side='left', fill='x', expand=True)
                self.L_slide.set(self.L)

            storebutton = Button(mini_frame3, text="Store settings", command=lambda: self.store_settings(sf + 1),
                                 font=cust_text)
            storebutton.pack(side='left', padx=10)
            Button(mini_frame3, text="Refresh", command=lambda: self.onRefresh(sf + 1), font=cust_text).pack(
                side='right')
            self.printParameters(sf + 1)

        elif sf == 7:
            showallwindow = Toplevel()
            showallwindow.title('All Plots')

            ####Following section is hard-coded: Upgrade to soft-code later###
            #        total = 7
            #        num_rows = int(np.ceil(total/3))
            #
            #        name = np.zeros(num_rows)
            #
            #    #Create Frames
            #        for intv in range():
            #            name[intv] = 'frame' + str(intv)
            #            print(name[intv]) = Frame(showallwindow)
            #            name[intv].pack(side='left',fill=BOTH, expand=1)
            frame1 = Frame(showallwindow)
            frame1.pack(side='top', fill=BOTH, expand=1)
            frame2 = Frame(showallwindow)
            frame2.pack(side='top', fill=BOTH, expand=1)
            frame3 = Frame(showallwindow)
            frame3.pack(side='top', fill=BOTH, expand=1)

            fig_holder2 = Figure(dpi=60)

            for count in range(9):
                fig_holder = Figure(dpi=60)
                figs = fig_holder.add_subplot(111)
                if count == 0:
                    freq_plot(1, 3, figs)
                elif count == 1:
                    freq_plot(2, 3, figs)
                elif count == 2:
                    freq_plot(3, 3, figs)
                elif count == 3:
                    freq_plot(4, 3, figs)
                elif count == 4:
                    Poincare_plot(figs)
                elif count == 5:
                    DFA_plot(figs)
                elif count == 6:
                    RQA_plott(figs)
                elif count == 7:
                    figs.clear()
                elif count == 8:
                    figs.clear()

                if count <= 6:
                    if count <= 2:
                        Canv = FigureCanvasTkAgg(fig_holder, master=frame1)
                    elif count <= 5:
                        Canv = FigureCanvasTkAgg(fig_holder, master=frame2)
                    else:
                        Canv = FigureCanvasTkAgg(fig_holder, master=frame3)

                else:
                    Canv = FigureCanvasTkAgg(fig_holder2, master=frame3)

                Canv.get_tk_widget().pack(side='left', fill=BOTH, expand=1)

        graphCanvas2.draw()

    def store_settings(self, sf):
        if sf == 1:
            replace_line("Preferences.txt", 5, str(self.M_slide.get()) + '\n')
            replace_line("Preferences.txt", 6, str(self.O_slide.get()) + '\n')
        elif sf == 2:
            replace_line("Preferences.txt", 7, str(self.BT_slide.get()) + '\n')
        elif sf == 3:
            replace_line("Preferences.txt", 8, str(self.omeg_slide.get()) + '\n')
        elif sf == 4:
            replace_line("Preferences.txt", 9, str(self.AR_slide.get()) + '\n')

        #        elif (sf == 5):


#            #INITIAL VALUES FOR DFA PLOT
#            self.minbox = int(Preferences[10])       #1
#            self.copbox = int(Preferences[11])       #15
#            self.maxbox = int(Preferences[12])       #64
#            self.increm = int(Preferences[13])       #1
#        elif (sf == 6):
#            #INITIAL VALUES FOR RQA PLOT
#            self.M = int(Preferences[14])            #10
#            self.L = int(Preferences[15])            #1

# ~~~~~~~~~~~~~~ WINDOWS - MAIN RR-APET PROGRAM ~~~~~~~~~~~~~~~~~~~#
class H5_selector(Frame, object):

    def __init__(self, parent, string):
        Frame.__init__(self, parent)

        self.parent = parent
        self.string = string[0]
        self.str2 = string[1]
        self.parent.title("HDF5/MAT-File Navigator")
        self.pack(fill='both', expand=1)
        #        self.parent.minsize(500,500)
        self.config(bg='white')
        self.init_h5_UI()

    def init_h5_UI(self):
        file = h5py.File(self.string, 'r')
        x = list(file.keys())
        N = len(x)

        big_frame = Frame(self.parent, bg='white smoke', borderwidth=20)
        big_frame.pack(side='top')
        co1_frame = Frame(big_frame, bg='white', borderwidth=10, highlightcolor='grey', highlightthickness=1,
                          highlightbackground='black')
        co1_frame.pack(side='left', fill='y')

        F = Frame(co1_frame)
        F.pack(side='left', fill='y')

        self.tree = Treeview(F)
        self.tree.column("#0", width=270, minwidth=270)
        self.tree.heading("#0", text="Name", anchor='w')
        # LEVEL ONE
        levels = [None] * 10
        new_keys = [None] * 10

        flag = 1

        for aa in range(N):
            flag = 1
            levels[0] = self.tree.insert("", "end", text=x[aa])
            try:
                new_keys[0] = list(file[x[aa]].keys())
            except:
                flag = 0
            if flag:
                for bb in range(len(new_keys[0])):
                    flag = 1
                    levels[1] = self.tree.insert(levels[0], "end", text=new_keys[0][bb], tags=(x[aa]))
                    try:
                        new_keys[1] = list(file[x[aa] + '/' + new_keys[0][bb]].keys())
                        if (str(type(new_keys[1][bb][0]))) != "<class 'str'>":
                            flag = 0
                    except:
                        flag = 0
                    if flag:
                        for cc in range(len(new_keys[1])):
                            flag = 1
                            levels[2] = self.tree.insert(levels[1], "end", text=new_keys[1][cc],
                                                         tags=(x[aa], new_keys[0][bb]))
                            try:
                                new_keys[2] = list(file[x[aa] + '/' + new_keys[0][bb] + '/' + new_keys[1][cc]].keys())
                                if (str(type(new_keys[2][cc][0]))) != "<class 'str'>":
                                    flag = 0
                            except:
                                flag = 0
                            if flag:
                                for dd in range(len(new_keys[2])):
                                    levels[3] = self.tree.insert(levels[2], "end", text=new_keys[2][dd],
                                                                 tags=(x[aa], new_keys[0][bb], new_keys[1][cc]))
        #
        self.tree.pack(side=TOP, fill='x')

        btn = Button(self.parent, text="Ok", command=self.selectItem)
        btn.pack(side='right', anchor='e')

    def selectItem(self):
        global True_R_t
        global True_R_amp
        global dat
        global x
        global xminn
        global loaded_ann
        global Fs
        global load_dat
        global tcol
        global R_t
        curItem = self.tree.focus()
        x2 = self.tree.item(curItem)
        upper_tiers = x2.get('tags')
        lower_tier = x2.get('text')

        if (len(upper_tiers) == 0):
            sub_fol = lower_tier
        elif (len(upper_tiers) == 1):
            sub_fol = str(upper_tiers[0]) + '/' + lower_tier
        elif (len(upper_tiers) == 2):
            sub_fol = str(upper_tiers[0]) + '/' + str(upper_tiers[1]) + '/' + lower_tier
        elif (len(upper_tiers) == 3):
            sub_fol = str(upper_tiers[0]) + '/' + str(upper_tiers[1]) + '/' + str(upper_tiers[2]) + '/' + lower_tier

        open_me = h5py.File(self.string, 'r')
        if self.str2 == 'data':
            load_dat = open_me[sub_fol][:]

            r, c = np.shape(load_dat)
            if (c > r):
                load_dat = np.transpose(load_dat)
                c = r

            if ECG_pref_on:
                if (np.diff(load_dat[:, 0]) > 0).all() == True:
                    # They are all increasing therefore time series - use next column
                    tcol = 1
                    opts = []
                    for ix in range(1, c):
                        opts = np.append(opts, str(ix))

                    setupperbutton(opts)
                    Fs = int(1 / (np.mean(np.diff(load_dat[:, 0]))))
                    dat = load_dat[:, 1]
                    x = np.arange(len(dat))
                    xminn = 0
                    Prediction_mode(1)
                else:
                    # They vary in magintude and are therefore the ECG of interest
                    tcol = 0
                    opts = []
                    for ix in range(c):
                        opts = np.append(opts, str(ix))
                    setupperbutton(opts)
                    dat = load_dat[:, 0]
                    x = np.arange(len(dat))
                    xminn = 0
                    onNoFSdata()

            else:
                R_t = load_dat[:, 0]
                R_t = R_t[R_t != 0]
                R_t = np.reshape(R_t, [len(R_t), ])

                if (np.diff(R_t[:]) > 0).all() == False:
                    # They aren't all greater than the previous - therefore RRI series not time-stamps
                    tmp = np.zeros(np.size(R_t))

                    tmp[0] = R_t[0]

                    for i in range(1, np.size(R_t)):
                        tmp[i] = tmp[i - 1] + R_t[i]

                R_t = np.reshape(tmp, [len(tmp), 1]) * Fs

                onNoFSdata()

        elif self.str2 == 'ann':
            loaded_ann = 1
            True_R_t = open_me[sub_fol][:]
            if Preferences[23] == '1':
                True_R_t = True_R_t / 1e3
            if np.mean(
                    np.diff(
                        True_R_t)) < 6:  # Average time interval between heart beats wouldn't be less than 10bpm, so a gap of over 6 seconds on average or greater means loaded in as samples
                True_R_t = True_R_t * Fs  # Measured in time but need samples
            siz = np.size(True_R_t)

            if not (np.diff(True_R_t[:, 0]) > 0).all():
                # They aren't all greater than the previous - therefore RRI series not time-stamps
                # SUM THEM
                tmp = np.zeros(siz)
                tmp[0] = True_R_t[0]

                for i in range(1, siz):
                    tmp[i] = tmp[i - 1] + True_R_t[i]

            True_R_t = np.reshape(True_R_t, [siz, 1])
            True_R_amp = np.zeros(siz)
            for i in range(siz):
                True_R_amp[i] = dat[int(True_R_t[i])]

        self.parent.withdraw()


# ~~~~~~~~~~~~~~ WINDOWS - POP-UP WHEN FS IS UNKOWN ~~~~~~~~~~~~~~~~~~~#
class Sampling_rate(Frame):
    def __init__(self, parent):
        global screen_height
        global screen_width
        Frame.__init__(self, parent)
        self.parent = parent
        size = tuple(int(_) for _ in self.parent.geometry().split('+')[0].split('x'))
        x = screen_width / 2 - size[0] / 2 - screen_width / 10
        y = screen_height / 3 - size[1] / 2
        self.parent.geometry("+%d+%d" % (x, y))
        self.parent.title("Set Sampling Rate")
        self.pack(fill='both', expand=0)
        self.config(bg='white')
        self.initUI()

    def initUI(self):
        big_frame = Frame(self.parent, bg='white smoke', borderwidth=20)
        big_frame.pack(side='top')
        small_frame = Frame(big_frame, bg='white', highlightcolor='grey', highlightthickness=1,
                            highlightbackground='black')
        small_frame.pack(side='left', fill='y')
        USF = Frame(small_frame, bg='light blue')
        USF.pack(side='top', fill='both')
        LSF = Frame(small_frame, bg='white')
        LSF.pack(side='top', fill='y')
        ULSF = Frame(small_frame, bg='white')
        ULSF.pack(side='top', fill='y')
        LLSF = Frame(small_frame, bg='white')
        LLSF.pack(side='top', fill='y')
        Label(USF, text="ECG Sampling Rate was not detected from input file", wraplength=200, font='Helvetica 12 bold',
              bg='light blue', borderwidth=3).pack(side='top')
        Label(ULSF, text="The sampling rate of the ECG recording is:", font='Helvetica 10', wraplength=200, bg='white',
              pady=10).pack(side='top')
        self.Samp = Entry(ULSF, width=10, highlightbackground='white', readonlybackground='white', justify='center')
        self.Samp.pack(side='top')
        self.Samp.insert(0, Preferences[4])
        self.Samp.bind("<Return>", self.callback)
        self.Samp.bind("<KP_Enter>", self.callback)
        Button(LLSF, text="OK", highlightbackground='white', command=self.callback, font='Helvetica 12 bold').pack(
            side='left')
        Label(LLSF, bg='white', text="  ", font='Helvetica 12 bold').pack(side='left')
        Button(LLSF, text="Cancel", highlightbackground='white', command=self.callback2, font='Helvetica 12 bold').pack(
            side='left')

    def callback(self,
                 event=0):  # Setting event=0 makes it an optional input; meaning that a button and keyboard press can activate the same sub-function
        global Fs
        Fs = int(self.Samp.get())
        if ECG_pref_on:
            Prediction_mode(1)
            draw1()
        else:
            draw3()
        self.parent.withdraw()

    def callback2(self):
        self.parent.withdraw()


# ~~~~~~~~~~~~~~ WINDOWS - Helpful suggestion when hover over for a period of time ~~~~~~~~~~~~~~~~~~~#
class Pop_up(object):
    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def showpopup(self, text):
        xx = root.winfo_pointerx()
        yy = root.winfo_pointery() + 10
        self.tipwindow = Toplevel(self.widget)
        Label(self.tipwindow, text=text, font='Helvetica 8', bg='light blue').pack()
        self.tipwindow.wm_geometry("+%d+%d" % (xx, yy))
        self.tipwindow.overrideredirect(True)

    def hidepopup(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()


class Pop_up2(object):
    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def showpopup(self, text):
        xx = root.winfo_pointerx() - 150
        yy = root.winfo_pointery() + 10
        self.tipwindow = Toplevel(self.widget)
        Label(self.tipwindow, text=text, font='Helvetica 8', bg='light blue').pack()
        self.tipwindow.wm_geometry("+%d+%d" % (xx, yy))
        self.tipwindow.overrideredirect(True)

    def hidepopup(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()


class File_type_and_Fs(Frame):
    def __init__(self, parent):
        #        global siz
        Frame.__init__(self, parent)
        self.parent = parent
        self.parent.title("Select File Type and Sampling Rate")
        self.pack(fill='both', expand=0)
        self.config(bg='white')
        OuterFrame_top = Frame(self.parent, bg='white smoke', borderwidth=5)
        OuterFrame_top.pack(side='top', fill='both')
        small_frame3 = Frame(OuterFrame_top, bg='white', highlightcolor='grey', highlightthickness=1,
                             highlightbackground='black')
        small_frame3.pack(side='left', fill='y')
        spacer = Label(OuterFrame_top, text=' ', bg='white smoke')
        spacer.pack(side='left', fill='y')
        USF3 = Frame(small_frame3, bg='light blue')
        USF3.pack(side='top', fill='both')
        LSF3 = Frame(small_frame3, bg='white')
        LSF3.pack(side='top', fill='y')
        ULSF3 = Frame(small_frame3, bg='white')
        ULSF3.pack(side='top', fill='y')
        LLSF3 = Frame(small_frame3, bg='white')
        LLSF3.pack(side='top', fill='y')
        Label(USF3, text="Input Data Type", wraplength=200, font='Helvetica 12 bold', bg='light blue',
              borderwidth=3).pack(side='top')
        Label(ULSF3, text="Select the format of the input ECG data:", font='Helvetica 10', wraplength=200, bg='white',
              pady=10).pack(side='top')
        self.typ = StringVar(ULSF3)
        self.options = ['Text files *.txt', 'HDF5 files *.h5', 'MAT files *.mat', 'WFDB files *.dat']
        Ftypemenu = OptionMenu(ULSF3, self.typ, self.options[0], *self.options)
        Ftypemenu.config(width=17)
        #        RRImenu.configure(compound='right',image=self.photo)
        Ftypemenu.pack(side='top')
        #        Label(ULSF2, text="Enter the sampling rate of the ECG database selected:", font='Helvetica 10', wraplength=200, bg='white', pady=10).pack(side='top')
        small_frame2 = Frame(OuterFrame_top, bg='white', highlightcolor='grey', highlightthickness=1,
                             highlightbackground='black')
        small_frame2.pack(side='left', fill='y')
        spacer = Label(OuterFrame_top, text=' ', bg='white smoke')
        spacer.pack(side='left', fill='y')
        USF2 = Frame(small_frame2, bg='light blue')
        USF2.pack(side='top', fill='both')
        LSF2 = Frame(small_frame2, bg='white')
        LSF2.pack(side='top', fill='y')
        ULSF2 = Frame(small_frame2, bg='white')
        ULSF2.pack(side='top', fill='y')
        LLSF2 = Frame(small_frame2, bg='white')
        LLSF2.pack(side='top', fill='y')
        Label(USF2, text="Output File Specifications", wraplength=200, font='Helvetica 12 bold', bg='light blue',
              borderwidth=3).pack(side='top')
        Label(ULSF2, text="Enter the name of the output file (with file type):", font='Helvetica 10', wraplength=200,
              bg='white', pady=10).pack(side='top')
        self.name = Entry(ULSF2, width=20, highlightbackground='white', readonlybackground='white', justify='center')
        self.name.pack(side='top')
        self.name.bind("<Return>", self.callback)
        self.name.bind("<KP_Enter>", self.callback)
        small_frame = Frame(OuterFrame_top, bg='white', highlightcolor='grey', highlightthickness=1,
                            highlightbackground='black')
        small_frame.pack(side='left', fill='y')
        USF = Frame(small_frame, bg='light blue')
        USF.pack(side='top', fill='both')
        LSF = Frame(small_frame, bg='white')
        LSF.pack(side='top', fill='y')
        ULSF = Frame(small_frame, bg='white')
        ULSF.pack(side='top', fill='y')
        LLSF = Frame(small_frame, bg='white')
        LLSF.pack(side='top', fill='y')
        Label(USF, text="Sampling Rate (Fs)", wraplength=200, font='Helvetica 12 bold', bg='light blue',
              borderwidth=3).pack(side='top')
        Label(ULSF, text="Enter the sampling rate of the ECG database selected:", font='Helvetica 10', wraplength=200,
              bg='white', pady=10).pack(side='top')
        self.Samp = Entry(ULSF, width=20, highlightbackground='white', readonlybackground='white', justify='center')
        self.Samp.pack(side='top')
        self.Samp.bind("<Return>", self.callback)
        self.Samp.bind("<KP_Enter>", self.callback)
        OuterFrame_bottom = Frame(self.parent, bg='white smoke')
        OuterFrame_bottom.pack(side='top', fill='both')
        Button(OuterFrame_bottom, text="Cancel", command=self.callback2, font='Helvetica 12 bold').pack(side='right')
        Label(OuterFrame_bottom, text="  ", bg='white smoke', font='Helvetica 12 bold').pack(side='right')
        Button(OuterFrame_bottom, text="OK", command=self.callback, font='Helvetica 12 bold').pack(side='right')
        self.parent.withdraw()
        self.parent.update_idletasks()  # Update "requested size" from geometry manager
        x = (self.parent.winfo_screenwidth() - self.parent.winfo_reqwidth()) / 2
        y = (self.parent.winfo_screenheight() - self.parent.winfo_reqheight()) / 2
        self.parent.geometry("+%d+%d" % (x, y))
        self.parent.deiconify()

    def callback(self,
                 event=0):  # Setting event=0 makes it an optional input; meaning that a button and keyboard press can activate the same sub-function
        global PATH
        fname, file_extension = os.path.splitext(self.name.get())
        if (self.Samp.get() == '') or (self.Samp.get() == ' '):
            messagebox.showwarning("Warning", "Sampling frequency required. Try again.")
        elif (self.name.get() == '') or (self.name.get() == ' '):
            messagebox.showwarning("Warning", "File name required for output file! Try again.")
        elif (file_extension == ''):
            messagebox.showwarning("Warning", "File type required for output file! Try again.")
        else:
            self.parent.withdraw()
            multRUN(PATH, fname, int(self.Samp.get()), file_extension)

    def callback2(self, event=0):
        self.parent.withdraw()


class TimeDomain(tb.IsDescription):
    SDNN = tb.FloatCol(pos=1)
    SDANN = tb.FloatCol(pos=2)
    MeanRR = tb.FloatCol(pos=3)
    RMSSD = tb.FloatCol(pos=4)
    pNN50 = tb.FloatCol(pos=5)


class FrequencyDomain(tb.IsDescription):
    VLF_power = tb.FloatCol(pos=1)
    LF_power = tb.FloatCol(pos=2)
    HF_power = tb.FloatCol(pos=3)
    VLF_P_power = tb.FloatCol(pos=4)
    LF_P_power = tb.FloatCol(pos=5)
    HF_P_power = tb.FloatCol(pos=6)
    VLF_PF = tb.FloatCol(pos=7)
    LF_PF = tb.FloatCol(pos=8)
    HF_PF = tb.FloatCol(pos=9)
    LFHFRatio = tb.FloatCol(pos=10)


class NonlinearMets(tb.IsDescription):
    Recurrence = tb.FloatCol(pos=1)
    Determinism = tb.FloatCol(pos=2)
    Laminarity = tb.FloatCol(pos=3)
    L_mean = tb.FloatCol(pos=4)
    L_max = tb.FloatCol(pos=5)
    V_mean = tb.FloatCol(pos=6)
    V_max = tb.FloatCol(pos=7)
    SD1 = tb.FloatCol(pos=8)
    SD2 = tb.FloatCol(pos=9)
    Alpha1 = tb.FloatCol(pos=10)
    Alpha2 = tb.FloatCol(pos=11)


# LAUNCH SCRIPT#
RRAPET(root)
# ~~~~~~~~~~~ Keys bound to operating window ~~~~~~~~~~#
if windows_compile:
    root.bind('<Escape>', shut)
if linux_compile:
    root.bind('<Control-Escape>', shut)
root.bind('Control-i', Invert)

root.mainloop()
