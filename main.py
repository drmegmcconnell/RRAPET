"""
RR-APET

Author: Meghan McConnell
"""
# import sys
import os
# from os import listdir
# from os.path import isfile, join, splitext
import tkinter.constants as TKc
import scipy.io as sio
import h5py
import wfdb
from tkinter import Frame, Tk, Scale, Menu, StringVar, filedialog, Toplevel, \
    Label, BOTTOM, TOP, BOTH, messagebox, RIGHT, Entry, PhotoImage  # END, Button, ANCHOR, FALSE, Listbox, Scrollbar
from tkinter.ttk import Style, OptionMenu
from tkinter.ttk import Button as Button2
from tkinter.ttk import Label as Label2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys

sys.path.append('/home/meg/Documents/RRAPET_Linux/')

from utils import MHTD, pan_tompkin, ECG_processing, Exporter, H5_Selector
from utils.HRV_Functions import *
from utils.peak_detection.custom_detection import my_function
from utils.Style_Functions import *

global enable, tt, edit_flag, loaded_ann, invert_flag, warnings_on, enable2, plot_pred, plot_ann, R_t, \
    h, pref, lower_RHS, button_help_on, delete_flag, TOTAL_FRAME, windows_compile, linux_compile, cnt, \
    R_amp, dat, True_R_t, True_R_amp, xminn, labelled_flag, Preferences, disp_length, pred_mode, Fs, ECG_pref_on, \
    graphCanvas, fig, t, fig2, RR_interval_Canvas, Slider, edit_btn, screen_height, screen_width, upper, plot_fig, \
    plot_fig2, rhs_ecg_frame, test_txt, Outerframe, F, F2, big_frame, btn, trybtn, resetbtn, okbtn, Meth, entry, \
    fs_frame_2, fs_frame, x, draw_figure, graphCanvas2, plt_options, pltmenu, load_dat, tcol, plot_num, \
    cid, h5window, h5window2, contact, helper


def set_initial_values():
    """
    Setting all initial values
    """
    global enable, tt, edit_flag, loaded_ann, invert_flag, warnings_on, enable2, plot_pred, plot_ann, R_t, \
        h, pref, lower_RHS, button_help_on, delete_flag, TOTAL_FRAME, windows_compile, linux_compile, \
        labelled_flag, Preferences, disp_length, pred_mode, Fs, ECG_pref_on, True_R_t, True_R_amp
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


Pref_dict = {
    'values': {
        'font': 'Helvetica',
        'base_font_size': 12,
        'disp_len': 20,
        'rpeak_meth': 1,
        'fs': 360,
        'welch_L': 5,
        'welch_O': 50,
        'bltk_input': 10,
        'ls_omega_max': 500,
        'ar_order': 10,
        'minbox': 1,
        'copbox': 4,
        'maxbox': 64,
        'increm': 1,
        'rqa_m': 10,
        'rqa_l': 1,
        'vthr': 1.15,
        'sbl': 5,
        'mag_lim': 0.1,
        'eng_lim': 0.05,
        'l_min': 0.3,
        'ECG_pref_on': 1
    },
    'description': {
        'font': 'Font style',
        'base_font_size': 'Base Font Size',
        'disp_len': 'Display length of ECG single screen',
        'rpeak_meth': 'Set 1 for MHTD, 2 for Pan-Tompkins, 3 for original HTD, 4 for K-means, or 5 for your own method',
        'fs': 'Sampling Frequency of ECG',
        'welch_L': 'Welch length for analysis',
        'welch_O': 'Welch overlap percentage',
        'bltk_input': ' ',
        'ls_omega_max': ' ',
        'ar_order': ' ',
        'minbox': ' ',
        'copbox': ' ',
        'maxbox': ' ',
        'increm': ' ',
        'rqa_m': ' ',
        'rqa_l': ' ',
        'vthr': 'Variable threshold ratio between max values and height of thr',
        'sbl': 'Search back length - time (in seconds) for vthr',
        'mag_lim': 'Magnitude lim - peaks within set proxmity must meet this criteria to both remain in predictions',
        'eng_lim': 'Energy lim - peaks within set proxmity must meet this criteria to both remain in predictions',
        'l_min': 'Minimum length - time proximity (in seconds) for mag_lim and eng_lim',
        'ECG_pref_on': ' '
    }
}

# 0
# 0
# 0

root = Tk()
root.style = Style()
root.style.theme_use("alt")
root.style.configure('C.TButton', background='light grey', padding=0, relief='raised', width=0, highlightthickness=0)
root.style.configure('B.TButton', background='white smoke', padding=0, relief='raised', width=0, highlightthickness=0)

set_initial_values()
exp = Exporter(Preferences)

# ~~~~~~~~~~~~~~ KEY PRESS FUNCTIONS ~~~~~~~~~~~~~~~~~~~#
cust_text, cust_subheader, cust_subheadernb, cust_header = headerStyles(Preferences)


def Invert():
    """
    Invert Functionality
    """
    global invert_flag
    invert_flag ^= 1
    if invert_flag & warnings_on:
        messagebox.showwarning("Warning",
                               "You have selected the inverted peak option. \n\nWhen editing annotations, "
                               "the program will now find local minima with the left mouse click and local "
                               "maxima with the right mouse click. To reverse this selection, simply press "
                               "the 'i' key.")
    elif warnings_on & invert_flag != 1:
        messagebox.showwarning("Warning", "The inverted peak option has been cancelled.")


def shut(event):
    """
    Shutdown program
    """
    root.destroy()
    print(event)
    # os._exit(1)


# ~~~~~~~~~~~~~~ MOUSE PRESS FUNCTIONS ~~~~~~~~~~~~~~~~~~~#
def __onclick(event):
    global Fs, cnt, invert_flag, R_t, R_amp, True_R_t, True_R_amp, plot_ann, plot_pred, x, xminn, dat, \
        plot_fig, graphCanvas, fig, loaded_ann, labelled_flag, disp_length, fig2, plot_fig2, RR_interval_Canvas

    bound1 = int(0.05 / (1 / Fs))  # Gets approximately +- 50ms le-way with button click

    if ((event.button == 1) and (delete_flag == 0)) or ((event.button == 2) and (delete_flag == 1)):
        if plot_ann == 1 and plot_pred == 0:  # Checks to see if editing loaded annotations or predicted annotations
            if loaded_ann == 0 & warnings_on:
                messagebox.showwarning("Warning", "Cannot edit 'imported annotations' \n\nPlease note: Switch set the "
                                                  "RRI plot to RR_Predicitions or load in previously determined "
                                                  "annotations.")
            else:
                xx = int(event.xdata * Fs)
                ll = np.size(True_R_t)

                if invert_flag == 0:
                    R_amp_temp = np.max(dat[xx - bound1:xx + bound1])
                    pl = np.argmax(dat[xx - bound1:xx + bound1])
                else:
                    R_amp_temp = np.min(dat[xx - bound1:xx + bound1])
                    pl = np.argmin(dat[xx - bound1:xx + bound1])

                R_t_temp = xx - (bound1 + 1) + pl

                pl2 = np.argmin(np.abs(True_R_t - R_t_temp))

                if R_t_temp < True_R_t[pl2]:
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

                draw1(x, xminn, Fs, dat, plot_fig, graphCanvas, fig, R_t, R_amp, loaded_ann, labelled_flag, True_R_t,
                      True_R_amp, disp_length, plot_pred, plot_ann, fig2, plot_fig2, RR_interval_Canvas)

        elif plot_pred == 1 and plot_ann == 0:

            if len(R_t) == 0 & warnings_on:
                messagebox.showwarning("Warning", "Cannot edit empty annotations \n\nPlease generate annotations "
                                                  "using the prediction functionality.")
            else:
                xx = int(event.xdata * Fs)
                ll = np.size(R_t)

                if invert_flag == 0:
                    R_amp_temp = np.max(dat[xx - bound1:xx + bound1])
                    pl = np.argmax(dat[xx - bound1:xx + bound1])
                else:
                    R_amp_temp = np.min(dat[xx - bound1:xx + bound1])
                    pl = np.argmin(dat[xx - bound1:xx + bound1])

                R_t_temp = xx - (bound1 + 1) + pl

                pl2 = np.argmin(np.abs(R_t - R_t_temp))

                if R_t_temp < R_t[pl2]:
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

                draw1(x, xminn, Fs, dat, plot_fig, graphCanvas, fig, R_t, R_amp, loaded_ann, labelled_flag, True_R_t,
                      True_R_amp, disp_length, plot_pred, plot_ann, fig2, plot_fig2, RR_interval_Canvas)

        else:
            if warnings_on:
                messagebox.showwarning("Warning", "Edit mode not activated. \n\nCannot edit while both RRI plots are "
                                                  "active, please select either the RR_Predictions or RR_Annotations "
                                                  "and re-activate edit mode.")
            edit_toggle()

    elif ((event.button == 2) and (delete_flag == 0)) or ((event.button == 1) and (delete_flag == 1)):

        if plot_ann == 1 and plot_pred == 0:  # Checks to see if editing loaded annotations or predicted annotations
            if loaded_ann == 0 & warnings_on:
                messagebox.showwarning("Warning", "Cannot edit 'imported annotations' \n\nPlease note: Switch set the"
                                                  " RRI plot to RR_Predicitions or load in previously determined "
                                                  "annotations.")
            elif loaded_ann:
                if invert_flag == 0:
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

                draw1(x, xminn, Fs, dat, plot_fig, graphCanvas, fig, R_t, R_amp, loaded_ann, labelled_flag, True_R_t,
                      True_R_amp, disp_length, plot_pred, plot_ann, fig2, plot_fig2, RR_interval_Canvas)

        elif plot_pred == 1 and plot_ann == 0:

            if len(R_t) == 0 & warnings_on:
                messagebox.showwarning("Warning", "Cannot edit empty annotations \n\nPlease generate annotations "
                                                  "using the prediction functionality.")
            elif len(R_t) > 0:
                if invert_flag == 0:
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

                draw1(x, xminn, Fs, dat, plot_fig, graphCanvas, fig, R_t, R_amp, loaded_ann, labelled_flag, True_R_t,
                      True_R_amp, disp_length, plot_pred, plot_ann, fig2, plot_fig2, RR_interval_Canvas)
        else:
            if warnings_on:
                messagebox.showwarning("Warning", "Edit mode not activated. \n\nCannot edit while both RRI plots are "
                                                  "active, please select either the RR_Predictions or RR_Annotations "
                                                  "and re-activate edit mode.")
            edit_toggle()

    elif event.button == 3:
        if plot_ann == 1 and plot_pred == 0:  # Checks to see if editing loaded annotations or predicted annotations
            if loaded_ann == 0 & warnings_on:
                messagebox.showwarning("Warning",
                                       "Cannot edit 'imported annotations' \n\nPlease note: Switch set the RRI plot "
                                       "to RR_Predicitions or load in previously determined annotations.")
            elif loaded_ann:  # Checks to see if editing loaded annotations or predicted annotations
                xx = int(event.xdata * Fs)
                ll = np.size(True_R_t)

                if invert_flag == 0:
                    R_amp_temp = np.min(dat[xx - bound1:xx + bound1])
                    pl = np.argmin(dat[xx - bound1:xx + bound1])

                else:
                    R_amp_temp = np.max(dat[xx - bound1:xx + bound1])
                    pl = np.argmax(dat[xx - bound1:xx + bound1])

                R_t_temp = xx - (bound1 + 1) + pl

                pl2 = np.argmin(np.abs(True_R_t - R_t_temp))

                if R_t_temp < True_R_t[pl2]:
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

                draw1(x, xminn, Fs, dat, plot_fig, graphCanvas, fig, R_t, R_amp, loaded_ann, labelled_flag, True_R_t,
                      True_R_amp, disp_length, plot_pred, plot_ann, fig2, plot_fig2, RR_interval_Canvas)

        elif plot_pred == 1 and plot_ann == 0:

            if len(R_t) == 0 & warnings_on:
                messagebox.showwarning("Warning", "Cannot edit empty annotations \n\nPlease generate annotations "
                                                  "using the prediction functionality.")
            elif len(R_t) > 0:
                xx = int(event.xdata * Fs)
                ll = np.size(R_t)

                if invert_flag == 0:
                    R_amp_temp = np.min(dat[xx - bound1:xx + bound1])
                    pl = np.argmin(dat[xx - bound1:xx + bound1])

                else:
                    R_amp_temp = np.max(dat[xx - bound1:xx + bound1])
                    pl = np.argmax(dat[xx - bound1:xx + bound1])

                R_t_temp = xx - (bound1 + 1) + pl

                pl2 = np.argmin(np.abs(R_t - R_t_temp))

                if R_t_temp < R_t[pl2]:
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

                draw1(x, xminn, Fs, dat, plot_fig, graphCanvas, fig, R_t, R_amp, loaded_ann, labelled_flag,
                      True_R_t, True_R_amp, disp_length, plot_pred, plot_ann, fig2, plot_fig2, RR_interval_Canvas)
        else:
            if warnings_on:
                messagebox.showwarning("Warning", "Edit mode not activated. \n\nCannot edit while both RRI plots are "
                                                  "active, please select either the RR_Predictions or RR_Annotations "
                                                  "and re-activate edit mode.")
            edit_toggle()


def onclick2(event):
    """
    Slider controlling event
    :param event:
    """
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
    """
    Edit Toggle for xxx
    """
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
        cid = fig.canvas.mpl_connect('button_press_event', __onclick)


# ~~~~~~~~~~~~~~ OTHER FUNCTIONS ~~~~~~~~~~~~~~~~~~~#
def replace_line(file_name, line_number, new_text):
    """

    :param file_name:
    :param line_number:
    :param new_text:
    """
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

# def draw1():
#     """
#     Draw 1
#     """
#     global x, xminn, Fs, dat, plot_fig, graphCanvas, fig, R_t, R_amp, loaded_ann, labelled_flag, True_R_t, \
#         True_R_amp, disp_length, plot_pred, plot_ann
#
#     xmaxx = int(xminn + disp_length * Fs)
#     yminn = np.min(dat[int(xminn):xmaxx]) - 0.1
#     ymaxx = np.max(dat[int(xminn):xmaxx]) + 0.1
#
#     # Top Figure
#     plot_fig.clear()
#     plot_fig.plot(x / Fs, dat, color='k', linewidth=1)
#     if labelled_flag & plot_pred:
#         plot_fig.plot(R_t / Fs, R_amp, 'b*', linewidth=1, markersize=7)
#     if loaded_ann & plot_ann:
#         plot_fig.plot(True_R_t / Fs, True_R_amp, 'ro', linewidth=1, markersize=5, fillstyle='none')
#     plot_fig.axis([xminn / Fs, xmaxx / Fs, yminn, ymaxx])  # ([xminn,xmaxx,yminn,ymaxx])
#     plot_fig.set_xlabel('Time (sec)')
#     plot_fig.set_ylabel('ECG Amplitude (mV)')
#     fig.tight_layout()
#     graphCanvas.draw()
#
#     draw2()


# def draw2(xmaxx):
#     """
#     Draw 2
#     """
#     global R_t, R_amp, True_R_t, True_R_amp, dat, x, fig2, plot_fig2, RR_interval_Canvas, lab, labelled_flag, xminn, \
#         plot_ann, plot_pred, Fs
#
#     if plot_ann ^ plot_pred:
#         if plot_pred:
#             plotx = R_t
#         else:
#             plotx = True_R_t
#
#         plotx = np.reshape(plotx, (len(plotx), 1))
#
#         x2_max = np.max(plotx) / Fs
#         sRt = np.size(plotx)
#
#         y_diff = (np.diff(plotx[:, 0]) / Fs) * 1000
#
#         pl = xminn / Fs
#         pl2 = xmaxx / Fs
#
#         y_minn = np.min(y_diff) - 10
#         y_maxx = np.max(y_diff) + 10
#
#         plot_fig2.clear()
#         x2 = plotx[1:sRt] / Fs
#         x2 = np.reshape(x2, (len(x2),))
#
#         if plot_pred:
#             plot_fig2.plot(x2, y_diff, 'b*', label='Predicted beats')
#         else:
#             plot_fig2.plot(x2, y_diff, 'ro', label='Annotated beats', markersize=5, fillstyle='none')
#
#         plot_fig2.plot([pl + 0.05, pl + 0.05], [y_minn, y_maxx], 'k')
#         plot_fig2.plot([pl2, pl2], [y_minn, y_maxx], 'k')
#         plot_fig2.axis([0, x2_max, y_minn, y_maxx])
#         plot_fig2.set_xlabel('Time (sec)')
#         plot_fig2.set_ylabel('R-R Interval (ms)')
#         plot_fig2.fill_between(x2, y_minn, y_maxx, where=x2 <= pl2, facecolor='gainsboro')
#         plot_fig2.fill_between(x2, y_minn, y_maxx, where=x2 <= pl, facecolor='white')
#         plot_fig2.legend()
#         fig2.tight_layout()
#         RR_interval_Canvas.draw()
#
#     elif plot_ann & plot_pred:
#         plot_fig2.clear()
#         y_minn = np.zeros(2)
#         y_maxx = np.zeros(2)
#         x2_maxx = np.zeros(2)
#
#         #        pl = xminn/Fs#np.argmin(np.abs(R_t-xminn))
#         #        pl2 = xmaxx/Fs# np.argmin(np.abs(R_t-xmaxx))
#         #
#         #        for i in range (0, (sRt-1)):
#         #            y_diff[i] = plotx[i+1] - plotx[i]
#         #
#         #        y_diff = (y_diff/Fs)*1000   #Converts from sample number to millseconds
#         #        x2 = range(0, (sRt-1))
#
#         #
#         for N in range(2):
#             if N == 0:
#                 plotx = R_t
#             else:
#                 plotx = True_R_t
#             sRt = np.size(plotx)
#             y_diff = (np.diff(plotx[:, 0]) / Fs) * 1000
#             y_minn[N] = np.min(y_diff) - 10
#             y_maxx[N] = np.max(y_diff) + 10
#             x2_maxx[N] = np.max(plotx) / Fs
#
#             if N == 0:
#                 plot_fig2.plot(plotx[1:sRt] / Fs, y_diff, 'b*', label='Predicted beats')
#             else:
#                 plot_fig2.plot(plotx[1:sRt] / Fs, y_diff, 'ro', label='Annotated beats', markersize=5,
#                 fillstyle='none')
#
#         plot_fig2.set_xlabel('Time (sec)')
#         plot_fig2.set_ylabel('R-R Interval (ms)')
#         if y_minn[0] < y_minn[1]:
#             plot_min = y_minn[0]
#         else:
#             plot_min = y_minn[1]
#         if y_maxx[0] > y_maxx[1]:
#             plot_max = y_maxx[0]
#         else:
#             plot_max = y_maxx[1]
#         if x2_maxx[0] < x2_maxx[1]:
#             plot_max_x2 = x2_maxx[0]
#         else:
#             plot_max_x2 = x2_maxx[1]
#         plot_fig2.plot([(xminn / Fs) + 0.5, (xminn / Fs) + 0.5], [y_minn, y_maxx], 'k')
#         plot_fig2.plot([(xmaxx / Fs), (xmaxx / Fs)], [y_minn, y_maxx], 'k')
#         plot_fig2.axis([0, plot_max_x2, plot_min, plot_max])
#         plot_fig2.legend()
#         fig2.tight_layout()
#         RR_interval_Canvas.draw()


def draw3():
    """
    Draw Number 3
    """
    global R_t
    global fig2
    global plot_fig2
    global RR_interval_Canvas
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
    """

    :param METHOD:
    :param DECIMALS:
    :param subplot_:
    :param m:
    :param o:
    :param btval:
    :param omax:
    :param Ord:
    """
    global draw_figure
    global R_t
    global graphCanvas2
    Rpeakss = np.reshape(R_t, (len(R_t),))
    Rpeak_input = Rpeakss / Fs
    fig_holder = Figure(dpi=150)
    f, P2 = Freq_Analysis_fig(Rpeak_input, meth=METHOD, decim=DECIMALS, Fig=fig_holder, M=m, O=o, BTval=btval,
                              omega_max=omax, order=Ord)
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
    """

    :param subplot_:
    :param Min:
    :param Max:
    :param Inc:
    :param COP:
    """
    global draw_figure
    global R_t
    global graphCanvas2
    Rpeakss = np.reshape(R_t, (len(R_t),))
    Rpeak_input = Rpeakss / Fs
    RRI = np.diff(Rpeak_input)
    (x_vals1, y_vals1, y_new1, x_vals2, y_vals2, y_new2, a1, a2) = DFA_fig(RRI, min_box=Min, max_box=Max, inc=Inc,
                                                                           cop=COP, decim=3)
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
    """

    :param subplot_:
    """
    global draw_figure
    global R_t
    global graphCanvas2
    Rpeakss = np.reshape(R_t, (len(R_t),))
    Rpeak_input = Rpeakss / Fs
    RRI = np.diff(Rpeak_input)
    sd1, sd2, c1, c2 = Poincare(RRI, decim=3)
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
    """

    :param subplot_:
    :param Mval:
    :param Lval:
    """
    Rpeakss = np.reshape(R_t, (len(R_t),))
    Rpeak_input = Rpeakss / Fs
    RRI = np.diff(Rpeak_input)
    Matrix, N = RQA_matrix(RRI, m=Mval, L=Lval)
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
    """

    :param mode_type:
    :param thr_ratio:
    :param SBL:
    :param MAG_LIM:
    :param ENG_LIM:
    :param MIN_L:
    """
    global dat, R_t, R_amp, xminn, labelled_flag, x, Fs, dat, plot_fig, graphCanvas, fig, R_t, R_amp, loaded_ann, \
        labelled_flag, True_R_t, True_R_amp, disp_length, plot_pred, plot_ann, fig2, plot_fig2, RR_interval_Canvas

    if Preferences[22] == '1':  # if Volts multiple to get mV
        dat = dat * 1e3

    if mode_type == 1:
        # ========================== Set Values =========================#
        labelled_flag = 1
        fs = Fs
        fpass = 0.5
        fstop = 45
        # ====================== Conduct Predictions =======================#
        R_t = MHTD(dat, fs, fpass, fstop, thr_ratio=thr_ratio, sbl=SBL, mag_lim=MAG_LIM, eng_lim=ENG_LIM, min_L=MIN_L)

        siz = np.size(R_t)
        R_t = np.reshape(R_t, [siz, 1])
        R_amp = np.zeros(siz)
        for i in range(siz):
            R_amp[i] = dat[int(R_t[i])]
        if len(R_amp) == 1:
            R_amp = np.transpose(R_amp)
        draw1(x, xminn, Fs, dat, plot_fig, graphCanvas, fig, R_t, R_amp, loaded_ann, labelled_flag, True_R_t,
              True_R_amp, disp_length, plot_pred, plot_ann, fig2, plot_fig2, RR_interval_Canvas)
    elif mode_type == 2:
        # ========================== Set Values =========================#
        labelled_flag = 1
        data_input = np.reshape(dat, (len(dat),))
        # ====================== Conduct Predictions =======================#
        R_amp, R_t, delay = pan_tompkin(data_input, Fs, 0)
        draw1(x, xminn, Fs, dat, plot_fig, graphCanvas, fig, R_t, R_amp, loaded_ann, labelled_flag, True_R_t,
              True_R_amp, disp_length, plot_pred, plot_ann, fig2, plot_fig2, RR_interval_Canvas)
    elif mode_type == 3:
        # ========================== Set Values =========================#
        labelled_flag = 1
        # ====================== Conduct Predictions =======================#
        R_t = ECG_processing(dat)
        siz = np.size(R_t)
        R_t = np.reshape(R_t, [siz, 1])
        R_amp = np.zeros(siz)
        for i in range(siz):
            R_amp[i] = dat[int(R_t[i])]
        if len(R_amp) == 1:
            R_amp = np.transpose(R_amp)
        draw1(x, xminn, Fs, dat, plot_fig, graphCanvas, fig, R_t, R_amp, loaded_ann, labelled_flag, True_R_t,
              True_R_amp, disp_length, plot_pred, plot_ann, fig2, plot_fig2, RR_interval_Canvas)
    elif mode_type == 4:
        # ========================== Set Values =========================#
        labelled_flag = 1
        # ====================== Conduct Predictions =======================#
        R_t, R_amp = my_function(dat)
        draw1(x, xminn, Fs, dat, plot_fig, graphCanvas, fig, R_t, R_amp, loaded_ann, labelled_flag, True_R_t,
              True_R_amp, disp_length, plot_pred, plot_ann, fig2, plot_fig2, RR_interval_Canvas)
    else:
        print('not imported yet')


def check_orientation(data):
    """
    Check the orientation of data.
    :param data:
    :return:
    """
    rows, cols = np.shape(data)
    if cols > rows:
        data = np.transpose(dat)
        cols = rows
    return data, rows, cols


def check_for_fs_col():
    """
    Common Repetition
    """

    global tcol, Fs, dat, load_dat, x, xminn
    _, c = np.shape(load_dat)
    if (np.diff(load_dat[:, 0]) > 0).all():
        # They are all increasing therefore time series - use next column
        tcol = 1
        opts = []
        for ix in range(1, c):
            opts = np.append(opts, str(ix))
        setupperbutton(opts)
        if Preferences[23] == '0':
            Fs = int(1 / (np.mean(np.diff(load_dat[:, 0]))))
        else:  # if ms divide to get s
            Fs = int(1 / (np.mean(np.diff(load_dat[:, 0] / 1e3))))
        dat = load_dat[:, 1]
        dat = np.reshape(dat, [len(dat), 1])
        x = np.arange(len(dat))
        xminn = 0
        Prediction_mode(1)
        # draw1(x, xminn, Fs, dat, plot_fig, graphCanvas, fig, R_t, R_amp, loaded_ann, labelled_flag, True_R_t,
        #       True_R_amp, disp_length, plot_pred, plot_ann, fig2, plot_fig2, RR_interval_Canvas, lab)
    else:
        # They vary in magintude and are therefore the ECG of interest
        tcol = 0
        opts = []
        for ix in range(c):
            opts = np.append(opts, str(ix))
        setupperbutton(opts)
        dat = load_dat[:, 0]
        dat = np.reshape(dat, [len(dat), 1])
        x = np.arange(len(dat))
        xminn = 0
        onNoFSdata()


def onNoFSdata():
    """
    Function for no sampling frequeny
    """
    global fs_frame
    fs_frame = Toplevel()
    # Sampling_rate(fs_frame)

    if windows_compile:
        fs_frame.bind('<Escape>', __close_fs)
    else:
        fs_frame.bind('<Control-Escape>', __close_fs)


def __close_fs(event):
    fs_frame.withdraw()


def load_h5(filename, ftype, h5folder, file_extension):
    """

    :param filename:
    :param ftype:
    :param h5folder:
    :param file_extension:
    """
    global tcol, Fs, dat, x, xminn, R_t, loaded_ann, True_R_t, True_R_amp, load_dat

    open_me = h5py.File(filename, 'r')
    if ftype == 'data':
        load_dat = open_me[h5folder][:]
        load_dat, r, c = check_orientation(load_dat)

        if ECG_pref_on:
            check_for_fs_col()

        else:
            R_t = load_dat[:, 0]
            R_t = R_t[R_t != 0]
            R_t = np.reshape(R_t, [len(R_t), ])

            if not (np.diff(R_t[:]) > 0).all():
                # They aren't all greater than the previous - therefore RRI series not time-stamps
                tmp = np.zeros(np.size(R_t))

                tmp[0] = R_t[0]

                for i in range(1, np.size(R_t)):
                    tmp[i] = tmp[i - 1] + R_t[i]

                R_t = np.reshape(tmp, [len(tmp), 1]) * Fs

            # onNoFSdata()

    elif ftype == 'ann':
        loaded_ann = 1
        True_R_t = open_me[h5folder][:]
        if Preferences[23] == '1':
            True_R_t = True_R_t / 1e3
        if np.mean(np.diff(True_R_t)) < 6:
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

    elif (file_extension == '.dat') or (file_extension == '.hea'):
        load_dat, fields = wfdb.rdsamp(filename)
        rub, columns = np.shape(load_dat)


def read_txt_file(fname):
    """
    Read a text file
    :param fname:
    """

    file = open(fname, 'r')
    if len(F) == 0 & warnings_on:
        messagebox.showwarning("Warning", "The file you have selected is unreadable. Please ensure that "
                                          "the file interest is saved in the correct format. For further "
                                          "information, refer to the 'Help' module")
    else:
        if Preferences[24] == '0':
            temp = file.read().split()
        elif Preferences[24] == '1':
            temp = file.read().split(":")
        else:
            temp = file.read().split(";")
        var1 = len(temp)
        data = np.zeros(np.size(temp))
        for i in range(var1):  # for i in range(len(temp)):
            data[i] = float(temp[i].rstrip('\n'))
        file.seek(0)
        temp2 = file.readlines()
        var2 = len(temp2)
        columns = int(var1 / var2)
        data = np.reshape(data, [len(temp2), int(columns)])
        file.close()
        return data


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


def setupperbutton(opts):
    """

    :param opts:
    """
    global root
    global rhs_ecg_frame
    global plt_options
    global pltmenu
    global lower_RHS
    global plot_num
    if lower_RHS is not None:
        lower_RHS.destroy()

    lower_RHS = Frame(rhs_ecg_frame, bg='white smoke')
    lower_RHS.pack(side='bottom', fill=BOTH, expand=False)
    # Was root.plot_num = we will see if this still works
    plot_num = StringVar(lower_RHS)
    plt_options = opts
    pltmenu = OptionMenu(lower_RHS, plot_num, plt_options[0], *plt_options)
    pltmenu.config(width=4)
    pltmenu.pack(side='bottom')
    plot_num.trace('w', change_dropdown1)
    Label2(lower_RHS, text='Input', style='Text2.TLabel').pack(side='bottom', anchor='w')


def change_dropdown1():
    """
    Change Dropdown
    """
    global enable2, plt_options, load_dat, dat, x, xminn, tcol, plot_num
    arg = plot_num.get()

    enable2 ^= 1
    if enable2:
        for counter in range(len(plt_options)):
            if arg == plt_options[counter]:
                if tcol == 1:
                    counter = counter + 1
                dat = load_dat[:, counter]
                dat = np.reshape(dat, [len(dat), 1])
                x = np.arange(len(dat))
                Prediction_mode(1)
                # draw1(x, xminn, Fs, dat, plot_fig, graphCanvas, fig, R_t, R_amp, loaded_ann, labelled_flag,
                #    True_R_t, True_R_amp, disp_length, plot_pred, plot_ann, fig2, plot_fig2, RR_interval_Canvas, lab)
                break


def Link_hover_popup_tips(widget, text):
    """

    :param widget:
    :param text:
    """
    if button_help_on:
        toolTip = Pop_up(widget)

        def _enter(event):
            toolTip.showpopup(text)

        def _leave(event):
            toolTip.hidepopup()

        widget.bind('<Enter>', _enter)
        widget.bind('<Leave>', _leave)


def Link_hover_popup_tips2(widget, text):
    """

    :param widget:
    :param text:
    """
    if button_help_on:
        toolTip = Pop_up2(widget)

        def _enter(event):
            toolTip.showpopup(text)

        def _leave(event):
            toolTip.hidepopup()

        widget.bind('<Enter>', _enter)
        widget.bind('<Leave>', _leave)


# FILTER PARAMETERS

# ~~~~~~~~~~~~~~ WINDOWS - MAIN RR-APET PROGRAM ~~~~~~~~~~~~~~~~~~~#

class RRAPET(Frame):
    """
    RR-APET is a GUI class for the estimation and analysis of HRV.
    """

    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.options = None
        self.RRI_variable = None
        self._job = None
        self.parent = parent
        self.mets = None
        self.plot_wind = None
        self.ftypes = [('All files', '*'), ('Text files', '*.txt'), ('HDF5 files', '*.h5'), ('MAT files', '*.mat'),
                       ('WFDB files', '*.dat')]
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
        """
        GUI initialisation
        """
        self.parent.bind('<Control-a>', self._onMetrics)
        self.parent.bind('<Control-f>', self._onPref)
        self.parent.bind('<Control-o>', self._onOpen)
        self.parent.bind('<Control-l>', self._onLoad)
        self.parent.bind('<Control-s>', self._cntrls)
        self.parent.bind('<Control-b>', self._cntrlb)
        self.parent.bind('<Control-u>', self._cntrlu)
        # self.parent.bind('<Control-m>', exp.savemetrics(R_t, loaded_ann, labelled_flag, Fs))
        self.parent.bind('<Control-q>', self._onClose)
        self.parent.bind('<Control-i>', Invert)
        self.parent.bind('<Control-d>', self.__InvertDelete)

        #        self.parent.bind('<Control-p>', self.onContactUs)
        self.parent.bind('<Control-h>', self._onHelp)  # HELP

        self.parent.title("RR APET")
        self.pack(fill=BOTH, expand=1)

        # SET UP MENUBAR#
        menubar = Menu(self.parent, font=cust_header)
        self.parent.config(menu=menubar)

        fileMenu = Menu(menubar, font=cust_subheadernb, tearoff=False)
        fileMenu.add_command(label="Open", command=self._onOpen)  # ctrl+o
        fileMenu.add_command(label="Load Annotations", command=self._onLoad)  # ctrl+l
        fileMenu.add_command(label="Update Annotations", command=lambda: self._onSave(1))  # ctrl+u
        fileMenu.add_command(label="Save Annotations", command=lambda: self._onSave(2))  # ctrl+a
        # fileMenu.add_command(label="Save HRV Metrics", command=exp.savemetrics(R_t, loaded_ann, labelled_flag, Fs))
        fileMenu.add_command(label="Close", command=self._onClose)  # ctrl+q
        fileMenu.add_command(label="Quit", command=self.__shut)  # esc
        menubar.add_cascade(label="File", menu=fileMenu, font=cust_header)

        toolMenu = Menu(menubar, font=cust_subheadernb, tearoff=False)
        toolMenu.add_command(label="Preferences", command=self._onPref)  # ctrl+f
        toolMenu.add_command(label="Generate HRV metrics", command=self._onMetrics)  # ctrl+m
        toolMenu.add_command(label="Batch Save", command=lambda: self._onSave(3))
        #        toolMenu.add_command(label="Convert to PDF", command=self.fakeCommand) #ctrl+m
        menubar.add_cascade(label="Tools", menu=toolMenu, font=cust_header)

        helpMenu = Menu(menubar, font=cust_subheadernb, tearoff=False)
        helpMenu.add_command(label="RR-APET User Guide", command=self._onHelp)
        helpMenu.add_command(label="Contact Us", command=self._onContactUs)
        menubar.add_cascade(label="Help", menu=helpMenu, font=cust_header)

        # GET THIS AS A LOAD IN VALUE FROM PREFRENCCES AND CONNECT TO PREFERENCES WINDOW
        # CHECK FOR ECG OR RRI!!!!

        if ECG_pref_on:
            self.LAUNCH_ECG()
        else:
            self.LAUNCH_RRI()

    def LAUNCH_RRI(self):
        """
        Launch RRI viewer
        """
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

        if TOTAL_FRAME is not None:
            TOTAL_FRAME.destroy()

        # SET UP SUB FRAMES FOR STORAGE OF EACH COMPONENT#
        TOTAL_FRAME = Frame(self, bg='light grey')
        TOTAL_FRAME.pack(side=TOP, fill=BOTH, expand=True)

        tsbh = Frame(TOTAL_FRAME, bg='light grey', relief='raised', height=10)
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
        Link_hover_popup_tips2(editbut, text='Edit annotations. \nLeft click to add positive peak.\nRight click to '
                                             'add negative peak.\nMouse scroll wheel click to \nremove closest peak.')

        # TOP of screen control buttons
        openbutton = Button2(tsbh, image=self.openicon, compound='center', command=self._onOpen, style='C.TButton',
                             takefocus=False)
        openbutton.pack(side='left')
        Link_hover_popup_tips(openbutton, text='Open file (ctrl+O)')

        save1 = Button2(tsbh, image=self.saveicon2, compound='center', command=lambda: self._onSave(2),
                        style='C.TButton', takefocus=False)
        save1.pack(side='left')
        Link_hover_popup_tips(save1, text='Save annotations (ctrl+S)')

        save2 = Button2(tsbh, image=self.updateicon, compound='center', command=lambda: self._onSave(1),
                        style='C.TButton', takefocus=False)
        save2.pack(side='left')
        Link_hover_popup_tips(save2, text='Update annotations (ctrl+U)')

        save4 = Button2(tsbh, image=self.batch, compound='center', command=lambda: self._onSave(3), style='C.TButton',
                        takefocus=False)
        save4.pack(side='left')
        Link_hover_popup_tips(save4, text='Batch Metrics (ctrl+B)')

        Label(tsbh, text='', width=1, bg='light grey', takefocus=False).pack(side='left')
        genmets = Button2(tsbh, image=self.calcicon, compound='center', command=self._onMetrics, style='C.TButton',
                          takefocus=False)
        genmets.pack(side='left')
        Link_hover_popup_tips(genmets, text='Calculate HRV metrics (ctrl+A)')
        #
        # save3 = Button2(tsbh, image=self.saveicon, compound='center',
        #                 command=exp.savemetrics(R_t, loaded_ann, labelled_flag, Fs),
        #                 style='C.TButton', takefocus=False)
        # save3.pack(side='left')
        # Link_hover_popup_tips(save3, text='Save metrics (ctrl+M)')

        prefs = Button2(tsbh, image=self.settingsicon, compound='center', command=self._onPref, style='C.TButton',
                        takefocus=False)
        prefs.pack(side='left')
        Link_hover_popup_tips(prefs, text='Preferences (ctrl+F)')

        helps = Button2(tsbh, image=self.helpicon, compound='center', command=self._onHelp, style='C.TButton',
                        takefocus=False)
        helps.pack(side='left')
        Link_hover_popup_tips(helps, text='Help (ctrl+H)')

        Label(tsbh, text='', width=1, bg='light grey', takefocus=False).pack(side='left')
        closebutton = Button2(tsbh, image=self.closeicon, compound='center', command=self._onClose, style='C.TButton',
                              takefocus=False)
        closebutton.pack(side='left')
        Link_hover_popup_tips(closebutton, text='Close file (ctrl+Q)')

        fig2 = Figure(tight_layout=1, facecolor='#f5f5f5')
        plot_fig2 = fig2.add_subplot(111)

        RR_interval_Canvas = FigureCanvasTkAgg(fig2, master=RRI_plot_housing)
        RR_interval_Canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)
        # RR_interval_Canvas._tkcanvas ??

    #        RR_interval_Canvas.mpl_connect('button_press_event', onclick2)
    #
    #        Slider = Scale(Slider_housing, from_=0, to=100, resolution=0.001, showvalue=0, orient=TKc.HORIZONTAL,
    #        bg='white', command=self.getSlideValue, length=screen_width-0.05*screen_width)
    #        Slider.pack(side=BOTTOM)
    #
    #        t= Entry(Slider_housing, width = 10, readonlybackground='white')
    #        t.pack(side=RIGHT, anchor='e')
    #        t.insert(0, Preferences[2])
    #        t.bind("<FocusIn>", self.callback)
    #        t.bind("<FocusOut>", self.updateRange)

    #        Label2(Slider_housing, text="Range (s)", style='Text2.TLabel').pack(side='right', anchor='e')

    def LAUNCH_ECG(self):
        """
        Launch Dataframe with ECG
        """
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

        if TOTAL_FRAME is not None:
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
        openbutton = Button2(tsbh, image=self.openicon, compound='center', command=self._onOpen, style='C.TButton',
                             takefocus=False)
        openbutton.pack(side='left')
        Link_hover_popup_tips(openbutton, text='Open file (ctrl+O)')

        save1 = Button2(tsbh, image=self.saveicon2, compound='center', command=lambda: self._onSave(2),
                        style='C.TButton', takefocus=False)
        save1.pack(side='left')
        Link_hover_popup_tips(save1, text='Save annotations (ctrl+S)')

        save2 = Button2(tsbh, image=self.updateicon, compound='center', command=lambda: self._onSave(1),
                        style='C.TButton', takefocus=False)
        save2.pack(side='left')
        Link_hover_popup_tips(save2, text='Update annotations (ctrl+U)')

        save4 = Button2(tsbh, image=self.batch, compound='center', command=lambda: self._onSave(3), style='C.TButton',
                        takefocus=False)
        save4.pack(side='left')
        Link_hover_popup_tips(save4, text='Batch Metrics (ctrl+B)')

        #        printer=Button2(tsbh, image=self.printicon, compound='center', command=lambda:self.fakeCommand,
        #        style='C.TButton', takefocus=False)
        #        printer.pack(side='left')
        #        Link_hover_popup_tips(printer, text = 'Convert to PDF (ctrl+P)')

        Label(tsbh, text='', width=1, bg='light grey', takefocus=False).pack(side='left')
        genmets = Button2(tsbh, image=self.calcicon, compound='center', command=self._onMetrics, style='C.TButton',
                          takefocus=False)
        genmets.pack(side='left')
        Link_hover_popup_tips(genmets, text='Calculate HRV metrics (ctrl+A)')

        # save3 = Button2(tsbh, image=self.saveicon, compound='center',
        #                 command=exp.savemetrics(R_t, loaded_ann, labelled_flag, Fs), style='C.TButton',
        #                 takefocus=False)
        # save3.pack(side='left')
        # Link_hover_popup_tips(save3, text='Save metrics (ctrl+M)')

        prefs = Button2(tsbh, image=self.settingsicon, compound='center', command=self._onPref, style='C.TButton',
                        takefocus=False)
        prefs.pack(side='left')
        Link_hover_popup_tips(prefs, text='Preferences (ctrl+F)')

        helps = Button2(tsbh, image=self.helpicon, compound='center', command=self._onHelp, style='C.TButton',
                        takefocus=False)
        helps.pack(side='left')
        Link_hover_popup_tips(helps, text='Help (ctrl+H)')

        Label(tsbh, text='', width=1, bg='light grey', takefocus=False).pack(side='left')
        closebutton = Button2(tsbh, image=self.closeicon, compound='center', command=self._onClose, style='C.TButton',
                              takefocus=False)
        closebutton.pack(side='left')
        Link_hover_popup_tips(closebutton, text='Close file (ctrl+Q)')

        # ECG related buttons
        editbut = Button2(upper, image=self.editicon, compound='center', command=edit_toggle, style='B.TButton',
                          takefocus=False)
        editbut.pack(side='top', anchor='w')
        Link_hover_popup_tips2(editbut, text='Edit annotations. \nLeft click to add positive peak.\nRight click to '
                                             'add negative peak.\nMouse scroll wheel click to \nremove closest peak.')

        # Menu for RRI plot
        self.RRI_variable = StringVar(lower)
        self.options = ['RR_P', 'RR_A', 'RR_B']
        RRImenu = OptionMenu(lower, self.RRI_variable, self.options[0], *self.options)
        RRImenu.config(width=5)
        #        RRImenu.configure(compound='right',image=self.photo)
        RRImenu.pack(side='top')
        self.RRI_variable.trace('w', self._change_dropdown)

        fig = Figure(tight_layout=1,
                     facecolor='#f5f5f5')  # Configuration of the Figure to be plotted on the first canvas
        plot_fig = fig.add_subplot(111)  # Adding a subplot which can be updated for viewing of ECG and R-peaks

        graphCanvas = FigureCanvasTkAgg(fig, master=ECG_plot_housing)  # Bind the figure to the canvas
        graphCanvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)  # Positioning of figure within canvas window
        # Ensuring that "graphCanvas" can be bound to mouse input later on
        # graphCanvas._tkcanvas  # .pack(side=TOP, fill=BOTH, expand=True)

        fig2 = Figure(tight_layout=1, facecolor='#f5f5f5')
        plot_fig2 = fig2.add_subplot(111)

        RR_interval_Canvas = FigureCanvasTkAgg(fig2, master=RRI_housing)
        RR_interval_Canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)
        # RR_interval_Canvas._tkcanvas
        RR_interval_Canvas.mpl_connect('button_press_event', onclick2)

        Slider = Scale(Slider_housing, from_=0, to=100, resolution=0.001, showvalue=0, orient=TKc.HORIZONTAL,
                       bg='white', command=self._getSlideValue, length=screen_width - 0.05 * screen_width)
        Slider.pack(side=BOTTOM)

        t = Entry(Slider_housing, width=10, readonlybackground='white')
        t.pack(side=RIGHT, anchor='e')
        t.insert(0, Preferences[2])
        t.bind("<FocusIn>", self._callback)
        t.bind("<FocusOut>", self.updateRange)

        Label2(Slider_housing, text="Range (s)", style='Text2.TLabel').pack(side='right', anchor='e')

    @staticmethod
    def _callbackFunc(event):
        print("New Element Selected")

    def _change_dropdown(self, *args):
        global enable, plot_pred, plot_ann, loaded_ann
        arg = self.RRI_variable.get()

        enable ^= 1
        if enable:
            if arg == 'RR_P':
                plot_pred = 1
                plot_ann = 0
                draw1(x, xminn, Fs, dat, plot_fig, graphCanvas, fig, R_t, R_amp, loaded_ann, labelled_flag,
                      True_R_t, True_R_amp, disp_length, plot_pred, plot_ann, fig2, plot_fig2, RR_interval_Canvas)
            elif loaded_ann == 1:
                if arg == 'RR_A':
                    plot_pred = 0
                    plot_ann = 1
                    draw1(x, xminn, Fs, dat, plot_fig, graphCanvas, fig, R_t, R_amp, loaded_ann, labelled_flag,
                          True_R_t, True_R_amp, disp_length, plot_pred, plot_ann, fig2, plot_fig2, RR_interval_Canvas)
                elif arg == 'RR_B':
                    plot_pred = 1
                    plot_ann = 1
                    draw1(x, xminn, Fs, dat, plot_fig, graphCanvas, fig, R_t, R_amp, loaded_ann, labelled_flag,
                          True_R_t, True_R_amp, disp_length, plot_pred, plot_ann, fig2, plot_fig2, RR_interval_Canvas)
            else:
                messagebox.showwarning("Warning", "External annotations have not been loaded into the program. \n\n"
                                                  "RR_Annotations and RR_Both are invalid selections - please load "
                                                  "annotations before selecting these options.")
                plot_pred = 1
                plot_ann = 0
                self.RRI_variable.set(self.options[0])

    def _callback(self, event):
        global t
        t.config(insertontime=600, state='normal')
        t.bind("<Return>", self.updateRange)
        t.bind("<KP_Enter>", self.updateRange)

    @staticmethod
    def updateRange(event):
        """
        Update Graph Range
        :param event:
        """
        global t, disp_length
        t.config(insertontime=0, state='readonly')
        t.unbind("<Return>")
        t.unbind("<KP_Enter>")
        newdisp = int(t.get())
        if newdisp != disp_length:
            disp_length = newdisp
            draw1(x, xminn, Fs, dat, plot_fig, graphCanvas, fig, R_t, R_amp, loaded_ann, labelled_flag, True_R_t,
                  True_R_amp, disp_length, plot_pred, plot_ann, fig2, plot_fig2, RR_interval_Canvas)

    def _getSlideValue(self, event):
        if self._job:
            self.after_cancel(self._job)
        self._job = self.after(100, self._onSlide)

    def _onSlide(self):
        global x, dat, xminn, Fs, plot_fig, graphCanvas, Slider, disp_length
        self._job = None
        tmp = int(((len(dat) / Fs) * Slider.get() / 100) - 0.5 * disp_length)
        if tmp < 0:
            tmp = 0

        xminn = tmp * Fs
        draw1(x, xminn, Fs, dat, plot_fig, graphCanvas, fig, R_t, R_amp, loaded_ann, labelled_flag, True_R_t,
              True_R_amp, disp_length, plot_pred, plot_ann, fig2, plot_fig2, RR_interval_Canvas)

    def _onClose(self, event=1):
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

        if self.mets is not None:
            self.mets.withdraw()
        self.mets = None

        if self.plot_wind is not None:
            self.plot_wind.withdraw()
        self.plot_wind = None

        P = Preferences
        ECG_pref_on = int(P[21])

        if ECG_pref_on:
            self.LAUNCH_ECG()
        else:
            self.LAUNCH_RRI()

    def _onOpen(self, event=1):
        global dat
        global x
        global xminn
        global Fs
        global tcol
        global load_dat
        global pref
        global ECG_pref_on
        global R_t
        global labelled_flag
        global Preferences

        if ECG_pref_on:
            input_file = filedialog.Open(self, filetypes=self.ftypes)
            File = input_file.show()
            filename, file_extension = os.path.splitext(File)
            columns = None
            if pref is not None:
                pref.withdraw()

            if self.plot_wind is not None:
                self.plot_wind.destroy()

            if file_extension == '.txt':
                load_dat = read_txt_file(fname=File)
                load_dat, rows, columns = check_orientation(load_dat)
                check_for_fs_col()

            elif (file_extension == '.h5') or (file_extension == '.mat'):
                global h5window
                h5window = Toplevel()
                h5sel = H5_Selector(h5window, Preferences, (File, 'data'))
                if windows_compile:
                    h5window.bind('<Escape>', self.__close_h5win)
                else:
                    h5window.bind('<Control-Escape>', self.__close_h5win)

                self.parent.wait_window(h5window)
                load_h5(h5sel.string, 'data', h5sel.folder[0], file_extension)

            elif warnings_on:
                messagebox.showwarning("Warning", "The file type you have selected in not compatible. "
                                                  "Please select a *.txt, *.h5, *.mat, or *.dat file only.")

        else:
            input_file = filedialog.Open(self, filetypes=self.ftypes)
            File = input_file.show()
            filename, file_extension = os.path.splitext(File)
            go_ahead = 1

            if pref is not None:
                pref.withdraw()

            if self.plot_wind is not None:
                self.plot_wind.destroy()

            column_number = 1
            if file_extension == '.txt':
                og_ann = read_txt_file(fname=File)
                R_t = og_ann[:, (column_number - 1)]

            elif (file_extension == '.h5') or (file_extension == '.mat'):
                global h5window2
                h5window2 = Toplevel()
                H5_Selector(h5window2, Preferences, (File, 'data'))
                if windows_compile:
                    h5window2.bind('<Escape>', self.__close_h5win2)
                if linux_compile:
                    h5window2.bind('<Control-Escape>', self.__close_h5win2)
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

            elif warnings_on:
                messagebox.showwarning("Warning", "The file type you have selected in not compatible. "
                                                  "Please select a *.txt, *.h5, *.mat, or *.atr file only.")
                go_ahead = 0

            if go_ahead:
                # DETERMINE IF DATA INCLUDES TIME STAMPS OR NOT
                R_t = R_t[R_t != 0]
                R_t = np.reshape(R_t, [len(R_t), ])

                if not (np.diff(R_t[:]) > 0).all():
                    # They aren't all greater than the previous - therefore RRI series not time-stamps
                    tmp = np.zeros(np.size(R_t))
                    tmp[0] = R_t[0]

                    for i in range(1, np.size(R_t)):
                        tmp[i] = tmp[i - 1] + R_t[i]

                if Preferences[23] == '1':
                    R_t = R_t / 1e3
                R_t = np.reshape(R_t, [len(R_t), 1])
                labelled_flag = 1
                onNoFSdata()

    def _onLoad(self, event=1):
        global dat
        global xminn
        global Fs
        global loaded_ann
        global True_R_t
        global True_R_amp
        global Preferences

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
                og_ann = read_txt_file(annfile)
                True_R_t = og_ann[:, (column_number - 1)]

                if Preferences[23] == '1':
                    True_R_t = True_R_t / 1e3
                # Average time interval between heart beats wouldn't be less than 10bpm, so a gap of over 6
                # seconds on average or greater means loaded in as samples
                if np.mean(np.diff(True_R_t)) < 6:
                    True_R_t = True_R_t * Fs  # Measured in time but need samples

            elif (file_extension == '.h5') or (file_extension == '.mat'):
                global h5window
                h5window = Toplevel()
                H5_Selector(h5window, Preferences, (annfile, 'data'))
                if windows_compile:
                    h5window.bind('<Escape>', self.__close_h5win)
                if linux_compile:
                    h5window.bind('<Control-Escape>', self.__close_h5win)
                go_ahead = 0

            elif file_extension == '.atr':
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
                messagebox.showwarning("Warning", "The file type you have selected in not compatible. "
                                                  "Please select a *.txt, *.h5, *.mat, or *.atr file only.")
                go_ahead = 0
            #
            if go_ahead:
                # DETERMINE IF DATA INCLUDES TIME STAMPS OR NOT
                True_R_t = True_R_t[True_R_t != 0]
                True_R_t = np.reshape(True_R_t, [len(True_R_t), ])

                if not (np.diff(True_R_t[:]) > 0).all():
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

    def _cntrls(self, event=1):
        self._onSave(2)

    def _cntrlu(self, event=1):
        self._onSave(1)

    def _cntrlb(self, event=1):
        self._onSave(3)

    @staticmethod
    def _onSave(stype):
        """
        Saving Outputs
        :param stype:
        """
        global R_amp
        global R_t
        global True_R_amp
        global True_R_t
        global ECG_pref_on

        if ECG_pref_on:
            if stype == 1:
                stype = 2

        if stype == 1:  # UPDATES CHANGES TO A LOADED ANNOTATION FILE
            if windows_compile:
                saveroot = filedialog.asksaveasfilename(title="Select file", defaultextension=".*",
                                                        filetypes=(("text files", "*.txt"), ("all files", "*.*")))
            else:
                saveroot = filedialog.asksaveasfilename(title="Select file",
                                                        filetypes=(("text files", "*.txt"), ("all files", "*.*")))
            fname, file_extension = os.path.splitext(saveroot)
            if loaded_ann == 1:
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
                    messagebox.showwarning("Warning", "Annotations were not updated! \n\nPlease Note: "
                                                      "Currently RR-APET cannot export the file type selected")
            elif warnings_on:
                messagebox.showwarning("Warning", "Annotations were not updated! \n\nPlease Note: Updating "
                                                  "annotations requires an imported annotation file (red data-points "
                                                  "on RRI plot) that has been edited. The 'Save Annotations' option "
                                                  "allows annotations that were generated within the program "
                                                  "(blue data-points on RRI plot) to be saved.")

        elif stype == 2:
            if windows_compile:
                saveroot = filedialog.asksaveasfilename(title="Select file", defaultextension=".*",
                                                        filetypes=(("text files", "*.txt"), ("all files", "*.*")))
            else:
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

        # elif stype == 3:
        #     global PATH
        #     global fs_frame_2
        #     input_file = filedialog.askdirectory()
        #     PATH = input_file
        #     fs_frame_2 = Toplevel()
        #     File_type_and_Fs(fs_frame_2)
        #     if windows_compile:
        #         fs_frame_2.bind('<Escape>', self.__close_fs2)
        #     if linux_compile:
        #         fs_frame_2.bind('<Control-Escape>', self.__close_fs2)

    def _onPref(self, event=1):
        global pref
        pref = Toplevel()
        # UserPreferences(pref)
        if windows_compile:
            pref.bind('<Escape>', self.__close_pref)
        if linux_compile:
            pref.bind('<Control-Escape>', self.__close_pref)

    def _onMetrics(self, event=1):
        self.mets = Toplevel()
        # HRVstatics(mets)
        if windows_compile:
            self.mets.bind('<Escape>', self.__close_mets)
        if linux_compile:
            self.mets.bind('<Control-Escape>', self.__close_mets)

    def _onHelp(self, event=1):
        global helper
        helper = Toplevel()
        # HELP(helper)
        if windows_compile:
            helper.bind('<Escape>', self.__close_helper)
        if linux_compile:
            helper.bind('<Control-Escape>', self.__close_helper)

    def _onContactUs(self, event=1):
        global contact
        contact = Toplevel()
        # Contact_Us(contact)
        if windows_compile:
            contact.bind('<Escape>', self.__close_contact)
        if linux_compile:
            contact.bind('<Control-Escape>', self.__close_contact)

    @staticmethod
    def _readFile(filename):

        f = open(filename, "r")
        text = f.read()
        return text

    @staticmethod
    def __InvertDelete():
        global delete_flag
        delete_flag ^= 1
        if delete_flag & warnings_on:
            messagebox.showwarning("Warning",
                                   "Inverted peak add/delete option selected.\n\nMouse left click now deletes "
                                   "closest peak and mouse wheel button adds peak (polarity determined by "
                                   "inversion option).")
        elif warnings_on & invert_flag != 1:
            messagebox.showwarning("Warning", "Reverted to normal conditions. Mouse left click now adds peak and mouse "
                                              "wheel button removes peak (polarity determined by inversion option).")

    def __shut(self):
        if self.mets is not None:
            self.mets.destroy()
        if self.plot_wind is not None:
            self.plot_wind.destroy()
        root.destroy()
        # os._exit(1)

    @staticmethod
    def __close_pref(event):
        pref.withdraw()

    @staticmethod
    def __close_fs2(event):
        fs_frame_2.withdraw()

    @staticmethod
    def __close_h5win(event):
        h5window.withdraw()

    @staticmethod
    def __close_h5win2(event):
        h5window2.withdraw()

    def __close_mets(self, event):
        self.mets.withdraw()

    @staticmethod
    def __close_helper(event):
        helper.withdraw()

    @staticmethod
    def __close_contact(event):
        contact.withdraw()


class Pop_up(object):
    """
    Pop up class 1
    """

    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def showpopup(self, text):
        """
        Show Pop-up
        :param text:
        """
        xx = root.winfo_pointerx()
        yy = root.winfo_pointery() + 10
        self.tipwindow = Toplevel(self.widget)
        Label(self.tipwindow, text=text, font='Helvetica 8', bg='light blue').pack()
        self.tipwindow.wm_geometry("+%d+%d" % (xx, yy))
        self.tipwindow.overrideredirect(True)

    def hidepopup(self):
        """
        Hide Pop-Up
        :return:
        """
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()


class Pop_up2(object):
    """
    Pop Up Class 2
    """

    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def showpopup(self, text):
        """
        Show Pop Up
        :param text:
        """
        xx = root.winfo_pointerx() - 150
        yy = root.winfo_pointery() + 10
        self.tipwindow = Toplevel(self.widget)
        Label(self.tipwindow, text=text, font='Helvetica 8', bg='light blue').pack()
        self.tipwindow.wm_geometry("+%d+%d" % (xx, yy))
        self.tipwindow.overrideredirect(True)

    def hidepopup(self):
        """
        Hide Pop-Up
        """
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()


# LAUNCH SCRIPT#
RRAPET(root)
# ~~~~~~~~~~~ Keys bound to operating window ~~~~~~~~~~#
if windows_compile:
    root.bind('<Escape>', shut)
if linux_compile:
    root.bind('<Control-Escape>', shut)
root.bind('Control-i', Invert)
root.style.configure('Header.TLabel', background='white', padding=0, highlightthickness=0, font=cust_header)
root.style.configure('SubHeader.TLabel', background='white', padding=0, highlightthickness=0, font=cust_subheader)
root.style.configure('Text.TLabel', background='white', padding=0, highlightthickness=0, font=cust_text)
root.style.configure('UserPref.TButton', background='white', padding=0, relief='flat', width=25, anchor='w',
                     highlightthickness=0, font=cust_subheadernb)
root.style.configure('SelectUserPref.TButton', background='grey', padding=0, relief='flat', width=25, anchor='w',
                     highlightthickness=0, font=cust_subheadernb)
root.style.configure('Text2.TLabel', background='white smoke', width=8, anchor='w', padding=0, takefocus=False,
                     highlightthickness=0, font=cust_text)
root.style.configure('Text.TMenubutton', font=cust_text, width=12)

root.mainloop()
