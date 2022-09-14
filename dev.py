"""
Dev env for Random Code

"""
from utils.HRV_Statistics import HRVstatics
from tkinter import Tk, Label
import tkinter.constants as TKc


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
        'ECG_pref_on': True,
        'labelled_flag': False,
        'warnings_on': True,
        'tt': 4
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


root = Tk()

pairs = [["SDNN (ms)", 1], ["SDANN (ms)", 3], ["Mean RR interval (ms)", 5],
         ["RMSSD (ms)", 2], ["pNN50 (%)", 4]]


def add_paired_label(dframe, txt, value, row):
    Label(dframe, text=txt, anchor=TKc.W, width=20).grid(row=row, column=0)
    Label(dframe, text=value, anchor=TKc.W, width=int(20 / 2)).grid(row=row, column=1)


for i, dat in enumerate(pairs):
    add_paired_label(root, dat[0], dat[1], i)

root.mainloop()
