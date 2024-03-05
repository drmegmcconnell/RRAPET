"""
Calculate HRV for Batch Files

"""
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
from utils.HRV_Functions import RQA, DFA, Poincare, Calculate_Features, Freq_Analysis

path = '/home/meg/Desktop/DataRRI/Corrected'

file_names = [f for f in listdir(path) if isfile(join(path, f))]

cols = ['file', 'SDNN (ms)', 'SDANN (ms)', 'MeanRR (ms)', 'RMSSD (ms)', 'pNN50 (%)', 'REC (%)', 'DET (%)', 'LAM (%)',
        'Lmean (bts)', 'Lmax (bts)', 'Vmean (bts)', 'Vmax (bts)', 'Alpha1', 'Alpha2', 'SD1', 'SD2', 'ULF_P (s^2)',
        'LF_P (s^2)', 'HF_P (s^2)', 'ULF_P (%)', 'LF_P (%)', 'HF_P (%)', 'peak_freq_ULF (Hz)', 'peak_freq_LF (Hz)',
        'peak_freq_HF (Hz)', 'LFHF']

df = pd.DataFrame(columns=cols)
# powVLF, powLF, powHF, perpowVLF, perpowLF, perpowHF, peak_freq_VLF, peak_freq_LF, peak_freq_HF, LFHF
for fname in file_names:
    data = pd.read_csv(path + '/' + fname, header=None)
    # r_loc = data[0].to_numpy()
    rri = data[1].to_numpy()
    r_loc = np.zeros(len(rri))
    total = 0
    for pos, value in enumerate(rri):
        total += value
        r_loc[pos] = total

    alp1, alp2, _ = DFA(rri)
    sd1, sd2, _, _, _, _ = Poincare(rri)
    # sdnn, meanrr, rmssd, pnn50 = Calculate_Features(r_loc)
    df = pd.concat((df, pd.DataFrame(data=np.array((fname,
                                                    *Calculate_Features(r_peaks=r_loc, fs=1),   # Already in seconds
                                                    *RQA(rri),
                                                    alp1, alp2,
                                                    sd1, sd2,
                                                    *Freq_Analysis(r_peaks=r_loc, meth=3, decim=3, omega_max=500))
                                                   ).reshape(1, len(cols)),
                                     columns=cols)))

df = df.reset_index(drop=True)
df.to_csv(path+'_lombscargle.csv', index=False)

print('Processing Complete!')
