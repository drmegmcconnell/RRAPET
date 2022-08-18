"""
Calculate HRV for Batch Files

"""
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
from utils.HRV_Functions import RQA, DFA, Poincare
from scipy.signal import lombscargle


def Calculate_Features(rloc: np.ndarray, d: int = 2):
    """

    :param rloc:
    :param d: Number of decimal places to return results with
    :return:
    """

    # Calculating SDNN
    R_td = np.diff(rloc)
    MeanRR = np.mean(R_td) * 1e3
    SDNN = np.std(R_td) * 1e3

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

    return np.around(SDNN, d), np.around(MeanRR, d), np.around(RMSSD, d), np.around(pNN50, d)


def Freq_Analysis(Rpeaks, RRI, decim=3, omega_max=500):
    """

    :param Rpeaks:
    :param RRI:
    :param omega_max:
    :param decim:
    :return:
    """

    RRI = RRI - np.mean(RRI)
    omega = np.linspace(0.0001, np.pi * 2, omega_max)
    P_2 = lombscargle(Rpeaks, RRI, omega, normalize=False)
    f = omega / (2 * np.pi)

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


# names = ['RBA_1min_Interpol__Lin1pt', 'RBA_1min_Interpol__Near1pt', 'RBA_2min_Interpol__Lin1pt',
#          'RBA_2min_Interpol__Near2pt', 'RBA_1min_Interpol__Lin3pt', 'RBA_1min_Interpol__Near3pt',
#          'RBA_2min_Interpol__Near1pt', 'RBA_3min_Interpol__Near1pt', 'RBA_1min_Interpol__Lin2pt',
#          'RBA_1min_Interpol__Near2pt', 'RBA_2min_Interpol__Lin2pt', 'RBA_3min_Interpol__Lin1pt']

# for folder in names:
# path = '/home/meg/Desktop/DataRRI/InterpolatedData/' + folder

path = '/home/meg/Desktop/DataRRI/Corrected'

file_names = [f for f in listdir(path) if isfile(join(path, f))]

cols = ['file', 'SDNN (ms)', 'MeanRR (ms)', 'RMSSD (ms)', 'pNN50 (%)', 'REC (%)', 'DET (%)', 'LAM (%)',
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
                                                    *Calculate_Features(r_loc),
                                                    *RQA(rri),
                                                    alp1, alp2,
                                                    sd1, sd2,
                                                    *Freq_Analysis(r_loc, rri))
                                                   ).reshape(1, len(cols)),
                                     columns=cols)))

df = df.reset_index(drop=True)
df.to_csv(path+'_lombscargle.csv', index=False)

print('Processing Complete!')
