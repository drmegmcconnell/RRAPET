"""
Export RR-APET data

Author: Meghan McConnell
"""
import os
import json
from tkinter import filedialog
from utils.HRV_Functions import *
import tables as tb
import scipy.io as sio


class TimeDomain(tb.IsDescription):
    """
    Format of Time Domain Metrics
    """
    SDNN = tb.FloatCol(pos=1)
    SDANN = tb.FloatCol(pos=2)
    MeanRR = tb.FloatCol(pos=3)
    RMSSD = tb.FloatCol(pos=4)
    pNN50 = tb.FloatCol(pos=5)


class FrequencyDomain(tb.IsDescription):
    """
    Format of Frequency Domain Metrics
    """
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
    """
    Format of NonLinear Metrics
    """
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


class Exporter:
    """
    Utility for exportation of RR-APET data
    """

    def __init__(self, Preferences, system: str = 'linux'):
        self.system = system
        self.Preferences = Preferences

    def savefigure(self, draw_figure):
        """
        Save any figure
        """
        if self.system == 'linux':
            path_for_save = filedialog.asksaveasfilename(title="Select file", filetypes=(
                ("eps", "*.eps"), ("png", "*.png"), ("svg", "*.svg"), ("all files", "*.*")))
        else:
            path_for_save = filedialog.asksaveasfilename(title="Select file", defaultextension=".*", filetypes=(
                ("eps", "*.eps"), ("png", "*.png"), ("svg", "*.svg"), ("all files", "*.*")))

        fname, file_extension = os.path.splitext(path_for_save)

        if file_extension == '.png':
            draw_figure.savefig(path_for_save, format='png', dpi=300)
        elif file_extension == '.svg':
            draw_figure.savefig(path_for_save, format='svg', dpi=300)
        else:
            draw_figure.savefig(path_for_save, format='eps', dpi=300)

    def gen_metrics_for_save(self, R_t, Fs):
        """
        Generate the metrics if not passed to save file
        """
        Rpeakss = np.reshape(R_t, (len(R_t),))

        welchL = float(self.Preferences[5])
        welchO = float(self.Preferences[6])
        btval_input = int(self.Preferences[7])  # 10
        omax_input = int(self.Preferences[8])  # 500
        order = int(self.Preferences[9])  # 10

        # Time-domain Statistics
        SDNN, SDANN, MeanRR, RMSSD, pNN50 = Calculate_Features(Rpeakss, Fs)

        # Frequency-domain Statistics
        Rpeak_input = Rpeakss / Fs
        freq_results = []
        for i in range(4):
            freq_results.append([Freq_Analysis(Rpeak_input, meth=i, decim=3, M=welchL, O=welchO, BTval=btval_input,
                                               omega_max=omax_input, order=order)])

        mbox = int(self.Preferences[10])
        print(mbox)
        COP = int(self.Preferences[11])
        print(COP)
        m2box = int(self.Preferences[12])
        print(m2box)
        In = int(self.Preferences[13])
        print(In)

        # Nonlinear statistics
        RRI = np.diff(Rpeak_input)
        #    (pvp, Min=self.minbox, Max=self.maxbox, Inc=self.increm, COP=self.copbox)
        REC, DET, LAM, Lmean, Lmax, Vmean, Vmax = RQA(RRI, m=int(self.Preferences[14]), L=int(self.Preferences[15]))
        SD1, SD2, c1, c2 = Poincare(RRI)
        alp1, alp2, F = DFA(RRI, min_box=mbox, max_box=m2box, cop=COP, inc=In, decim=3)

        output = {
            'Time-Domain Features': {
                'SDNN': SDNN,
                'SDANN': SDANN,
                'MeanRR': MeanRR,
                'RMSSD': RMSSD,
                'pNN50': pNN50
            },
            'Freq-Domain Features': {
                'Welch': freq_results[0],
                'Blackman-Tukey': freq_results[1],
                'Lombscaragle': freq_results[2],
                'LPC': freq_results[3]
            },
            'Nonlinear Feaures': {
                'REC': REC,
                'DET': DET,
                'LAM': LAM,
                'Lmean': Lmean,
                'Lmax': Lmax,
                'Vmean': Vmean,
                'Vmax': Vmax,
                'SD1': SD1,
                'SD2': SD2,
                'alp1': alp1,
                'alp2': alp2
            }
        }

        return output

    def savemetrics(self, R_t, loaded_ann, labelled_flag, Fs):   # True_R_t, tt
        """

        :param R_t:
        :param loaded_ann:
        :param labelled_flag:
        :param Fs:
        :return:
        """

        # if loaded_ann == 1:
        #     TP, FP, FN = test2(R_t, True_R_t, tt)
        #     Se, PP, ACC, DER = acc2(TP, FP, FN)

        output = self.gen_metrics_for_save(R_t, Fs)

        if self.system == 'linux':
            saveroot = filedialog.asksaveasfilename(title="Select file", defaultextension=".*",
                                                    filetypes=(("text files", "*.txt"), ("all files", "*.*")))
        else:
            saveroot = filedialog.asksaveasfilename(title="Select file",
                                                    filetypes=(("text files", "*.txt"), ("all files", "*.*")))
        fname, file_extension = os.path.splitext(saveroot)

        if file_extension == '.h5':
            fileh = tb.open_file(saveroot, mode='w')
            table = fileh.create_table(fileh.root, 'Time_Domain_Metrics', TimeDomain,
                                       "HRV analysis - Time-Domain metrics")
            table.append([list(output['Time-Domain Features'].values())])

            table2 = fileh.create_table(fileh.root, 'Frequency_Domain_Metrics', FrequencyDomain,
                                        "HRV analysis - Frequency-Domain metrics")
            table2.append([output['Freq-Domain Features']['Welch'],
                           output['Freq-Domain Features']['Blackman-Tukey'],
                           output['Freq-Domain Features']['Lombscaragle'],
                           output['Freq-Domain Features']['LPC']])

            table3 = fileh.create_table(fileh.root, 'Nonlinear_Metrics', NonlinearMets,
                                        "HRV analysis - Nonlinear metrics")
            table3.append([list(output['Nonlinear Features'].values())])

            fileh.close()

        elif file_extension == '.txt':
            with open(saveroot, 'w') as text_file:
                if (labelled_flag == 1) & (loaded_ann == 1):
                    text_file.write('Quantified HRV Metrics and R-peak detection method analysis \n\n')
                else:
                    text_file.write('Quantified HRV Metrics \n\n')
                text_file.write(json.dumps(output, indent=4))

        elif file_extension == '.mat':
            sio.savemat(saveroot, output)

        else:
            print('Cannot export this file type')
