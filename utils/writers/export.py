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

    def __init__(self, preferences, system: str = 'linux'):
        self.system = system
        self.pref = preferences

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

    def gen_metrics_for_save(self, rpeaks, fs, t_rpeaks: np.ndarray = None, td: tuple = None, rqa: tuple = None,
                             dfa: tuple = None, pc: tuple = None):
        """
        Generate the metrics if not passed to save file
        """
        Rpeakss = np.reshape(rpeaks, (len(rpeaks),))

        # Time-domain Statistics
        if td is None:
            SDNN, SDANN, MeanRR, RMSSD, pNN50 = Calculate_Features(Rpeakss, fs)
        else:
            SDNN, SDANN, MeanRR, RMSSD, pNN50 = td

        # Frequency-domain Statistics
        Rpeak_input = Rpeakss / fs
        freq_results = []
        for i in range(4):
            freq_results.append([Freq_Analysis(Rpeak_input, meth=i, decim=3, M=self.pref['values']['welch_L'],
                                               O=self.pref['values']['welch_O'],
                                               BTval=self.pref['values']['bltk_input'],
                                               omega_max=self.pref['values']['ls_omega_max'],
                                               order=self.pref['values']['ar_order'])])

        # Nonlinear statistics
        RRI = np.diff(Rpeak_input)
        if rqa is None:
            REC, DET, LAM, Lmean, Lmax, Vmean, Vmax = RQA(RRI, m=int(self.pref['values']['rqa_m']),
                                                          L=int(self.pref['values']['rqa_l']))
        else:
            REC, DET, LAM, Lmean, Lmax, Vmean, Vmax = rqa

        if pc is None:
            SD1, SD2, c1, c2 = Poincare(RRI)
        else:
            SD1, SD2, c1, c2 = pc

        if dfa is None:
            alp1, alp2, F = DFA(RRI, min_box=self.pref['values']['minbox'],  max_box=self.pref['values']['maxbox'],
                                cop=self.pref['values']['copbox'], inc=self.pref['values']['increm'], decim=3)
        else:
            alp1, alp2, F = dfa

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

        if t_rpeaks is not None:
            TP, FP, FN = test2(rpeaks, t_rpeaks, self.pref['values']['tt'])
            Se, PP, ACC, DER = acc2(TP, FP, FN)
            output['Detection Accuracy'] = {
                'Se': Se,
                'Pp': PP,
                'Acc': ACC,
                'Der': DER
            }

        return output

    def savemetrics(self, rpeaks, fs, t_rpeaks: np.ndarray = None, td: tuple = None, rqa: tuple = None,
                    dfa: tuple = None, pc: tuple = None):
        """

        :param rpeaks: R-peak timestamps.
        :param fs: Sampling Frequency.
        :param t_rpeaks: Optional. True R-peak timestamps. If provided predicted accuracy will be calculated.
        :param td: Optional. Pre-calculated time domain results
        :param rqa: Optional. Prec-calculated RQA results
        :param dfa: Optional. Prec-calculated DFA results
        :param pc: Optional. Prec-calculated Poincare results
        :return:
        """
        output = self.gen_metrics_for_save(rpeaks, fs, t_rpeaks, td=td, rqa=rqa, dfa=dfa, pc=pc)

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
                if t_rpeaks is not None:
                    text_file.write('Quantified HRV Metrics and R-peak detection method analysis \n\n')
                else:
                    text_file.write('Quantified HRV Metrics \n\n')
                text_file.write(json.dumps(output, indent=4))

        elif file_extension == '.mat':
            sio.savemat(saveroot, output)

        else:
            print('Cannot export this file type')
