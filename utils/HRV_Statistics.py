# ~~~~~~~~~~~~~~ HRV ~~~~~~~~~~~~~~~~~~~ #
import tkinter.constants as TKc
from tkinter import Frame, FALSE, messagebox, Label, StringVar, OptionMenu, Entry, Button, Toplevel
from utils.Graphical_Functions import *
from utils.Style_Functions import headerStyles
from utils.writers.export import Exporter
from utils.viewers import PlotViewer


class HRVstatics(Frame):
    """
    Calculating HRV Statistics
    """
    def __init__(self, parent, data_dict, preferences):
        Frame.__init__(self, parent)
        self.exp = Exporter(preferences=preferences)
        self.prec = None
        self.frqmeth = None
        self.Outer_freq_dat = None
        self.Rpeak_input = None
        self.DA_dat = None
        self.Outer_DA_dat = None
        self.freq_dat_low = None
        self.parent = parent
        self.pref = preferences
        self.data_dict = data_dict
        self.acc_lbls = ['Sensitivity (%)', 'Positive Predictability (%)', 'Accuracy (%)', 'Detection Error Rate (%)']
        self.poincare_lbls = ['SD1 (ms)', 'SD2 (ms)']
        self.tf_lbls = ['SDNN', 'SDANN', 'MeanRR', 'RMSSD', 'pNN50']
        self.dfa_lbls = []
        self.rqa_lbls = ['DET (%)', 'REC (%)', 'LAM (%)', 'Lmean (bts)', 'Lmax (%)', 'Vmean (bts)', ' Lmax (%)']
        self.text, self.subheader, self.subheadernb, self.header = headerStyles(self.pref['values']['font'],
                                                                                self.pref['values']['base_font_size'])
        self.plot_wind = None
        self.parent.title("Algorithm and HRV Metrics")
        self.parent.resizable(width=FALSE, height=FALSE)
        self.parent.configure(highlightthickness=1, highlightbackground='grey')
        self.stats()

    @staticmethod
    def __remove_doubles(x, y=None):

        x = x.reshape(-1, )
        temp = np.append(np.diff(x), 1)
        x = x[temp != 0]

        if y is None:
            return x.reshape(-1, 1)
        else:
            y = y.reshape(-1, )
            y = y[temp != 0]
            return x.reshape(-1, 1), y.reshape(-1, 1)

    def stats(self):
        but_wtd = 20
        Fs = self.data_dict['Fs']

        if len(self.data_dict['R_t']) <= 1 & self.pref['values']['warnings_on']:
            messagebox.showwarning("Warning", "Cannot calculate HRV metrics \n\nPlease note: Annotations must be "
                                              "present for HRV metrics to be calculated.")
        else:
            if self.pref['values']['ECG_pref_on']:
                # Removes any accidental double-ups created during editing and sets metrics to be calculated based
                # on which plot is present
                if self.data_dict['plot_pred'] == 1:
                    R_t, tR_amp = self.__remove_doubles(self.data_dict['R_t'], self.data_dict['R_amp'])
                    Rpeakss = R_t
                else:
                    tR_t, tR_amp = self.__remove_doubles(self.data_dict['True_R_t'],  self.data_dict['True_R_amp'])
                    Rpeakss = tR_amp
                # WORK OUT HOW TO DRAW LATER
                # draw1()

            else:
                R_t = self.__remove_doubles(self.data_dict['R_t'])
                Rpeakss = R_t

            # Frame Parameters
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
            time_domain = Calculate_Features(Rpeakss, Fs)
            Label(time_dat, text="Time-Domain Parameters", anchor=TKc.W, font=self.text).grid(row=0, column=0,
                                                                                              columnspan=2)

            for i, dat in enumerate(time_domain):
                self.__add_paired_label(time_dat, self.tf_lbls[i], dat, row=i+1)

            Label(time_dat, text="", anchor=TKc.W, width=int(but_wtd / 4)).grid(row=5, column=2)  # SPACER

            # FREQUENCY-DOMAIN Parameters
            Label(freq_dat_up, text="Frequency-Domain Parameters", anchor=TKc.W, font=self.text).pack(side='top',
                                                                                                      anchor='center')

            # Frequency-domain Statistics
            self.Rpeak_input = Rpeakss / Fs

            # MENU FOR CHOICE OF ANALYSIS     title_list =
            self.frqmeth = StringVar(freq_dat_up)
            options = ['Welch', 'Blackman-Tukey', 'LombScargle', 'Auto Regression']
            RRImenu = OptionMenu(freq_dat_up, self.frqmeth, options[0], *options)
            RRImenu.config(width=16)
            #        RRImenu.configure(compound='right',image=self.photo)
            RRImenu.pack(side='top')

            self.frqmeth.trace('w', self.change_dropdown_HRV)
            self.__print_freq(but_wtd, method=1)

            # NON-LINEAR Parameters
            RRI = np.diff(self.Rpeak_input)
            rqa_ = RQA(RRI)
            pc_ = Poincare(RRI)
            dfa_ = DFA(RRI)
            self.__print_non_linear(non_dat, rqa_, pc_, dfa_, but_wtd)

            if self.pref['values']['loaded_ann'] == 1:
                self.prec = None
                try:
                    tR_t, tR_amp = self.__remove_doubles(self.data_dict['True_R_t'], self.data_dict['True_R_amp'])
                    self.__print_acc(tR_t, tR_amp, but_wtd)
                except KeyError:
                    raise Warning('Can\'t access true peaks. Check true R locations are loaded')

            else:
                tR_t = None

            Button(self.parent, text="Save", width=int(but_wtd / 2), height=2,
                   command=self.exp.savemetrics(Rpeakss, Fs, tR_t, td=time_domain, rqa=rqa_, dfa=dfa_, pc=pc_),
                   font='Helvetica 12 bold').pack(side='bottom', anchor='e')

            self.__open_plot()

    # NONLINEAR Parameters
    def __print_non_linear(self, df, rqa, pc, dfa, bw, fnt='Helvetica 10 bold'):
        """

        :param df: Dataframe.
        :param rqa: RQA parameters
        :param pc: Poincare parameters
        :param dfa: DFA parameters
        :param bw: Button Width
        :param fnt: Font for subtitles
        :return:
        """
        Label(df, text="Nonlinear Parameters", anchor=TKc.W, font=self.text).grid(row=0, column=0, columnspan=4)

        Label(df, text="Recurrence Analysis", anchor=TKc.W, width=bw, font=fnt).grid(row=1, column=0, columnspan=2)
        for i, dat in enumerate(rqa):
            self.__add_paired_label(df, self.rqa_lbls[i], dat, row=i + 2, col=0)

        Label(df, text="Poincare Analysis", anchor=TKc.W, width=bw, font=fnt).grid(row=1, column=2, columnspan=2)
        for i, dat in enumerate(pc):
            self.__add_paired_label(df, self.poincare_lbls[i], dat, row=i + 2, col=2)

        Label(df, text="DFA", anchor=TKc.W, width=bw, font=fnt).grid(row=5, column=2, columnspan=2)
        for i, dat in enumerate(dfa[:2]):
            self.__add_paired_label(df, self.dfa_lbls[i], dat, row=i + 6, col=2)

    def __print_acc(self, pks, true_pks, bw, tt: int = None):
        if tt is None:
            tt = self.pref['values']['tt']

        TP, FP, FN = test2(pks, true_pks, tt)
        Se, PP, ACC, DER = acc2(TP, FP, FN)

        Label(self.DA_dat, text="Detection Algorithm Metrics", anchor=TKc.W, font=self.text).grid(row=3, column=13,
                                                                                                  columnspan=4)
        for i, dat in enumerate([Se, PP, ACC, DER]):
            self.__add_paired_label(self.DA_dat, self.acc_lbls[i], dat, row=i+4, col=13)

        Label(self.DA_dat, text="Precision Window (ms)", anchor=TKc.W).grid(row=8, column=13)
        self.prec = Entry(self.DA_dat, width=int(bw / 2))
        self.prec.grid(row=8, column=14)
        time = self.data_dict['Fs'] / self.data_dict['Fs'] * 1000
        self.prec.insert(0, '{:.2f}'.format(time))
        Button(self.DA_dat, text="Update", anchor=TKc.W, width=int(bw / 2),
               command=lambda: self.updateprec(pks, true_pks, bw)).grid(row=8, column=15)

    def updateprec(self, pks, true_pks, but_wtd):
        self.DA_dat.destroy()
        self.DA_dat = Frame(master=self.Outer_DA_dat)
        self.DA_dat.pack(side='top')
        self.__print_acc(pks, true_pks, but_wtd, round(float(self.prec.get()) * self.data_dict['Fs'] / 1000))

    def change_dropdown_HRV(self, *args):
        methods = self.frqmeth.get()

        if methods == 'Welch':
            METH = 1
        elif methods == 'Blackman-Tukey':
            METH = 2
        elif methods == 'LombScargle':
            METH = 3
        else:
            METH = 4

        self.__print_freq(but_wtd=20, method=METH)

    def __print_freq(self, but_wtd, method):
        self.freq_dat_low.destroy()
        self.freq_dat_low = Frame(master=self.Outer_freq_dat)
        self.freq_dat_low.pack(side='top')

        def _gen_label(txt, w, r_, c_):
            Label(self.freq_dat_low, text=txt, anchor=TKc.W, width=w).grid(row=r_, column=c_)

        fq_vals = Freq_Analysis(self.Rpeak_input, meth=method, decim=3, M=self.pref['values']['welch_L'],
                                O=self.pref['values']['welch_O'], BTval=self.pref['values']['bltk_input'],
                                omega_max=self.pref['values']['ls_omega_max'], order=self.pref['values']['ar_order'])

        txt__ = ['VLF (Hz)', str(fq_vals[6]), 'LF (Hz)', fq_vals[7], 'HF (Hz)', fq_vals[8]]
        w1, w2 = int(but_wtd / 4 * 3), int(but_wtd / 2)

        _gen_label(txt='Peak Frequency', w=w1, r_=1, c_=0)
        c, r = 1, 1
        for i in [0, 2, 4]:
            for j in range(2):
                _gen_label(txt=txt__[i+j], w=w2, r_=r+j, c_=c)
            c += 1

        txt__ = ['VLF (%)', str(fq_vals[3]), 'LF (%)', fq_vals[4], 'HF (%)', fq_vals[5]]
        _gen_label(txt='Percentage Power', w=w1, r_=3, c_=0)
        c, r = 1, 3
        for i in [0, 2, 4]:
            for j in range(2):
                _gen_label(txt=txt__[i+j], w=w2, r_=r+j, c_=c)
            c += 1

        txt__ = ['VLF (ms^2)', str(fq_vals[0]), 'LF (ms^2)', fq_vals[1], 'HF (ms^2)', fq_vals[2]]
        _gen_label(txt='Absolute Power', w=w1, r_=5, c_=0)
        c, r = 1, 5
        for i in [0, 2, 4]:
            for j in range(2):
                _gen_label(txt=txt__[i+j], w=w2, r_=r+j, c_=c)
            c += 1

        Label(self.freq_dat_low, text="Peak Frequency", anchor=TKc.W, width=w1).grid(row=1, column=0)

    def __open_plot(self):
        # Change data from none
        dat = None

        if self.plot_wind is not None:
            self.plot_wind.destroy()
        self.plot_wind = Toplevel()
        self.pv = PlotViewer(self.plot_wind, dat, self.pref)
        self.plot_wind.bind('<Escape>', self.__close_plot_viewer)

    def __close_plot_viewer(self):
        # TODO - check if this works
        self.pv.close()

    def __add_paired_label(self, dframe, txt, value, row, col: int = 0, bw: int = 20):
        """

        :param dframe:
        :param txt:
        :param value:
        :param row:
        :param col:
        :param bw:
        :return:
        """
        Label(dframe, text=txt, anchor=TKc.W, width=bw, font=self.text).grid(row=row, column=col)
        Label(dframe, text=value, anchor=TKc.W, width=int(bw / 2), font=self.text).grid(row=row, column=col + 1)
