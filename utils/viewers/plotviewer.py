# ~~~~~~~~~~~~~~ WINDOWS - HRV analysis WINDOW ~~~~~~~~~~~~~~~~~~~#
from tkinter import Frame, BOTH, Menu
from matplotlib.figure import Figure
from utils.Style_Functions import headerStyles


class PlotViewer(Frame):
    """
    Plot Viewer Class
    """
    def __init__(self, parent, data_dict, preferences):
        Frame.__init__(self, parent)

        self.parent = parent
        self.initUI_plotting()
        self.pref = preferences
        # INITIAL VALUES FOR FREQUENCY PLOT
        self.welch_int_M = float(self.pref['values']['welch_L'])
        self.welch_int_O = int(self.pref['values']['welch_O'])
        self.btval_input = int(self.pref['values']['bltk_input'])  # 10
        self.omax_input = int(self.pref['values']['ls_omega_max'])  # 500
        self.order = int(self.pref['values']['ar_order'])  # 10
        self.m_input = self.welch_int_M  # int(self.sig_len*self.welch_int_M/100)
        self.o_input = self.welch_int_O  # int(self.m_input*self.welch_int_O/100)

        # INITIAL VALUES FOR DFA PLOT
        self.minbox = int(self.pref['values']['minbox'])  # 1
        self.copbox = int(self.pref['values']['copbox'])  # 15
        self.maxbox = int(self.pref['values']['maxbox'])  # 64
        self.increm = int(self.pref['values']['increm'])  # 1

        # INITIAL VALUES FOR RQA PLOT
        self.M = int(self.pref['values']['rqa_m'])  # 10
        self.L = int(self.pref['values']['rqa_l'])  # 1

        self.text, self.subheader, self.subheadernb, self.header = headerStyles(self.pref['values']['font'],
                                                                                self.pref['values']['base_font_size'])

    def initUI_plotting(self):
        global freq_wind
        global frequency_figure
        global RQA_figure
        global R_t
        global Fs
        global graphCanvas2
        global draw_figure
        global pvp
        global R_t
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

        draw_figure = Figure(tight_layout=1)
        pvp = draw_figure.add_subplot(111)

        graphCanvas2 = FigureCanvasTkAgg(draw_figure, master=picture_frame)
        graphCanvas2.get_tk_widget().pack(side='top', fill=BOTH, expand=True)

        # ~~~~~~~~~~~~ Dropdown Menu for Prediction Mode ~~~~~~~~~~~~~~~~#

        # SET UP MENUBAR#
        menubar = Menu(self.parent, font=self.text)
        self.parent.config(menu=menubar)

        fileMenu = Menu(menubar, font=self.text, tearoff=False)

        fileMenu.add_command(label="Welch\'s Periodogram", command=lambda: self.reselect_graph(0))
        fileMenu.add_command(label="Blackman-Tukey\'s Periodogram", command=lambda: self.reselect_graph(1))
        fileMenu.add_command(label="Lomb-Scargle\'s Periodogram", command=lambda: self.reselect_graph(2))
        fileMenu.add_command(label="Autoregression Periodogram", command=lambda: self.reselect_graph(3))
        fileMenu.add_command(label="Poincare Plot", command=lambda: self.reselect_graph(4))
        fileMenu.add_command(label="DFA Plot", command=lambda: self.reselect_graph(5))
        fileMenu.add_command(label="RQA Plot", command=lambda: self.reselect_graph(6))
        fileMenu.add_command(label="Show all", command=lambda: self.reselect_graph(7))
        menubar.add_cascade(label="Select Plot", menu=fileMenu, font=self.subheadernb)

        toolMenu = Menu(menubar, font=self.text, tearoff=False)
        toolMenu.add_command(label="Save", command=exp.savefigure(draw_figure))
        toolMenu.add_command(label="Quit", command=self.close_plot)
        menubar.add_cascade(label="Options", menu=toolMenu, font=self.subheadernb)

        # DFA_plot(subplot_, Min=4, Max=64, Inc=1, COP=12):

        self.reselect_graph(0)

    def close_plot(self):
        plot_wind.withdraw()

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
                    if ((minbox_temp > copbox_temp) or (copbox_temp > maxbox_temp)):
                        messagebox.showwarning("Warning", "The values you have selected are incompatiable.\n\n The "
                                                          "minimum box length must be less than the crossover point "
                                                          "and the crossover point must be less than the maximum box "
                                                          "length.")
                    else:
                        messagebox.showwarning("Warning", "The values you have selected are incompatiable.\n\n There "
                                                          "must be at least two points per gradient. Hint: Try lowering"
                                                          " the step size value.")


            elif method == 7:
                self.M = int(self.M2_slide.get())
                self.L = int(self.L_slide.get())
                RQA_plott(pvp, graphCanvas2, Fs, Mval=self.M, Lval=self.L)
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
            Label(self.results_frame, text='Frequency Analysis', bg='white smoke',
                  font=cust_subheader).grid(row=0, column=0, columnspan=4, sticky=TKc.E + TKc.W)
            Label(self.results_frame, text='Variable', bg='white smoke', anchor=TKc.W, borderwidth=1, relief='solid',
                  font=cust_text).grid(row=1, column=0, sticky=TKc.E + TKc.W)
            res[:] = Freq_Analysis(local_RRI_freq, meth=method, decim=3, M=self.m_input, O=self.o_input,
                                   BTval=self.btval_input, omega_max=self.omax_input, order=self.order)
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
            Label(self.results_frame, text='', bg='white smoke', anchor=TKc.W,
                  font=cust_text).grid(row=5, column=0, sticky=TKc.E + TKc.W)
            Label(self.results_frame, text='LF/HF ratio', bg='white smoke', anchor=TKc.W,
                  font=cust_text).grid(row=6, column=0, sticky=TKc.E + TKc.W)
            Label(self.results_frame, text=res[9], bg='white smoke', anchor=TKc.E,
                  font=cust_text).grid(row=6, column=1, sticky=TKc.E + TKc.W)

        elif method == 5:  # Poincare parameters
            sd1, sd2, c1, c2 = Poincare(local_RRI_nonlinear)
            Label(self.results_frame, text='Nonlinear Analysis', bg='white smoke',
                  font=cust_subheader).grid(row=0, column=0, columnspan=4, sticky=TKc.E + TKc.W)

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
            Label(self.results_frame, text='Nonlinear Analysis', bg='white smoke',
                  font=cust_subheader).grid(row=0, column=0, columnspan=4, ticky=TKc.E + TKc.W)
            labels = ['Variable', 'alpha 1 (ms)', 'alpha 2 (ms)']
            res = ['Value', alp1, alp2]

            for itr in range(3):
                Label(self.results_frame, text=labels[itr], anchor=TKc.W, bg='white smoke', borderwidth=1,
                      relief='solid', font=cust_text).grid(row=itr + 1, column=0, sticky=TKc.E + TKc.W)
                Label(self.results_frame, text=res[itr], anchor=TKc.E, bg='white smoke', borderwidth=1, relief='solid',
                      font=cust_text).grid(row=itr + 1, column=1, sticky=TKc.E + TKc.W)

        elif method == 7:  # RQA parameters
            REC, DET, LAM, Lmean, Lmax, Vmean, Vmax = RQA(local_RRI_nonlinear, m=self.M, l=self.L)
            Label(self.results_frame, text='Nonlinear Analysis', bg='white smoke',
                  font=cust_subheader).grid(row=0, column=0, columnspan=4, sticky=TKc.E + TKc.W)
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

                elif (sf == 3):
                    Label(self.Slider_housing, text="Autoregression Plotting Parameters", bg='white smoke',
                          font=cust_subheader).pack(side='top')
                    self.AR_slide = Scale(mini_frame1, label='Order', from_=1, to=200, orient=TKc.HORIZONTAL,
                                          bg='white smoke', borderwidth=0, highlightthickness=0)
                    self.AR_slide.pack(side='left', fill='x', expand=True)
                    self.AR_slide.set(self.order)


            elif (sf == 4):
                Poincare_plot(pvp)



            elif (sf == 5):
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



            elif (sf == 6):
                RQA_plott(pvp, graphCanvas2, Fs, Mval=self.M, Lval=self.L)

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


        elif (sf == 7):
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
                if (count == 0):
                    freq_plot(1, 3, figs)
                elif (count == 1):
                    freq_plot(2, 3, figs)
                elif (count == 2):
                    freq_plot(3, 3, figs)
                elif (count == 3):
                    freq_plot(4, 3, figs)
                elif (count == 4):
                    Poincare_plot(figs)
                elif (count == 5):
                    DFA_plot(figs)
                elif (count == 6):
                    RQA_plot(figs)
                elif (count == 7):
                    figs.clear()
                elif (count == 8):
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
        if (sf == 1):
            replace_line("Preferences.txt", 5, str(self.M_slide.get()) + '\n')
            replace_line("Preferences.txt", 6, str(self.O_slide.get()) + '\n')
        elif (sf == 2):
            replace_line("Preferences.txt", 7, str(self.BT_slide.get()) + '\n')
        elif (sf == 3):
            replace_line("Preferences.txt", 8, str(self.omeg_slide.get()) + '\n')
        elif (sf == 4):
            replace_line("Preferences.txt", 9, str(self.AR_slide.get()) + '\n')

        #        elif (sf == 5):

    def close(self):
        self.parent.withdraw()
