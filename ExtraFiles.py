"""
Extra Files for later
"""


with open("Preferences.txt", 'r') as f:
    Preferences = f.read().split()


def multRUN(path, savefilename, FS, ext):
    """

    :param path:
    :param savefilename:
    :param FS:
    :param ext:
    """
    # GET VARIABLES SET IN PROGRAM
    messagebox.showinfo("Runtime Info", "Multiple ECG processing is underway. Please do not touch RR-APET Program.")

    FreqMeth = 1  # Set 1 for Welch, 2 for Blackman-Tukey, 3 for Lombscargle, or 4 for AutoRegression
    Dec = 2
    RpeakMeth = int(Preferences[3])
    ML = float(Preferences[18])
    EL = float(Preferences[19])
    Lmin = float(Preferences[20])
    Ratio_vthr = float(Preferences[16])
    sbl = int(Preferences[17])
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
                          i] + ' - cannot import this file type! Please keep consistent file types when using RR-APET '
                               'for multiple ECG analysis')

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
                            powULF, powLF, powHF, perpowULF, perpowLF, perpowHF, peak_freq_ULF, peak_freq_LF, \
                            peak_freq_HF, LF_HF_ratio = Freq_Analysis(Rpeak_input, meth=FreqMeth, decim=Dec)
                            # ===================Nonlinear statistics====================#
                            RRI = np.diff(Rpeak_input)
                            REC, DET, LAM, Lmean, Lmax, Vmean, Vmax = RQA(RRI, decim=2)
                            SD1, SD2, c1, c2 = Poincare(RRI, decim=Dec)
                            alp1, alp2, _ = DFA(RRI, decim=Dec)

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
                          i] + ' - cannot import this file type! Please keep consistent file types when using RR-APET '
                               'for multiple ECG analysis')

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
                            REC, DET, LAM, Lmean, Lmax, Vmean, Vmax = RQA(RRI, decim=Dec)
                            SD1, SD2 = Poincare(RRI, decim=Dec)
                            alp1, alp2, _ = DFA(RRI, decim=Dec)
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
                print(file_names[i] + ' - cannot import this file type! Please keep consistent file types when '
                                      'using RR-APET for multiple ECG analysis')
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
                        REC, DET, LAM, Lmean, Lmax, Vmean, Vmax = RQA(RRI, decim=Dec)
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

    FILE = open("/home/meg/Documents/RRAPET_Linux/Preferences.txt", 'r')
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

    if dtyp == 1:
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
                R_peaks = og_ann[:, 0]
                if Preferences[23] == '1':
                    R_peaks = R_peaks / 1e3
                if np.mean(
                        np.diff(
                            R_peaks)) < 6:  # Average time interval between heart beats wouldn't be less than 10bpm, so a gap of over 6 seconds on average or greater means loaded in as samples
                    R_peaks = R_peaks * Fs  # Measured in time but need samples
                R_peaks = R_peaks[R_peaks != 0]
                R_peaks = np.reshape(R_peaks, [len(R_peaks), ])
                if not (np.diff(R_peaks[:]) > 0).all():
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
                        REC, DET, LAM, Lmean, Lmax, Vmean, Vmax = RQA(RRI, decim=Dec)
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
                          i] + ' - cannot import this file type! Please keep consistent file types when using '
                               'RR-APET for multiple ECG analysis')

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
                    if np.mean(np.diff(R_peaks)) < 6:  # Average time interval between heart beats wouldn't be
                        # less than  10bpm, so a gap of over 6 seconds on average or greater means loaded in as samples
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
                            powULF, powLF, powHF, perpowULF, perpowLF, perpowHF, peak_freq_ULF, peak_freq_LF, \
                            peak_freq_HF, LF_HF_ratio = Freq_Analysis(Rpeak_input, meth=FreqMeth, decim=Dec)
                            # ===================Nonlinear statistics====================#
                            RRI = np.diff(Rpeak_input)
                            REC, DET, LAM, Lmean, Lmax, Vmean, Vmax = RQA(RRI, decim=Dec)
                            SD1, SD2 = Poincare(RRI, decim=Dec)
                            alp1, alp2, F = DFA(RRI, decim=Dec)
                        with open(saveroot, "a") as text_file:
                            savename = fn + '->' + f_name
                            print(f"{savename}\t{SDNN}\t{SDANN}\t{MeanRR}\t{RMSSD}\t{pNN50}\t{powULF}\t{powLF}\t" +
                                  f"{powHF}\t{perpowULF}\t{perpowLF}\t{perpowHF}\t{peak_freq_HF}\t{peak_freq_LF}\t" +
                                  f"{peak_freq_HF}\t{LF_HF_ratio}\t{REC}\t{DET}\t{LAM}\t{Lmean}\t{Lmax}\t{Vmean}\t" +
                                  f"{Vmax}\t{SD1}\t{SD2}\t{alp1}\t{alp2}", file=text_file)


    elif (dtyp == 3):
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
                            powULF, powLF, powHF, perpowULF, perpowLF, perpowHF, peak_freq_ULF, peak_freq_LF, \
                            peak_freq_HF, LF_HF_ratio = Freq_Analysis(Rpeak_input, meth=FreqMeth, decim=Dec)
                            # ===================Nonlinear statistics====================#
                            RRI = np.diff(Rpeak_input)
                            REC, DET, LAM, Lmean, Lmax, Vmean, Vmax = RQA(RRI, decim=Dec)
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



# ~~~~~~~~~~~~~~ WINDOWS - USER PREFERENCES WINDOW ~~~~~~~~~~~~~~~~~~~#
class UserPreferences(Frame):

    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.btn1 = None
        self.btn3 = None
        self.btn4 = None
        self.btn5 = None
        self.contructionImage = None
        self.pref = None
        self.new_font = None
        self.parent = parent
        self.initUI_pref()

    def initUI_pref(self):
        """
        Basic set-up
        """
        global test_txt
        global Outerframe
        global F
        global F2
        global big_frame
        global btn
        global trybtn
        global resetbtn
        global okbtn
        global Preferences
        self.contructionImage = PhotoImage(master=self, file='./Pics/construction.png')
        self.contructionImage = self.contructionImage.subsample(3, 3)

        self.pref = Preferences
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

        self.btn1 = Button2(button_frame, text="Text Options", command=self._text_op, style='UserPref.TButton')
        self.btn1.pack()

        # self.btn2 = Button2(button_frame, text = "ECG View Options", command=self.ecg_op, style='UserPref.TButton')
        # self.btn2.pack()

        self.btn3 = Button2(button_frame, text="Prediction Mode Settings", command=self.prediction_op,
                            style='UserPref.TButton')
        self.btn3.pack()
        self.btn4 = Button2(button_frame, text="HRV Analysis Settings", command=self.metric_op,
                            style='UserPref.TButton')
        self.btn4.pack()
        self.btn5 = Button2(button_frame, text="General Settings", command=self._text_analysis_op,
                            style='UserPref.TButton')
        self.btn5.pack()

        self._text_op()

    def _text_op(self):
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
        Frame1 = Frame(Outerframe)
        Frame1.pack(side='left')
        Frame2 = Frame(Outerframe)
        Frame2.pack(side='left')

        Label(Frame1, text="Style", font=cust_subheadernb).pack(side="top")
        font_nodes = Listbox(Frame1, font=cust_text, exportselection=0, width=20)
        font_nodes.pack(side="left", fill="y")
        fonts = ['Helvetica', 'Courier', 'FreeSans', 'FreeSerif', 'Times', 'Verdana']
        for i in range(len(fonts)):
            font_nodes.insert(END, fonts[i])

        Label(Frame2, text="Size", font=cust_subheadernb).pack(side="top")
        text_size_nodes = Listbox(Frame2, font=cust_text, exportselection=0, width=5)
        text_size_nodes.pack(side="left", fill="y")
        scrollbar_tsz = Scrollbar(Frame2, orient="vertical")
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
    #        global Preferences
    #        self.btn1.config(style='UserPref.TButton', takefocus=False)
    #        self.btn2.config(style='SelectUserPref.TButton', takefocus=False)
    #        self.btn3.config(style='UserPref.TButton', takefocus=False)
    #        self.btn4.config(style='UserPref.TButton', takefocus=False)
    #        self.btn5.config(style='UserPref.TButton', takefocus=False)
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
    #        Outerframe = Frame(big_frame, bg = 'white', borderwidth=10, highlightcolor='grey', highlightthickness=1,
    #        highlightbackground='black')
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
        """

        :return:
        """
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

        self.btn1.config(style='UserPref.TButton', takefocus=False)
        #        self.btn2.config(style='UserPref.TButton', takefocus=False)
        self.btn3.config(style='SelectUserPref.TButton', takefocus=False)
        self.btn4.config(style='UserPref.TButton', takefocus=False)
        self.btn5.config(style='UserPref.TButton', takefocus=False)

        Outerframe = Frame(big_frame, bg='white', borderwidth=10, highlightcolor='grey', highlightthickness=1,
                           highlightbackground='black')
        Outerframe.pack(side='left', fill='y')
        Frame1 = Frame(Outerframe, bg='white')
        Frame1.pack(side='top', fill='y')

        Label2(Frame1, text='Detection Algorithm Parameters', style='Header.TLabel').grid(row=2, column=0, columnspan=3,
                                                                                          sticky='w')
        Label(Frame1, text='Detection Method', bg='white', font=cust_text, width=22, anchor='w').grid(row=3, column=0,
                                                                                                      sticky='w')
        Meth = StringVar(Frame1)
        Opts_meth = ['MHTD', 'Pan-Tompkins', 'K-means', 'Own Method/Code']
        Method_menu = OptionMenu(Frame1, Meth, Opts_meth[int(Preferences[3]) - 1], *Opts_meth)
        Method_menu.config(style='Text.TMenubutton')
        Method_menu.grid(row=3, column=1)
        Meth.trace('w', self._F2vals)

        self._F2vals()

    def _F2vals(self):
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

        self.vals = Preferences

        F2 = Frame(Outerframe, bg='white')
        F2.pack(side='top', fill='both')

        val = Meth.get()
        if val == 'MHTD':
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

            okbtn = Button(self.parent, text="Ok", command=lambda: self._repredict(1, 1),
                           font=cust_text)  # thr.get(),sbl.get(),magL.get(),engL.get(),minL.get()
            okbtn.pack(side='right', anchor='e')
            btn = Button(self.parent, text="Test", command=lambda: self._repredict(1, 0),
                         font=cust_text)  # thr.get(),sbl.get(),magL.get(),engL.get(),minL.get()
            btn.pack(side='right', anchor='e')
            resetbtn = Button(self.parent, text="Reset", command=lambda: self._reset(1), font=cust_text)
            resetbtn.pack(side='right', anchor='e')

        elif val == 'Pan-Tompkins':
            txt = "Pan-Tompkins currently only avaliable in RR-APET's implementation of it's original form " \
                  "(i.e. unable to alter parameters; however, this function will be avaliable in later releases)."
            Label(F2, text=txt, bg='white', font=cust_text, width=35, anchor='w', wraplength=330).grid(row=3, column=0,
                                                                                                       sticky='w')
            # Label(F2, text='Under construction', bg='white', font = cust_text, width=22,
            # anchor='w').grid(row=3, column=0, sticky='w')

            imag = Label(F2, image=self.contructionImage, compound='center', takefocus=False, bg='white')
            imag.grid(row=4, column=0, columnspan=3, sticky='e' + 'w')

            okbtn = Button(self.parent, text="Ok", command=lambda: self._repredict(2, 1),
                           font=cust_text)  # thr.get(),sbl.get(),magL.get(),engL.get(),minL.get()
            okbtn.pack(side='right', anchor='e')
            btn = Button(self.parent, text="Test", command=lambda: self._repredict(2, 0), font=cust_text)
            btn.pack(side='right', anchor='e')
            resetbtn = Button(self.parent, text="Reset", command=lambda: self._reset(2), font=cust_text)
            resetbtn.pack(side='right', anchor='e')

        elif val == 'K-means':
            txt = "K-means currently only avaliable in RR-APETs implementation of it's original form " \
                  "(i.e. unable to alter parameters; however, this function will be avaliable in later releases)."
            Label(F2, text=txt, bg='white', font=cust_text, width=35, anchor='w', wraplength=330).grid(row=3, column=0,
                                                                                                       sticky='w')
            imag = Label(F2, image=self.contructionImage, compound='center', takefocus=False, bg='white')
            imag.grid(row=4, column=0, columnspan=3, sticky='e' + 'w')
            okbtn = Button(self.parent, text="Ok", command=lambda: self._repredict(3, 1),
                           font=cust_text)  # thr.get(),sbl.get(),magL.get(),engL.get(),minL.get()
            okbtn.pack(side='right', anchor='e')
            btn = Button(self.parent, text="Test", command=lambda: self._repredict(3, 0), font=cust_text)
            btn.pack(side='right', anchor='e')
            resetbtn = Button(self.parent, text="Reset", command=lambda: self._reset(3), font=cust_text)
            resetbtn.pack(side='right', anchor='e')

        elif val == 'Own Method/Code':
            txt = "Change the settings of your own method in the provided python script; or use the provided code " \
                  "and insert your options below..."
            Label(F2, text=txt, bg='white', font=cust_text, width=35, anchor='w', wraplength=330).grid(row=3, column=0,
                                                                                                       columnspan=3,
                                                                                                       sticky='ew')

            okbtn = Button(self.parent, text="Ok", command=lambda: self._repredict(4, 1),
                           font=cust_text)  # thr.get(),sbl.get(),magL.get(),engL.get(),minL.get()
            okbtn.pack(side='right', anchor='e')
            btn = Button(self.parent, text="Test", command=lambda: self._repredict(4, 0), font=cust_text)
            btn.pack(side='right', anchor='e')
            resetbtn = Button(self.parent, text="Reset", command=lambda: self._reset(4), font=cust_text)
            resetbtn.pack(side='right', anchor='e')

    def _repredict(self, val, closewind):

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
                messagebox.showwarning("Warning", "Incorrect data-type detected. Please ensure you are using correct "
                                                  "format for each MHTD variable.")
                self._F2vals()

        if val == 2:
            try:
                #               PASS THE VALUES THAT RELATE TO PAN-TOMPKIN!! WHEN THIS IS AVALIABLE
                Prediction_mode(2)
                if closewind == 1:
                    pref.withdraw()
                    replace_line("Preferences.txt", 3, str(2) + '\n')
            except:
                messagebox.showwarning("Warning", "Pan-tompkins unavaliable for this signal. Please report error "
                                                  "to makers of RR-APET for further support.")
                self._F2vals()

        if val == 3:
            try:
                #               PASS THE VALUES THAT RELATE TO PAN-TOMPKIN!! WHEN THIS IS AVALIABLE
                Prediction_mode(3)
                if closewind == 1:
                    pref.withdraw()
                    replace_line("Preferences.txt", 3, str(3) + '\n')
            except:
                messagebox.showwarning("Warning", "K-means unavaliable for this signal. Please report error to makers "
                                                  "of RR-APET for further support.")
                self._F2vals()

        if val == 4:
            try:
                #               PASS THE VALUES THAT RELATE TO PAN-TOMPKIN!! WHEN THIS IS AVALIABLE
                Prediction_mode(4)
                if closewind == 1:
                    pref.withdraw()
                    replace_line("Preferences.txt", 3, str(4) + '\n')
            except:
                messagebox.showwarning("Warning", "Own method did not work for prediction. Please check your code "
                                                  "or contact makers of RR-APET for further support.")
                self._F2vals()

    def _reset(self, method):
        """
        Reseting the Metrics
        :param method:
        """
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
                self._repredict(1, 0)
                self._F2vals()

        if method == 2:  # PAN-TOMPKINS
            self._repredict(2, 0)
            self._F2vals()
            # Reset Pan-Tompkins using OG READ IN
        #            replace_line("Preferences.txt", 16, str(val[16]) + '\n')

        if method == 3:  # K-MEANS
            self._repredict(3, 0)
            self._F2vals()

        if method == 4:  # OWN METHOD
            self._repredict(4, 0)
            self._F2vals()

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
                     command=lambda: self._update_mets(Wel_M.get(), Wel_O.get(), BT.get(), LS.get(), AR.get(),
                                                       DFA1.get(), DFA2.get(), DFA3.get(), DFA4.get(), RQA1.get(),
                                                       RQA2.get()), font=cust_text)
        btn.pack(side='right', anchor='e')

    def _text_analysis_op(self):
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

        P = Preferences

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
                     command=lambda: self._update_cust_text(self.dtype.get(), self.dunits.get(), self.tunits.get(),
                                                            self.separator.get()), font=cust_text)
        btn.pack(side='right', anchor='e')

    # command=lambda:self.updateprec(but_wtd))

    @staticmethod
    def _update_cust_text(dt, du, tu, sep):
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

    def _update_mets(self, welm, welo, bt, ls, ar, dfa1, dfa2, dfa3, dfa4, rqa1, rqa2):
        pref.withdraw()
        try:
            val = int(welm)
            replace_line("Preferences.txt", 5, str(val) + '\n')
        except:
            messagebox.showwarning("Warning", "Welch PSD segment length must be an integer. "
                                              "\n\nPlease note: Value NOT updated.")

        try:
            val = int(welo)
            replace_line("Preferences.txt", 6, str(val) + '\n')
        except:
            messagebox.showwarning("Warning", "Welch PSD overlap length must be an integer. "
                                              "\n\nPlease note: Value NOT updated.")

        try:
            val = int(bt)
            replace_line("Preferences.txt", 7, str(val) + '\n')
        except:
            messagebox.showwarning("Warning", "Blackman-Tukey PSD N-bins value must be an integer. "
                                              "\n\nPlease note: Value NOT updated.")

        try:
            val = int(ls)
            replace_line("Preferences.txt", 8, str(val) + '\n')
        except:
            messagebox.showwarning("Warning", "Lombscargle Omega max value must be an integer. "
                                              "\n\nPlease note: Value NOT updated.")

        try:
            val = int(ar)
            replace_line("Preferences.txt", 9, str(val) + '\n')
        except:
            messagebox.showwarning("Warning", "Auto-Regression Order value must be an integer. "
                                              "\n\nPlease note: Value NOT updated.")

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
                if (minbox_temp > copbox_temp) or (copbox_temp > maxbox_temp):
                    messagebox.showwarning("Warning", "The values you have selected are incompatiable.\n\nThe minimum "
                                                      "box length must be less than the crossover point and the "
                                                      "crossover point must be less than the maximum box length.")
                else:
                    messagebox.showwarning("Warning", "The values you have selected are incompatiable.\n\nThere must "
                                                      "be at least two points per gradient. Hint: Try lowering the "
                                                      "step size value.")
        except:
            messagebox.showwarning("Warning", "All DFA parameters value must be integers. "
                                              "\n\nPlease note: Values NOT updated.")

        try:
            val = int(rqa1)
            replace_line("Preferences.txt", 14, str(val) + '\n')
        except:
            messagebox.showwarning("Warning", "RQA embedding dimension parameter must be an integer. "
                                              "\n\nPlease note: Value NOT updated.")

        try:
            val = int(rqa2)
            replace_line("Preferences.txt", 15, str(val) + '\n')
        except:
            messagebox.showwarning("Warning", "RQA lag parameter must be an integer. "
                                              "\n\nPlease note: Value NOT updated.")

    def fakeCommand(self):
        print('Under-Construction')

    def update_font(self, trial, new_font, new_size):

        global test_txt
        global root
        if new_font != '':
            replace_line("Preferences.txt", 0, new_font + '\n')
        if new_size != '':
            replace_line("Preferences.txt", 1, new_size + '\n')

        self.new_font = font.Font(family=Preferences[0], size=int(Preferences[1]))

        test_txt.destroy()
        test_txt = Label(self.parent, text="Sample Text", font=self.new_font)
        test_txt.pack()

        if trial == 0:
            pref.withdraw()
            headerStyles()


# ~~~~~~~~~~~~~~ WINDOWS - HRV analysis WINDOW ~~~~~~~~~~~~~~~~~~~#
class HRVstatics(Frame):
    """
    Calculating HRV Statistics
    """
    def __init__(self, parent):
        Frame.__init__(self, parent)

        self.frqmeth = None
        self.Outer_freq_dat = None
        self.Rpeak_input = None
        self.DA_dat = None
        self.Outer_DA_dat = None
        self.freq_dat_low = None
        self.welchL = None
        self.welchO = None
        self.btval_input = None
        self.omax_input = None
        self.order = None
        self.parent = parent
        self._initUI_mets()

    def _initUI_mets(self):

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
        but_wtd = 20

        if len(R_t) <= 1 & warnings_on:
            messagebox.showwarning("Warning", "Cannot calculate HRV metrics \n\nPlease note: Annotations must be "
                                              "present for HRV metrics to be calculated.")
        else:
            if ECG_pref_on:
                # Removes any accidental double-ups created during editing and sets metrics to be calculated based
                # on which plot is present
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

            self.welchL = float(Preferences[5])
            self.welchO = float(Preferences[6])
            self.btval_input = int(Preferences[7])  # 10
            self.omax_input = int(Preferences[8])  # 500
            self.order = int(Preferences[9])  # 10
            # Time-domain Statistics
            SDNN, SDANN, MeanRR, RMSSD, pNN50 = Calculate_Features(Rpeakss, Fs)

            # Frequency-domain Statistics
            self.Rpeak_input = Rpeakss / Fs
            Freq__ = Freq_Analysis(self.Rpeak_input, meth=1, decim=3, M=self.welchL, O=self.welchO,
                                   BTval=self.btval_input, omega_max=self.omax_input, order=self.order)

            # Nonlinear statistics
            RRI = np.diff(self.Rpeak_input)

            REC, DET, LAM, Lmean, Lmax, Vmean, Vmax = RQA(RRI)
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

            Label(non_dat, text="Poincare Analysis", anchor=TKc.W, width=but_wtd,
                  font='Helvetica 10 bold').grid(row=1, column=2, columnspan=2)
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

            Button(self.parent, text="Save", width=int(but_wtd / 2), height=2,
                   command=exp.savemetrics(R_t, loaded_ann, labelled_flag, Fs),
                   font='Helvetica 12 bold').pack(side='bottom', anchor='e')

            self.__open_plot()

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

        self._updatefreqstats(but_wtd=20, method=METH)

    def _updatefreqstats(self, but_wtd, method):
        self.freq_dat_low.destroy()
        self.freq_dat_low = Frame(master=self.Outer_freq_dat)
        self.freq_dat_low.pack(side='top')

        def _gen_label(txt, w, r, c):
            Label(self.freq_dat_low, text=txt, anchor=TKc.W, width=w).grid(row=r, column=c)

        fq_vals = Freq_Analysis(self.Rpeak_input, meth=1, decim=3, M=self.welchL, O=self.welchO, BTval=self.btval_input,
                                omega_max=self.omax_input, order=self.order)

        txt__ = ['VLF (Hz)', str(fq_vals[6]), 'LF (Hz)', fq_vals[7], 'HF (Hz)', fq_vals[8]]
        w1, w2 = int(but_wtd / 4 * 3), int(but_wtd / 2)

        _gen_label(txt='Peak Frequency', w=w1, r=1, c=0)
        c, r = 1, 1
        for i in [0, 2, 4]:
            for j in range(2):
                _gen_label(txt=txt__[i+j], w=w2, r=r+j, c=c)
            c += 1

        txt__ = ['VLF (%)', str(fq_vals[3]), 'LF (%)', fq_vals[4], 'HF (%)', fq_vals[5]]
        _gen_label(txt='Percentage Power', w=w1, r=3, c=0)
        c, r = 1, 3
        for i in [0, 2, 4]:
            for j in range(2):
                _gen_label(txt=txt__[i+j], w=w2, r=r+j, c=c)
            c += 1

        txt__ = ['VLF (ms^2)', str(fq_vals[0]), 'LF (ms^2)', fq_vals[1], 'HF (ms^2)', fq_vals[2]]
        _gen_label(txt='Absolute Power', w=w1, r=5, c=0)
        c, r = 1, 5
        for i in [0, 2, 4]:
            for j in range(2):
                _gen_label(txt=txt__[i+j], w=w2, r=r+j, c=c)
            c += 1

        # Label(self.freq_dat_low, text="Peak Frequency", anchor=TKc.W, width=w1).grid(row=1, column=0)
        # Label(self.freq_dat_low, text="VLF (Hz)", anchor=TKc.W, width=w2).grid(row=1, column=1)
        # Label(self.freq_dat_low, text=fq_vals[6], anchor=TKc.W, width=w2).grid(row=2, column=1)
        # Label(self.freq_dat_low, text="LF (Hz)", anchor=TKc.W, width=w2).grid(row=1, column=2)
        # Label(self.freq_dat_low, text=fq_vals[7], anchor=TKc.W, width=w2).grid(row=2, column=2)
        # Label(self.freq_dat_low, text="HF (Hz)", anchor=TKc.W, width=w2).grid(row=1, column=3)
        # Label(self.freq_dat_low, text=fq_vals[8], anchor=TKc.W, width=w2).grid(row=2, column=3)

        # Label(self.freq_dat_low, text="Percentage Power", anchor=TKc.W, width=w1).grid(row=3, column=0)
        # Label(self.freq_dat_low, text="VLF (%)", anchor=TKc.W, width=w2).grid(row=3, column=1)
        # Label(self.freq_dat_low, text=fq_vals[3], anchor=TKc.W, width=w2).grid(row=4, column=1)
        # Label(self.freq_dat_low, text="LF (%)", anchor=TKc.W, width=w2).grid(row=3, column=2)
        # Label(self.freq_dat_low, text=fq_vals[4], anchor=TKc.W, width=w2).grid(row=4, column=2)
        # Label(self.freq_dat_low, text="HF (%)", anchor=TKc.W, width=w2).grid(row=3, column=3)
        # Label(self.freq_dat_low, text=fq_vals[5], anchor=TKc.W, width=w2).grid(row=4, column=3)

        # Label(self.freq_dat_low, text="Absolute Power", anchor=TKc.W, width=w1).grid(row=5, column=0)
        # Label(self.freq_dat_low, text="VLF (ms^2)", anchor=TKc.W, width=w2).grid(row=5, column=1)
        # Label(self.freq_dat_low, text=fq_vals[0], anchor=TKc.W, width=w2).grid(row=6, column=1)
        # Label(self.freq_dat_low, text="LF (ms^2)", anchor=TKc.W, width=w2).grid(row=5, column=2)
        # Label(self.freq_dat_low, text=fq_vals[1], anchor=TKc.W, width=w2).grid(row=6, column=2)
        # Label(self.freq_dat_low, text="HF (ms^2)", anchor=TKc.W, width=w2).grid(row=5, column=3)
        # Label(self.freq_dat_low, text=fq_vals[2], anchor=TKc.W, width=w2).grid(row=6, column=3)
        # Label(self.freq_dat_low, text="LF/HF Ratio", anchor=TKc.W, width=w1).grid(row=7, column=0)
        # Label(self.freq_dat_low, text=fq_vals[9], anchor=TKc.W, width=w2).grid(row=7, column=1)

        # Label(self.freq_dat_low, text="", anchor=TKc.W, width=int(but_wtd / 4)).grid(row=5, column=4)

    def __open_plot(self):
        global plot_wind
        if plot_wind is not None:
            plot_wind.destroy()
        plot_wind = Toplevel()
        plot_viewer(plot_wind)
        if windows_compile:
            plot_wind.bind('<Escape>', self.__close_plot_viewer)
        if linux_compile:
            plot_wind.bind('<Control-Escape>', self.__close_plot_viewer)

    @staticmethod
    def __close_plot_viewer(event):
        plot_wind.withdraw()


class plot_viewer(Frame):
    """
    Plot Viewer Class
    """
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.initUI_plotting()
        # INITIAL VALUES FOR FREQUENCY PLOT
        self.welch_int_M = float(Preferences[5])
        self.welch_int_O = int(Preferences[6])
        self.btval_input = int(Preferences[7])  # 10
        self.omax_input = int(Preferences[8])  # 500
        self.order = int(Preferences[9])  # 10
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
        toolMenu.add_command(label="Save", command=exp.savefigure(draw_figure))
        toolMenu.add_command(label="Quit", command=self.close_plot)
        menubar.add_cascade(label="Options", menu=toolMenu, font=cust_subheadernb)

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
            REC, DET, LAM, Lmean, Lmax, Vmean, Vmax = RQA_plot(local_RRI_nonlinear, m=self.M, l=self.L)
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
                    RQA_plott(figs)
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


class HELP(Frame):
    def __init__(self, parent):
        #        global siz
        Frame.__init__(self, parent)
        self.parent = parent
        self.parent.title("Help")
        self.pack(fill='both', expand=0)
        self.config(bg='white')
        OuterFrame_top = Frame(self.parent, bg='white smoke', borderwidth=5)
        OuterFrame_top.pack(side='top', fill='both')
        small_frame3 = Frame(OuterFrame_top, bg='white', highlightcolor='grey', highlightthickness=1,
                             highlightbackground='black')
        small_frame3.pack(side='left', fill='y')
        #        spacer = Label(OuterFrame_top, text = ' ', bg = 'white smoke')
        #        spacer.pack(side='left', fill='y')
        USF3 = Frame(small_frame3, bg='light blue')
        USF3.pack(side='top', fill='both')
        LSF3 = Frame(small_frame3, bg='white')
        LSF3.pack(side='top', fill='y')
        Label(USF3, text="Help", wraplength=200, font='Helvetica 12 bold', bg='light blue').pack(side='top')
        Label(LSF3,
              text="Please open the pdf file 'RR_APET_readme.pdf' provided within the packaged program for detailed instructions on program use, or contact the authors using the 'contact us' option.",
              font='Helvetica 10', wraplength=200, bg='white', pady=10).pack(side='top')


class Contact_Us(Frame):
    def __init__(self, parent):
        #        global siz
        Frame.__init__(self, parent)
        self.parent = parent
        self.parent.title("Contact")
        self.pack(fill='both', expand=0)
        self.config(bg='white')
        OuterFrame_top = Frame(self.parent, bg='white smoke', borderwidth=5)
        OuterFrame_top.pack(side='top', fill='both')
        small_frame3 = Frame(OuterFrame_top, bg='white', highlightcolor='grey', highlightthickness=1,
                             highlightbackground='black')
        small_frame3.pack(side='left', fill='y')
        #        spacer = Label(OuterFrame_top, text = ' ', bg = 'white smoke')
        #        spacer.pack(side='left', fill='y')
        USF3 = Frame(small_frame3, bg='light blue')
        USF3.pack(side='top', fill='both')
        LSF3 = Frame(small_frame3, bg='white')
        LSF3.pack(side='top', fill='y')
        Label(USF3, text="Contact Us", wraplength=200, font='Helvetica 12 bold', bg='light blue').pack(side='top')
        tt = "Main developer: Meghan McConnell\nm.mcconnell@griffith.edu.au\n\nCo developer: Belinda Schwerin\nb.schwerin@griffith.edu.au\n\nCo developer: Stephen So\ns.so@griffith.edu.au"
        Label(LSF3, text=tt, font='Helvetica 10', bg='white', pady=10, justify='left').pack(side='top', anchor='e')


# ~~~~~~~~~~~~~~ WINDOWS - Helpful suggestion when hover over for a period of time ~~~~~~~~~~~~~~~~~~~#


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

    # Setting event=0 makes it an optional input -> a button and keyboard press can activate the same sub-function
    def callback(self, event=0):
        global PATH
        fname, file_extension = os.path.splitext(self.name.get())
        if (self.Samp.get() == '') or (self.Samp.get() == ' '):
            messagebox.showwarning("Warning", "Sampling frequency required. Try again.")
        elif (self.name.get() == '') or (self.name.get() == ' '):
            messagebox.showwarning("Warning", "File name required for output file! Try again.")
        elif file_extension == '':
            messagebox.showwarning("Warning", "File type required for output file! Try again.")
        else:
            self.parent.withdraw()
            self.multRUN(PATH, fname, int(self.Samp.get()), file_extension)

    def callback2(self, event=0):
        self.parent.withdraw()

    def multRUN(self, *args):
        print(*args)
