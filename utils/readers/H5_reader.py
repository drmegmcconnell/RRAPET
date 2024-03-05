

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