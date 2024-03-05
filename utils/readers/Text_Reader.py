import numpy as np
from tkinter import messagebox


def read_txt_file(fname, sep, warnings_on):
    """
    Read a text file
    :param sep:
    :param fname:
    :param warnings_on:
    """

    file = open(fname, 'r')
    if warnings_on:
        messagebox.showwarning("Warning", "The file you have selected is unreadable. Please ensure that "
                                          "the file interest is saved in the correct format. For further "
                                          "information, refer to the 'Help' module")
    else:
        if sep == '0':
            temp = file.read().split()
        elif sep == '1':
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
