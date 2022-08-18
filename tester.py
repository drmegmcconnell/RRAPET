"""
Testing h5 stuff
"""

# from tkinter import *
# from tkinter.messagebox import *
# root = Tk()
# val = 0
# val2 = 0
#
# def op1():
#     global e, l, root, val, e2, b, new
#     try:
#         val = int(e.get())
#     except ValueError:
#         showerror("Error", "Enter an int")
#     else:
#         new = Toplevel()
#         e2 = Entry(new)
#         e2.pack(side=LEFT)
#         b2 = Button(new, text="OK", command=op2)
#         b2.pack(side=RIGHT)
#         l2 = Label(new, text="Enter new number to multiply %d by" %val)
#         l2.pack()
#         e2.focus_force()
#         root.wait_window(new)
#         for i in range(5):
#             print(i + 1)
#
# def op2():
#     global val
#     try:
#         val2 = int(e2.get())
#     except ValueError:
#         showerror("Error", "Enter an int")
#         e2.focus_force()
#     else:
#         val = val * val2
#         l.config(text="This is your total: %d Click OK to exit" %val)
#         new.destroy()
#         b.config(command=op3)
#
# def op3():
#     root.destroy()
#
# e = Entry(root)
# e.pack(side=LEFT)
# b = Button(root, text="OK", command=op1)
# b.pack(side=RIGHT)
# l = Label(root, text="Enter a number")
# l.pack()
# root.mainloop()

from tkinter import Toplevel, filedialog, Tk, Frame   # Button, messagebox
from utils import H5_Selector
import time


class FakeModule(Frame):
    """
    Class string
    """
    def __init__(self, parent):
        super().__init__()
        self.folder = None
        self.parent = parent
        self.onOpen()

    def onOpen(self):
        with open("Preferences.txt", 'r') as f:
            Preferences = f.read().split()

        ftypes = [('HDF5 files', '*.h5')]
        input_file = filedialog.Open(self.parent, filetypes=ftypes)
        File = input_file.show()

        h5window = Toplevel()
        h55 = H5_Selector(h5window, Preferences, (File, 'data'))
        print('waiting')
        self.parent.wait_window(h5window)
        print('finished wait')
        print(h55.folder)
        self.folder = h55.folder


root = Tk()
FM = FakeModule(root)
root.mainloop()
#
# #
# # from abc import ABC
# # import pandas as pd
# # import random
# #
# #
# # class Patient_Viewer:
# #     """
# #     Patient Viewer Class for GCUH
# #     """
# #
# #     def __init__(self, patientid, database='gcuh', region='ap-southeast-2'):
# #         self.patientid = patientid
# #
# #         if database.lower() == 'gcuh' or database.lower() == 'deiddatetimes':
# #             self.database = 'deiddatetimes'
# #             self.PVclass = MIMIC(patientid, self.database, region)
# #
# #         self.PVclass.get_table()
# #
# #
# # class Patient_Viewer_Base:
# #     """
# #     Skeleton structure for Patient Viewer Class
# #     """
# #
# #     def __init__(self):
# #         print('Initialised!')
# #
# #     def get_table(self, *args) -> pd.DataFrame:
# #         """
# #         Returns patient-specific tables.
# #         """
# #         raise NotImplementedError
# #
# #
# # """
# # Patient Viewer Design for MIMIC
# # """
# #
# # patient_select_query = "SELECT {} FROM {}.{} WHERE subject_id = {}"
# #
# #
# # class MIMIC(Patient_Viewer_Base, ABC):
# #     """
# #     Second Level Class
# #     """
# #
# #     def __init__(self, patientid, database='mimic_iv', region='ap-southeast-2',
# #                  result_bucket="datarwe-sandbox-queryresults"):
# #         super().__init__(region=region, result_bucket=result_bucket)
# #         self.database = database
# #         self.patientid = patientid
# #
# #     def get_table(self):
# #         """
# #
# #         :return:
# #         """
# #         return 10
