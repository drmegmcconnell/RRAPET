"""
H5 Tree View Selector
Author: Meghan McConnell

"""
import h5py
from tkinter import Frame, Button
from tkinter.ttk import Treeview


class H5_Selector(Frame, object):
    """
    H5 Tree View Selector
    """

    def __init__(self, parent, preferences, string):
        Frame.__init__(self, parent)

        self.folder = None
        self.parent = parent
        self.Preferences = preferences
        self.string = string[0]
        self.str2 = string[1]
        self.tree = None
        self.value = 0
        self.branches = {}
        self.parent.title("HDF5/MAT-File Navigator")
        self.pack(fill='both', expand=1)
        self.config(bg='white')
        self._init_h5_UI()

    def _init_h5_UI(self):
        file = h5py.File(self.string, 'r')
        big_frame = Frame(self.parent, bg='white smoke', borderwidth=20)
        big_frame.pack(side='top')
        co1_frame = Frame(big_frame, bg='white', borderwidth=10, highlightcolor='grey', highlightthickness=1,
                          highlightbackground='black')
        co1_frame.pack(side='left', fill='y')

        F = Frame(co1_frame)
        F.pack(side='left', fill='y')

        self.tree = Treeview(F)
        self.tree.column("#0", width=400, minwidth=270)
        self.tree.heading("#0", text="Name", anchor='w')
        file.visit(self._insert_branch)
        self.tree.pack(side='top', fill='x')

        btn = Button(big_frame, text="Ok", command=self.get_branch)
        btn.pack(side='right', anchor='e')

    def _insert_branch(self, name):
        split_name = list(name.split('/'))
        N = len(split_name)
        prv_lvl = ""

        if N == 1:
            self.branches[name] = self.tree.insert("", "end", text=split_name[0])
        elif N > 1:
            for i in range(N-1):
                prv_lvl += split_name[i] + '/'
            prv_lvl = prv_lvl[:-1]   # Drop final slash
            self.branches[name] = self.tree.insert(self.branches[prv_lvl], "end", text=split_name[-1],
                                                   tags=name)

    def get_branch(self):
        """
        Select an item
        """

        curItem = self.tree.focus()
        x2 = self.tree.item(curItem)
        self.folder = x2.get('tags')
        if len(self.folder) != 0:
            self.parent.destroy()
