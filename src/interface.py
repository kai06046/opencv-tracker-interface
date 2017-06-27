import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import askyesno, askokcancel, showerror, showwarning, showinfo
from src.common import *

class Interface(object):

    # ask quit function
    def _ask_quit(self, title='Quit', string='Are you sure you want to quit?', icontype='warning'):

        self.root = tk.Tk()
        self.root.withdraw()
        result = askyesno(title, string, icon=icontype)
        self.root.destroy()
        self.root.mainloop()                    

        return result

    # ask cancel type name
    def _ask_cancel(self, title='Cancel', string='Are you sure you want to cancel?', icontype='info'):

        if askyesno(title, string, icon=icontype):
            self._askname.destroy()
            self._init_bbox.pop()

	# quit function for string entry
    def _quit_add_name(self, name):

        unique_char = list(set(name))
        # append entry to object name
        if (len(unique_char) == 1 and unique_char[0] == ' ') or len(unique_char) == 0:
            showerror('Error', 'Please enter a name')
            self.cb['values'] = [''] + list(self.cb['values'])
            self.cb.current(0)
        elif name not in self.object_name:
            if name in self.deleted_name:
                self.deleted_name.pop(self.deleted_name.index(name))
            self.object_name.append(name)
            self._askname.destroy()
        else:
            showerror('Error', '%s is already existed' % name)

    # method for typing on combobox list
    def _update_values(self, evt):
        # add entered text to combobox list of values
        widget = evt.widget           # get widget
        txt = widget.get()            # get current text
        vals = widget.cget('values')  # get values
         
        if not vals:
            widget.configure(values = (txt, ))
        elif txt not in vals:
            widget.configure(values = vals + (txt, ))
             
        return 'break'  # don't propagate event

    # after drawing new bounding box, ask user to enter object name
    def _add_name(self):

        self._askname = tk.Tk()
        self._askname.wm_title("Ask name")
        self._askname.geometry('240x80')
        
        center(self._askname) # center the widget
        cbp1 = ttk.Labelframe(self._askname, text='Type or choose a name for the object')
        
        cbp1.pack(side=tk.TOP, fill=tk.BOTH)
        self._askname.grab_current() # make it modal dialog

        self.cb = ttk.Combobox(cbp1, values = self.deleted_name)
        if len(self.deleted_name) > 0:
            self.cb.current(0)
        
        self.cb.focus_force()
        
        self.cb.bind('<Return>', self._update_values)
        self.cb.bind("<Return>", lambda event: self._quit_add_name(self.cb.get()))
        
        self.cb.pack(side=tk.TOP)
        btn = tk.Button(self._askname, text='Submit', command=(lambda: self._quit_add_name(self.cb.get())))
        btn.pack(side=tk.TOP)

        self._askname.protocol('WM_DELETE_WINDOW', self._ask_cancel)
        self._askname.mainloop()

    # ask whether to add potential target
    def _ask_add_box(self):

        if len(self._pot_rect) != 0:    
            self.root = tk.Tk()
            self.root.withdraw()
            result = askyesno('Add bounding box', 'Do you wanna add a bouding box?', icon='info')
            self.root.destroy()
            self.root.mainloop()
            if result:
                self._add_bboxes()

    # ask whether to delele box
    def _ask_delete_box(self):

        self.root = tk.Tk()
        self.root.withdraw()
        result = askyesno('Delete', 'Do you wanna detele %s' % self.object_name[self._n], icon='warning')
        if result:
            self._del_method()
            self.root.destroy()
        else:
            self.root.destroy()
        self.root.mainloop()                    

        return result
    # ask whether to delele box
    def _ask_retarget_box(self):

        self.root = tk.Tk()
        self.root.withdraw()
        result = askyesno('Retarget', 'Do you wanna retarget %s' % self.object_name[self._n], icon='warning')
        if result:
            self._retarget_bboxes()
            self.root.destroy()
        else:
            self.root.destroy()
        self.root.mainloop()                    

        return result
    # delete box method
    def _del_method(self):
        # update model if delete a object
        self._update_model()
        # delete selected bounding box
        self._bboxes = np.delete(self._bboxes, self._n, axis=0)
        self._len_bbox -= 1
        self._init_bbox.pop()
        if self._stop_obj:
            self._stop_obj.pop(self._stop_obj.index(self._n))
        self.deleted_name.append(self.object_name.pop(self._n))
        self._initialize_tracker()
        # update ROI
        self._roi = [convert(a[0], a[1], a[2], a[3]) for a in self._bboxes]

        self._n = 0       

    # show warning
    def alert(self, string='Please at least add one target'):
        self.root = tk.Tk()
        self.root.withdraw()
        showwarning('Alert', string)
        self.root.destroy()
        self.root.mainloop()

    # show help setting
    def help(self):
        self.root = tk.Tk()
        center(self.root) # center the widget
        self.root.geometry('280x180')
        # self.root.withdraw()
        self.root.title('Settings')
        self.root.resizable(0, 0)

        ACTION = ['Add bounding box', 'Delete bounding box', 'Jump to specific frame', 'Retarget bounding box', 'Pause/Continue', 'Close the program',
                 'Go to previous frame', 'Go to next frame', 'Switch of the auto add beetle model', 'Switch on/off of auto retarget', 'Switch on/off of showing rat']
        HOTKEY = ['a', 'd', 'j', 'r', 'Space', 'Esc', 'LEFT', 'RIGHT', 'b',  '1/2/3/4', 'z']

        hotkey = ttk.LabelFrame(self.root, text="Hotkey")
        action = ttk.LabelFrame(self.root, text="Action")

        # action description section
        for i, a in enumerate(ACTION):
            ttk.Label(action, text=a).grid(column=0, row=i, sticky=tk.W)
            ttk.Label(hotkey, text=HOTKEY[i]).grid(column=0, row=i)

        hotkey.pack(side=tk.LEFT, fill=tk.BOTH)
        action.pack(side=tk.LEFT, fill=tk.BOTH)
        # self.root.destroy()
        self.root.mainloop()