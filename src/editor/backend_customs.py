from matplotlib.backends._backend_tk import NavigationToolbar2Tk
import os
import matplotlib as mpl
import tkinter

class SavePltLinker:
    
    ax = None
    fig = None

    @classmethod
    def link_ax_fig(cls, ax, fig):
        cls.ax = ax
        cls.fig = fig

    @classmethod
    def get_ax_fig(cls):
        return cls.ax, cls.fig
    

def new_save_figure(self, *args):
    filetypes = self.canvas.get_supported_filetypes().copy()
    default_filetype = self.canvas.get_default_filetype()

    default_filetype_name = filetypes.pop(default_filetype)
    sorted_filetypes = ([(default_filetype, default_filetype_name)]
                        + sorted(filetypes.items()))
    tk_filetypes = [(name, '*.%s' % ext) for ext, name in sorted_filetypes]

    defaultextension = ''
    initialdir = os.path.expanduser(mpl.rcParams['savefig.directory'])
    initialfile = self.canvas.get_default_filename()
    fname = tkinter.filedialog.asksaveasfilename(
        master=self.canvas.get_tk_widget().master,
        title='Save the figure',
        filetypes=tk_filetypes,
        defaultextension=defaultextension,
        initialdir=initialdir,
        initialfile=initialfile,
        )

    if fname in ["", ()]:
        return

    if initialdir != "":
        mpl.rcParams['savefig.directory'] = (
            os.path.dirname(str(fname)))
    try:
        pax, pfig = SavePltLinker.get_ax_fig()
        bbox = pax.get_tightbbox().transformed(pfig.dpi_scale_trans.inverted())
        self.canvas.figure.savefig(fname, bbox_inches=bbox, pad_inches=0)
    except Exception as e:
        tkinter.messagebox.showerror("Error saving file", str(e))


base_save_figure = NavigationToolbar2Tk.save_figure
NavigationToolbar2Tk.save_figure = new_save_figure
