from matplotlib.backends._backend_tk import NavigationToolbar2Tk, ToolTip
from matplotlib.backend_bases import Event
from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.figure import Figure
import os
import matplotlib as mpl
import tkinter

from .state import *


class RefreshEvent(Event):

    rf = None

    def __init__(self, name: str, canvas: FigureCanvasBase, guiEvent=None) -> None:
        super().__init__(name, canvas, guiEvent)
           
    def trigger_refresh_event(self, *args, **kwargs):
        self._process()


# register new event in the library
FigureCanvasBase.events.append('refresh_event')


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
    

def editor_state_file_save_choose(fig: Figure):
    filetypes = fig.canvas.get_supported_filetypes().copy()
    filetypes['json'] = 'Javascript Object Notation'
    default_filetype = 'json'

    default_filetype_name = filetypes.pop(default_filetype)
    sorted_filetypes = ([(default_filetype, default_filetype_name)]
                        + sorted(filetypes.items()))
    
    tk_filetypes = [(name, '*.%s' % ext) for ext, name in sorted_filetypes]

    defaultextension = ''
    initialdir = os.path.expanduser(mpl.rcParams['savefig.directory'])
    initialfile = "Editor_1.json"
    
    fname = tkinter.filedialog.asksaveasfilename(
        master=fig.canvas.get_tk_widget().master,
        title='Save editor state',
        filetypes=tk_filetypes,
        defaultextension=defaultextension,
        initialdir=initialdir,
        initialfile=initialfile,
        )
    return fname


def editor_state_file_load_choose(fig: Figure):
    filetypes = dict()
    filetypes['json'] = 'Javascript Object Notation'
    default_filetype = 'json'

    default_filetype_name = filetypes.pop(default_filetype)
    sorted_filetypes = ([(default_filetype, default_filetype_name)]
                        + sorted(filetypes.items()))
    
    tk_filetypes = [(name, '*.%s' % ext) for ext, name in sorted_filetypes]

    defaultextension = ''
    initialdir = os.path.expanduser(mpl.rcParams['savefig.directory'])
    initialfile = ""
    
    fname = tkinter.filedialog.askopenfilename(
        title='Load editor state',
        filetypes=tk_filetypes,
        defaultextension=defaultextension,
        initialdir=initialdir,
        initialfile=initialfile
    )
    return fname


def editor_save_cb(fig: Figure, state: State):
    fname = editor_state_file_save_choose(fig)
    state.save_state_to_file(fname)
    logging.info(f"Saved editor state to {fname}")
    
    
def editor_load_cb(fig: Figure, state: State):
    fname = editor_state_file_load_choose(fig)
    state.load_state_from_file(fname)
    RefreshEvent.rf.trigger_refresh_event()
    logging.info(f"Loaded editor state from {fname}")


def custom_buttons_setup(fig: Figure, state: State):
    tb = fig.canvas.manager.toolbar
    tb: NavigationToolbar2Tk
    b1 = tb._Button("sv", None, False, lambda : editor_save_cb(fig, state))
    ToolTip.createToolTip(b1, "Save editor state")
    b2 = tb._Button("ld", None, False, lambda : editor_load_cb(fig, state))
    ToolTip.createToolTip(b2, "Load editor state")
    tb._Spacer()
    

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
