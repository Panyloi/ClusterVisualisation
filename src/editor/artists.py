from matplotlib.axes._axes import Axes
from matplotlib.text import Text
from matplotlib.lines import Line2D

from .backend_customs import *

class ArrowArtist(Line2D, StateLinker):
    """
    Class wrapper for Line2D to implement additional getters, 
    setters and linking with LabelArtist.
    """

    def __init__(self, ax: Axes, sid: int, x: float, y: float, 
                 rfx: float, rfy: float, shx: float, shy: float, 
                 parent_label: 'LabelArtist', val: str, **kwargs) -> None:
        """init
        
        Parameters
        ----------
        ax: Axes
            Main chart ax
        sid: int
            Id of the created label
        x, y: float
            Coordinates of parent label
        rfx, rfy: float
            Coordinates of the point arrow referse to
        shx, shy: float
            Shift of the attachment point
        parent_label: LabelArtist
            The parent label
        val: str
            The anotation string

        """
        
        # custom init
        self.ax           = ax
        self.id           = sid
        self.val          = val
        self.parent_label = parent_label
        self.x            = x
        self.y            = y
        self.rfx          = rfx
        self.rfy          = rfy
        self.shx          = shx
        self.shy          = shy
        s = self.state.get_arrow_size()

        super().__init__([x + shx, rfx], [y + shy, rfy], picker=True, 
                         pickradius=5, zorder=70, color='black', linewidth=s, **kwargs)
        
    def set(self, *, x: float | None = None, y: float | None = None, 
                 rfx: float | None = None, rfy: float | None = None,
                 shx: float | None = None, shy: float | None = None,
                 val: str | None = None):
        """
        Setter for all ArrowArtist attributes.

        Parameters
        ----------
        x, y: float
            Coordinates of parent label
        rfx, rfy: float
            Coordinates of the point arrow refers to
        shx, shy: float
            Shift of the attachment point
        val: str
            The annotation string

        """

        self.x   = x if   x   is not None else self.x
        self.y   = y if   y   is not None else self.y
        self.rfx = rfx if rfx is not None else self.rfx
        self.rfy = rfy if rfy is not None else self.rfy
        self.shx = shx if shx is not None else self.shx
        self.shy = shy if shy is not None else self.shy
        self.val = val if val is not None else self.val

        self._update_state()

        return super().set(xdata=[self.x + self.shx, self.rfx], ydata=[self.y + self.shy, self.rfy])
    
    def set_sh_by_raw(self, rx: float, ry: float) -> None:
        """set attachment shift by raw point on chart

        Parameters
        ----------
        rx, ry: float
            Coordinates of the chart point
            
        Notes
        -----
        This operation also snaps the shift to the closest outline of the label.
        
        """
        
        rbbx = self.parent_label.get_window_extent()
        bbx = self.ax.transData.inverted().transform(rbbx)
        x, y = rx, ry
        if y < bbx[0][1]:
            y = bbx[0][1]
        if y > bbx[1][1]:
            y = bbx[1][1]
        if x < bbx[0][0]:
            x = bbx[0][0]
        if x > bbx[1][0]:
            x = bbx[1][0]
        
        self.set(shx=x-self.x, shy=y-self.y)
        
    def get_shs(self) -> tuple[float, float]:
        """shift point values getter"""
        return self.shx, self.shy
    
    def get_rfs(self) -> tuple[float, float]:
        """reference point values getter"""
        return self.rfx, self.rfy
    
    def _update_state(self) -> None:
        """update all arrow attributes to global state"""
        self.state.set_arrow_ref_pos(self.parent_label.id, self.id, 
                                     self.rfx, self.rfy)
        self.state.set_arrow_att_pos(self.parent_label.id, self.id, 
                                     self.x+self.shx, self.y+self.shy)
        self.state.set_arrow_val(self.parent_label.id, self.id, self.val)

    def _update_size(self, size: float) -> None:
        """update arrow size"""
        self.set_linewidth(size)
        
    def remove(self) -> None:
        """remove arrow from chart and parent label dict"""
        super().remove()
        
        # delete arrow from parent label
        self.parent_label.arrows.pop(self.id)

        # delete arrow from state
        self.state.delete_arrow(self.parent_label.id, self.id)
        
    @staticmethod
    def arrow(ax: Axes, *args, **kwargs) -> 'ArrowArtist':
        """arrow creator
        
        Parameters
        ----------

        ax: Axes
            Main chart ax
        *args:
            args passed to ArrowArtist __init__
        **kwargs:
            kwargs passed to ArrowArtist __init__

        Return
        ------
        ArrowArtist:
            The created arrow
            
        """
        la = ArrowArtist(ax, *args, **kwargs)
        ax.add_line(la)
        return la
    
    @staticmethod
    def update_all_arrows_size(ax: Axes, size: float) -> None:
        """update all arrows"""
        for child in ax.get_children():
            if isinstance(child, ArrowArtist):
                child._update_size(size)
        ArrowArtist.state.set_arrow_size(size)
    
    @staticmethod
    def get_by_id(ax: Axes, sid: int) -> 'None | ArrowArtist':
        """arrow getter by state id"""
        children = ax.get_children()
        for child in children:
            if isinstance(child, ArrowArtist):
                if child.id == sid:
                    return child
        return None
    
    @staticmethod
    def get_all_arrows(ax: Axes):
        """arrow getter by state id"""
        children = ax.get_children()
        for child in children:
            if isinstance(child, ArrowArtist):
                yield child


class LabelArtist(Text, StateLinker):
    """
    Class wrapper for Text to implement additional getters, 
    setters and ArrowArtist linking.
    """

    def __init__(self, ax: Axes, sid: int, x: float = 0, y: float = 0, text='', **kwargs) -> None:
        """init
        
        Parameters
        ----------
        ax: Axes
            Main chart ax
        sid: int
            The created LabelArtist id
        x, y: float
            The label coordinates
        text: str
            The label text

        """

        self.id = sid
        self.ax = ax

        # state readout
        x, y = self.state.get_label_pos(self.id)
        text = self.state.get_label_text(self.id)
        size = self.state.get_label_size()

        super().__init__(x,
                         y, 
                         text,
                         picker=True,
                         zorder=100,
                         size=size,
                         **kwargs)
        
        self.set_bbox(dict(boxstyle='round', pad=0.2, facecolor='white', edgecolor='black'))
        
        # arrow artists
        self.arrows: dict[int, ArrowArtist] = {}
        for arrow_id in self.state.get_label_arrows(self.id):
            atx, aty = self.state.get_arrow_att_point(self.id, arrow_id)
            rfx, rfy = self.state.get_arrow_ref_point(self.id, arrow_id)
            val = self.state.get_arrow_val(self.id, arrow_id)
            self.arrows[arrow_id] = ArrowArtist.arrow(ax, arrow_id, x, y, rfx, rfy, 
                                                      atx-x, aty-y, self, val)

    def set_position(self, xy) -> None:
        """label position setter"""
        super().set_position(xy)

        # arrows update
        x, y = xy
        for arrow in self.arrows.values():
            arrow.set(x=x, y=y)

        self.state.set_label_pos(self.id, x, y)

    def set_text(self, new_text):
        """label text setter"""
        super().set_text(new_text)
        self.state.set_label_text(self.id, new_text)

    def update_fontsize(self, size: float) -> None:
        """label size local update"""
        self.set(fontsize=size)

    @staticmethod
    def update_all_labels_fontsize(ax: Axes, size: float) -> None:
        """update all labels"""
        for child in ax.get_children():
            if isinstance(child, LabelArtist):
                child.update_fontsize(size)
        LabelArtist.state.set_label_size(size)

    def remove(self) -> None:
        """removes label and all attached arrows"""
        super().remove()
        self.state.delete_label(self.id)
        dict_cpy = list(self.arrows.values()).copy()
        for arrow in dict_cpy:
            arrow.remove()
    
    @staticmethod
    def text(ax: Axes, sid: int, **kwargs) -> 'LabelArtist':
        """label creator
        
        Parameters
        ----------

        ax: Axes
            Main chart ax
        *args:
            args passed to LabelArtist __init__
        **kwargs:
            kwargs passed to LabelArtist __init__

        Return
        ------
        LabelArtist:
            The created label
            
        """

        effective_kwargs = {
            'verticalalignment': 'center',
            'horizontalalignment': 'center',
            'transform': ax.transData,
            'clip_on': False,
            **kwargs,
        }
        t = LabelArtist(ax, sid, **effective_kwargs)
        t.set_clip_path(ax.patch)
        ax._add_text(t)
        return t

    def get_state(self) -> tuple:
        """state getter for position undo operation"""
        return self.id, self.get_position()

    def set_state(self, s) -> None:
        """state setter for position undo operation"""
        return self.set_position(*s)
    
    @staticmethod
    def get_by_id(ax: Axes, sid: int) -> 'None | LabelArtist':
        """label getter by state id"""
        children = ax.get_children()
        for child in children:
            if isinstance(child, LabelArtist):
                if child.id == sid:
                    return child
        return None
    
    @staticmethod
    def get_all_labels(ax: Axes):
        """label getter by state id"""
        children = ax.get_children()
        for child in children:
            if isinstance(child, LabelArtist):
                yield child
    

# ------------------------------ DRAW DEFINITION ----------------------------- #

def draw(self, ax: Axes) -> None:

    # clear ax
    ax.clear()

    # draw points
    for culture_name in self.data['data'].keys():
        ax.scatter(self.data['data'][culture_name]['x'], self.data['data'][culture_name]['y'])

    # draw labels
    for label_id in self.data['labels_data'].keys():
        if isinstance(label_id, int):
            LabelArtist.text(ax, label_id)

    ax.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False)
    ax.tick_params(
        axis='y',
        which='both',
        left=False,
        right=False,
        labelleft=False)
    
    ax.bbox._bbox.x0 = 0.01
    ax.bbox._bbox.y0 = 0.15
    ax.bbox._bbox.x1 = 0.99
    ax.bbox._bbox.y1 = 0.99

    ax.set_xlim((-150, 150))
    ax.set_ylim((-150, 150))

    plt.draw()
    logging.info(f"State redraw")


State.draw = draw
