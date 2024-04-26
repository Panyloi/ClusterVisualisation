from matplotlib.axes._axes import Axes
from matplotlib.text import Text
from matplotlib.lines import Line2D

from .state import *

class ArrowArtist(Line2D, StateLinker):

    def __init__(self, ax: Axes, id: int, x: float, y: float, rfx: float, rfy: float, shx: float, shy: float, 
                 parent_label: 'LabelArtist', val: str, **kwargs) -> None:
        
        # custom init
        self.ax           = ax
        self.id           = id
        self.val          = val
        self.parent_label = parent_label
        self.x            = x
        self.y            = y
        self.rfx          = rfx
        self.rfy          = rfy
        self.shx          = shx
        self.shy          = shy

        super().__init__([x + shx, rfx], [y + shy, rfy], picker=True, pickradius=5, zorder=70, color='black', **kwargs)
        
    def set(self, *, x: float | None = None, y: float | None = None, 
                 rfx: float | None = None, rfy: float | None = None,
                 shx: float | None = None, shy: float | None = None,
                 val: str | None = None):

        self.x   = x if   x   is not None else self.x
        self.y   = y if   y   is not None else self.y
        self.rfx = rfx if rfx is not None else self.rfx
        self.rfy = rfy if rfy is not None else self.rfy
        self.shx = shx if shx is not None else self.shx
        self.shy = shy if shy is not None else self.shy
        self.val = val if val is not None else self.val

        self._update_state()

        return super().set(xdata=[self.x + self.shx, self.rfx], ydata=[self.y + self.shy, self.rfy])
    
    def set_sh_by_raw(self, rx: float, ry: float):
        self.set(shx=rx-self.x, shy=ry-self.y)
        
    def get_shs(self) -> tuple[float, float]:
        return self.shx, self.shy
    
    def get_rfs(self) -> tuple[float, float]:
        return self.rfx, self.rfy
    
    def _update_state(self) -> None:
        self.state.set_arrow_ref_pos(self.parent_label.id, self.id, self.rfx, self.rfy)
        self.state.set_arrow_att_pos(self.parent_label.id, self.id, self.x+self.shx, self.y+self.shy)
        self.state.set_arrow_val(self.parent_label.id, self.id, self.val)
        
    def remove(self) -> None:
        super().remove()
        
        # delete arrow from parent label
        self.parent_label.arrows.pop(self.id)

        # delete arrow from state
        self.state.delete_arrow(self.parent_label.id, self.id)
        
    @staticmethod
    def arrow(ax: Axes, *args, **kwargs) -> 'ArrowArtist':
        la = ArrowArtist(ax, *args, **kwargs)
        ax.add_line(la)
        return la


class LabelArtist(Text, StateLinker):

    def __init__(self, ax: Axes, id: int, x=0, y=0, text='', **kwargs) -> None:
        
        # label dict id
        self.id = id
        self.ax = ax

        # state readout
        x, y = self.state.get_label_pos(self.id)
        text = self.state.get_label_text(self.id)

        super().__init__(x,
                         y, 
                         text,
                         picker=True,
                         zorder=100,
                         **kwargs)
        
        self.set_bbox(dict(boxstyle='round', pad=0.2, facecolor='white', edgecolor='black'))
        
        # arrow artists
        self.arrows: dict[int, ArrowArtist] = {}
        for arrow_id in self.state.get_label_arrows(self.id):
            atx, aty = self.state.get_arrow_att_point(self.id, arrow_id)
            rfx, rfy = self.state.get_arrow_ref_point(self.id, arrow_id)
            val = self.state.get_arrow_val(self.id, arrow_id)
            self.arrows[arrow_id] = ArrowArtist.arrow(ax, arrow_id, x, y, rfx, rfy, atx-x, aty-y, self, val)

    def set_position(self, xy) -> None:
        super().set_position(xy)

        # arrows update
        x, y = xy
        for arrow in self.arrows.values():
            arrow.set(x=x, y=y)

        self.state.set_label_pos(self.id, x, y)

    def set_text(self, new_text):
        super().set_text(new_text)
        self.state.set_label_text(self.id, new_text)

    def remove(self) -> None:
        super().remove()
        self.state.delete_label(self.id)
        for arrow in self.arrows.values():
            arrow.remove()
    
    @staticmethod
    def text(ax: Axes, id, **kwargs) -> 'LabelArtist':
        effective_kwargs = {
            'verticalalignment': 'center',
            'horizontalalignment': 'center',
            'transform': ax.transData,
            'clip_on': False,
            **kwargs,
        }
        t = LabelArtist(ax, id, **effective_kwargs)
        t.set_clip_path(ax.patch)
        ax._add_text(t)
        return t
    
    @staticmethod
    def get_by_id(ax: Axes, id: int) -> 'None | LabelArtist':
        children = ax.get_children()
        for child in children:
            if isinstance(child, LabelArtist):
                if child.id == id:
                    return child

    def get_state(self) -> tuple:
        return self.id, self.get_position()

    def set_state(self, s) -> None:
        return self.set_position(*s)
    

# ------------------------------ DRAW DEFINITION ----------------------------- #

def draw(self, ax: Axes) -> None:
        # draw points
        for culture_name in self.data['data'].keys():
            ax.scatter(self.data['data'][culture_name]['x'], self.data['data'][culture_name]['y'])

        # draw labels
        for label_id in self.data['labels_data'].keys():
            LabelArtist.text(ax, label_id)

State.draw = draw