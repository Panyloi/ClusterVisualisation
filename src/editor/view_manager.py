from matplotlib._enums import CapStyle, JoinStyle
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from matplotlib.figure import Figure
from matplotlib.backend_bases import FigureCanvasBase, PickEvent, MouseEvent
from matplotlib.axes._axes import Axes
from matplotlib.text import Text
from matplotlib.patches import FancyArrow
from matplotlib.lines import Line2D
from enum import Enum
from abc import ABC, abstractmethod
import logging
from typing import Literal, Callable, Type, TypeVar, ParamSpec

T = TypeVar('T')
P = ParamSpec('P')

def KeyErrorWrap(default) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                result = func(*args, **kwargs)
                return result
            except KeyError as e:
                logging.error(f"{e}")
                return default
        return wrapper
    return decorator
  
def get_artists_by_type(ax: Axes, artist_type: Type[T]) -> list[T]:
    children = ax.get_children()
    artists = []
    for child in children:
        if isinstance(child, artist_type):
            artists.append(child)
    return artists

# ----------------------------------- STATE ---------------------------------- #

class State:
    """ Wrapper class for editor data and editor state. Made for providing getters, setters and options storage """

    def __init__(self, data: dict) -> None:
        """ Data init
        
        Parameters
        ----------
        data: dict
            Data in proper editor format
        
        """
        
        self.data = data

    def draw(self, ax: Axes) -> None:
        # draw points
        for culture_name in self.data['data'].keys():
            ax.scatter(self.data['data'][culture_name]['x'], self.data['data'][culture_name]['y'])

        # draw labels
        for label_id in self.data['labels_data'].keys():
            LabelArtist.text(ax, label_id)

    # ---------------------------------- GETTERS --------------------------------- #

    @KeyErrorWrap("")
    def get_label_text(self, label_id: int) -> str:
        return self.data['labels_data'][label_id]['text']
    
    @KeyErrorWrap((0, 0))
    def get_label_pos(self, label_id: int) -> tuple[float, float]:
        return self.data['labels_data'][label_id]['x'], self.data['labels_data'][label_id]['y']
    
    @KeyErrorWrap({})
    def get_label_arrows(self, label_id: int) -> dict:
        return self.data['labels_data'][label_id]['arrows']
    
    @KeyErrorWrap({})
    def get_label_arrow(self, label_id: int, arrow_id: int) -> dict:
        return self.data['labels_data'][label_id]['arrows'][arrow_id]
    
    @KeyErrorWrap((0, 0))
    def get_arrow_ref_point(self, label_id: int, arrow_id: int) -> tuple[float, float]:
        return self.data['labels_data'][label_id]['arrows'][arrow_id]['ref_x'],\
               self.data['labels_data'][label_id]['arrows'][arrow_id]['ref_y']

    @KeyErrorWrap((0, 0))
    def get_arrow_att_point(self, label_id: int, arrow_id: int) -> tuple[float, float]:
        return self.data['labels_data'][label_id]['arrows'][arrow_id]['att_x'],\
               self.data['labels_data'][label_id]['arrows'][arrow_id]['att_y']
    
    @KeyErrorWrap("")
    def get_arrow_val(self, label_id: int, arrow_id: int) -> str:
        return self.data['labels_data'][label_id]['arrows'][arrow_id]['val']
    
    # ---------------------------------- SETTERS --------------------------------- #

    @KeyErrorWrap(None)
    def set_label_text(self, label_id: int, text: str) -> None:
        self.data['labels_data'][label_id]['text'] = text
        
    @KeyErrorWrap(None)
    def set_label_pos(self, label_id: int, x: float, y: float) -> None:
        self.data['labels_data'][label_id]['x'] = x
        self.data['labels_data'][label_id]['y'] = y

    @KeyErrorWrap(None)
    def set_arrow_att_pos(self, label_id: int, arrow_id: int, att_x: float, att_y: float) -> None:
        self.data['labels_data'][label_id]['arrows'][arrow_id]['att_x'] = att_x
        self.data['labels_data'][label_id]['arrows'][arrow_id]['att_y'] = att_y

    @KeyErrorWrap(None)
    def set_arrow_ref_pos(self, label_id: int, arrow_id: int, ref_x: float, ref_y: float) -> None:
        self.data['labels_data'][label_id]['arrows'][arrow_id]['ref_x'] = ref_x
        self.data['labels_data'][label_id]['arrows'][arrow_id]['ref_y'] = ref_y

    @KeyErrorWrap(None)
    def set_arrow_val(self, label_id: int, arrow_id: int, val: str) -> None:
        self.data['labels_data'][label_id]['arrows'][arrow_id]['val'] = val
        
    # ------------------------------------ ADD ----------------------------------- #
    
    @KeyErrorWrap(-1)
    def add_empty_label(self) -> int:
        nid = max(self.data['labels_data'].keys()) + 1
        self.data['labels_data'][nid] = \
        {
            'text': "...",
            'x': 0,
            'y': 0,
            'arrows': {}
        }
        return nid
    
    @KeyErrorWrap(-1)
    def add_empty_arrow(self, label_id: int) -> int:
        if self.get_label_arrows(label_id):
            nid = max(self.get_label_arrows(label_id).keys()) + 1
        else:
            nid = 0
        self.data['labels_data'][label_id]['arrows'][nid] = \
        {
            'ref_x': 0,
            'ref_y': 0,
            'att_x': self.get_label_pos(label_id)[0],
            'att_y': self.get_label_pos(label_id)[1],
            'val': ""
        }
        return nid
    
    # ---------------------------------- DELETE ---------------------------------- #
    
    @KeyErrorWrap(None)
    def delete_label(self, label_id: int) -> None:
        self.data['labels_data'].pop(label_id)
        
    @KeyErrorWrap(None)
    def delete_arrow(self, label_id: int, arrow_id: int) -> None:
        self.data['labels_data'][label_id]['arrows'].pop(arrow_id)

    # ----------------------------------- MISC ----------------------------------- #

    def get_raw(self) -> dict:
        return self.data
    
    def set_raw(self, data) -> None:
        self.data = data


class StateLinker:
    """ State class linker
    
    Every class that should have access to global editor state should deriviate
    from this class. After that class variable with state is accessible.

    """

    state: State

    @classmethod
    def link_state(cls, lstate):
        cls.state = lstate


# ---------------------------- ARTIST DERIVIATIONS --------------------------- #

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

        super().__init__([x + shx, rfx], [y + shy, rfy], picker=True, pickradius=5, zorder=70, **kwargs)
        
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
        
    def get_shs(self) -> tuple[float, float]:
        return self.shx, self.shy
    
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


# ----------------------------- TOP LEVEL CLASSES ---------------------------- #


class ViewsEnum(Enum):
    HOME   = 0
    LABELS = 1
    ARROWS = 2


class ViewManager:

    def __init__(self, fig: Figure, ax: Axes) -> None:
        self.fig = fig
        self.ax  = ax
        self.views: list[View] = []
    
    def register_views(self, views: list) -> None:
        self.views = views

    def get_view(self, view_id: ViewsEnum) -> 'View':
        return self.views[view_id.value]
    
    def run(self) -> None:
        self.views[ViewsEnum.HOME.value].draw()
        self.views[ViewsEnum.HOME.value].state.draw(self.ax)


class ViewElementManager:
    
    def __init__(self) -> None:
        self.elements: list['ViewElement'] = []

    def add(self, el: 'ViewElement') -> 'ViewElement':
        self.elements.append(el)
        return el

    def refresh(self) -> None:
        for view_element in self.elements:
            view_element.refresh()
        
    def deconstruct(self) -> None:
        for view_element in self.elements:
            view_element.remove()
        self.elements.clear()


class CanvasEventManager:

    def __init__(self, canvas: FigureCanvasBase) -> None:
        self.canvas = canvas
        self.events: list[int] = []

    def add(self, ev_id: int) -> None:
        self.events.append(ev_id)

    def disconnect(self) -> None:
        for ev_id in self.events:
            self.canvas.mpl_disconnect(ev_id)
        self.events.clear()


class View(ABC, StateLinker):
    
    def __init__(self, view_manager: ViewManager) -> None:
        self.vm = view_manager
        self.vem = ViewElementManager()
        self.cem = CanvasEventManager(self.vm.fig.canvas)

    @abstractmethod
    def draw(self, *args, **kwargs) -> None:
        logging.info(f"{self.__class__} is drawing.")

    @abstractmethod
    def undraw(self) -> None:
        logging.info(f"{self.__class__} is undrawing.")
        self.vem.deconstruct()
        self.cem.disconnect()
        self.vm.fig.canvas.flush_events()

    def change_view(self, view_id: ViewsEnum, *args, **kwargs):
        self.undraw()
        self.vm.get_view(view_id).draw(*args, **kwargs)


class ViewElement(ABC, StateLinker):

    def __init__(self) -> None:
        logging.info(f"{self.__class__} is createing.")
    
    @abstractmethod
    def remove(self) -> None:
        logging.info(f"{self.__class__} is removeing.")

    @abstractmethod
    def refresh(self) -> None:
        logging.info(f"{self.__class__} is refreshing.")


# --------------------------- MIDDLE LEVEL CLASSES --------------------------- #


class ViewButton(ViewElement):
    
    def __init__(self, parent_view: View, axes: list[float], label: str, callback: Callable) -> None:
        super().__init__()
        self.pv         = parent_view
        self.button_ax  = parent_view.vm.fig.add_axes(axes)
        self.button_ref = Button(self.button_ax, label)
        self.button_ref.on_clicked(callback)
        
    @abstractmethod
    def remove(self):
        super().remove()
        self.button_ref.disconnect_events()     
        self.pv.vm.fig.delaxes(self.button_ax)

    @abstractmethod
    def refresh(self) -> None:
        return super().refresh()


class ViewTextBox(ViewElement):

    def __init__(self, parent_view: View, axes: list[float], label: str) -> None:
        super().__init__()
        self.pv       = parent_view
        self.label_ax = parent_view.vm.fig.add_axes(axes)
        self.box_ref  = TextBox(self.label_ax, '', initial=label)

    @abstractmethod
    def remove(self):
        super().remove() 
        self.pv.vm.fig.delaxes(self.label_ax)

    @abstractmethod
    def refresh(self) -> None:
        return super().refresh()

# ------------------------------ OUTPUT CLASSES ------------------------------ #

class ChangeViewButton(ViewButton):
    
    def __init__(self, parent_view: View, axes: list[float], label: str, new_view: ViewsEnum) -> None:
        super().__init__(parent_view, axes, label, lambda ev: parent_view.change_view(new_view, ev))
        
    def remove(self):
        return super().remove()

    def refresh(self) -> None:
        return super().refresh()
    

class NormalButton(ViewButton):

    def __init__(self, parent_view: View, axes: list[float], label: str, callback: Callable) -> None:
        super().__init__(parent_view, axes, label, lambda ev: callback())

    def remove(self):
        return super().remove()
    
    def refresh(self) -> None:
        return super().refresh()
    

# move to observer pattern?
class UpdateableTextBox(ViewTextBox):

    def __init__(self, parent_view: View, axes: list[float], label: str, update: Callable, submit: Callable) -> None:
        super().__init__(parent_view, axes, label)
        self.update = update
        self.box_ref.on_submit(submit)

    def remove(self) -> None:
        return super().remove()
    
    def refresh(self) -> None:
        super().refresh()
        self.box_ref.set_val(self.update())
