import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from matplotlib.figure import Figure
from matplotlib.backend_bases import FigureCanvasBase, PickEvent, MouseEvent
from matplotlib.axes._axes import Axes
from matplotlib.text import Text
from matplotlib.patches import FancyArrow
from enum import Enum
from abc import ABC, abstractmethod
import logging


def KeyErrorWrap(default):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                return result
            except KeyError as e:
                logging.error(f"{e}")
                return default
        return wrapper
    return decorator


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
            label_data: dict = self.data['labels_data'][label_id]

            if label_data['visible']:
                LabelArtist.text(ax, label_id, picker=True)

    @KeyErrorWrap((0, 0))
    def get_label_pos(self, id: int) -> tuple[float]:
        return self.data['labels_data'][id]['x'], self.data['labels_data'][id]['y']
    
    @KeyErrorWrap("")
    def get_label_text(self, id: int) -> str:
        return self.data['labels_data'][id]['text']
        
    @KeyErrorWrap(0)
    def get_label_number_of_lines(self, id: int) -> int:
        return len(self.data['labels_data'][id]['ref_points'])
        
    @KeyErrorWrap({})
    def get_label_arrow_data(self, id: int, arrow_id: int) -> dict:
        ret = {
            'ref_points': self.data['labels_data'][id]['ref_points'][arrow_id],
            'att_points': self.data['labels_data'][id]['att_points'][arrow_id],
            'ref_points_vals': self.data['labels_data'][id]['ref_points_vals'][arrow_id],
        }
        return ret
        
    @KeyErrorWrap({})
    def get_label_visibilities(self, id: int) -> dict:
        ret = {
            'visible': self.data['labels_data'][id]['visible'],
            'line_visible': self.data['labels_data'][id]['line_visible'],
            'ref_points_val_visible': self.data['labels_data'][id]['ref_points_val_visible']
        }
        
    @KeyErrorWrap(None)
    def set_label_pos(self, id: int, x: float, y: float) -> None:
        self.data['labels_data'][id]['x'] = x
        self.data['labels_data'][id]['y'] = y

    @KeyErrorWrap(None)
    def set_label_text(self, id: int, text: str) -> None:
        self.data['labels_data'][id]['text'] = text

    def get_raw(self) -> dict:
        return self.data
    
    def set_raw(self) -> None:
        return self.data


class StateLinker:
    """ State class linker
    
    Every class that should have access to global editor state should deriviate
    from this class. After that class variable with state is accessible.

    """

    state: State = None

    @classmethod
    def link_state(cls, lstate):
        cls.state = lstate


# ---------------------------- ARTIST DERIVIATIONS --------------------------- #


class LabelArtist(Text, StateLinker):

    def __init__(self, ax, id: int, x=0, y=0, text='', color=None,
                 verticalalignment='baseline', 
                 horizontalalignment='left', 
                 multialignment=None,
                 fontproperties=None,
                 rotation=None, 
                 linespacing=None, 
                 rotation_mode=None, 
                 usetex=None, 
                 wrap=False, 
                 transform_rotates_text=False, 
                 *, 
                 parse_math=None, 
                 **kwargs) -> None:
        
        # label dict id
        self.id = id
        self.ax = ax

        # state readout
        x, y = self.state.get_label_pos(self.id)
        text = self.state.get_label_text(self.id)

        super().__init__(x,
                         y, 
                         text, 
                         color,
                         verticalalignment, 
                         horizontalalignment, 
                         multialignment, 
                         fontproperties, 
                         rotation, 
                         linespacing, 
                         rotation_mode, 
                         usetex, 
                         wrap, 
                         transform_rotates_text, 
                         parse_math=parse_math, 
                         **kwargs)
        
        # line artists
        self.arrows = {}
        for arrow_id in range(self.state.get_label_number_of_lines(self.id)):
            arrow_data = self.state.get_label_arrow_data(self.id, arrow_id)
            sx, sy = arrow_data['att_points']
            rfx, rfy = arrow_data['ref_points']
            self.arrows[arrow_id] = FancyArrow(x+sx, y+sx, rfx-(x+sx), rfy-(y+sy))
            ax.add_patch(self.arrows[arrow_id])

    def set_position(self, xy) -> None:
        super().set_position(xy)

        # arrows update
        x, y = xy
        for arrow_id in self.arrows.keys():
            arrow = self.arrows[arrow_id]
            arrow_data = self.state.get_label_arrow_data(self.id, arrow_id)
            sx, sy = arrow_data['att_points']
            rfx, rfy = arrow._dx + arrow._x, arrow._dy + arrow._y
            arrow.set_data(x=x+sx, y=y+sy, dx=rfx-(x+sx), dy=rfy-(y+sy))

    def set_text(self, new_text):
        super().set_text(new_text)
        self.state.set_label_text(self.id, new_text)

    def get_shifts(self) -> tuple[float]:
        pass

    def remove(self) -> None:
        super().remove()
        for arrow in self.arrows.values():
            arrow.remove()
    
    @staticmethod
    def text(ax: Axes, id, **kwargs) -> 'LabelArtist':
        effective_kwargs = {
            'verticalalignment': 'baseline',
            'horizontalalignment': 'left',
            'transform': ax.transData,
            'clip_on': False,
            **kwargs,
        }
        t = LabelArtist(ax, id, **effective_kwargs)
        t.set_clip_path(ax.patch)
        ax._add_text(t)
        return t
    
    @staticmethod
    def get_by_id(ax: Axes, id) -> 'LabelArtist':
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
    def draw(self) -> None:
        logging.info(f"{self.__class__} is drawing.")

    @abstractmethod
    def undraw(self) -> None:
        logging.info(f"{self.__class__} is undrawing.")
        self.vem.deconstruct()
        self.cem.disconnect()
        self.vm.fig.canvas.flush_events()

    def change_view(self, view_id: ViewsEnum, *args):
        self.undraw()
        self.vm.get_view(view_id).draw()


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
    
    def __init__(self, parent_view: View, axes: list[float], label: str, callback: callable) -> None:
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

    def __init__(self, parent_view: View, axes: list[float], label: str, callback: callable) -> None:
        super().__init__(parent_view, axes, label, lambda ev: callback())

    def remove(self):
        return super().remove()
    
    def refresh(self) -> None:
        return super().refresh()
    

# move to observer pattern?
class UpdateableTextBox(ViewTextBox):

    def __init__(self, parent_view: View, axes: list[float], label: str, update: callable, submit: callable) -> None:
        super().__init__(parent_view, axes, label)
        self.update = update
        self.box_ref.on_submit(submit)

    def remove(self) -> None:
        return super().remove()
    
    def refresh(self) -> None:
        super().refresh()
        self.box_ref.set_val(self.update())
