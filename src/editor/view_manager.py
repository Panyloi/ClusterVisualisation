from matplotlib.figure import Figure
from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.axes._axes import Axes
from matplotlib.widgets import RadioButtons, Slider, Button, TextBox
from enum import Enum
from abc import ABC, abstractmethod
import logging
from typing import Callable

from .artists import *


# ----------------------------- TOP LEVEL CLASSES ---------------------------- #


class ViewsEnum(Enum):
    HOME   = 0
    LABELS = 1
    ARROWS = 2
    CLUSTER = 3


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
        self.events_stack: list['Event'] = []

    def add(self, event: 'UniqueEvent | SharedEvent | GlobalEvent | EmptyEvent') -> None:

        if isinstance(event, UniqueEvent):
            # disconnect all previous events
            for sevent in self.events_stack:
                if not isinstance(sevent, GlobalEvent):
                    sevent.disconnect()

        if isinstance(event, SharedEvent):
            # disconnect all events starting from first met non shared event
            fm = False
            for sevent in self.events_stack[::-1]:
                if not isinstance(sevent, SharedEvent):
                    fm = True
                if fm:
                    if not isinstance(sevent, GlobalEvent):
                        sevent.disconnect()

        if isinstance(event, GlobalEvent):
            # globals dont disconnect anything but they are inserted under the stack
            self.events_stack.insert(0, event)
            return

        self.events_stack.append(event)

    def _reconect_events(self) -> None:
        # find next event group to activate
        shared_group = False
        for event in self.events_stack[::-1]:

            if isinstance(event, EmptyEvent):
                break

            # if no shared gourp activate just this event
            if isinstance(event, UniqueEvent):
                if not shared_group:
                    event.reconnect()
                    return
                else:
                    # end of shared group
                    break

            # start or continue the shared group
            if isinstance(event, SharedEvent):
                # find other shared events
                shared_group = True
                event.reconnect()

    def _clear_empty_events(self) -> None:
        while self.events_stack:
            if isinstance(self.events_stack[-1], EmptyEvent):
                self.events_stack.pop()
                continue
            return

    def disconnect_unique(self) -> None:

        if not self.events_stack:
            return

        # if not maching just call disconnect
        if not isinstance(self.events_stack[-1], UniqueEvent):
            return self.disconnect()
        
        self.events_stack.pop().disconnect()
        self._clear_empty_events()
        self._reconect_events()

    def disconnect_shared(self) -> None:

        if not self.events_stack:
            return
        
        if not isinstance(self.events_stack[-1], SharedEvent):
            return self.disconnect()
        
        while self.events_stack:
            if isinstance(self.events_stack[-1], SharedEvent):
                self.events_stack.pop().disconnect()
                continue
            break

        self._clear_empty_events()
        self._reconect_events()

    def disconnect(self) -> None:
        for event in self.events_stack:
            event.disconnect()
        self.events_stack.clear()


class Event(ABC):

    canvas: FigureCanvasBase

    @abstractmethod
    def __init__(self, ev_type: str, ev_callback: Callable) -> None:
        self.ev_type = ev_type
        self.ev_callback = ev_callback
        self.id = self.canvas.mpl_connect(ev_type, ev_callback)

    @classmethod
    def set_canvas(cls, canvas: FigureCanvasBase) -> None:
        cls.canvas = canvas

    def reconnect(self) -> None:
        self.canvas.mpl_disconnect(self.id) # source code says there is no error if self.id does not exist c:
        self.id = self.canvas.mpl_connect(self.ev_type, self.ev_callback)

    def disconnect(self) -> None:
        if self.canvas is None:
            return
        self.canvas.mpl_disconnect(self.id)


class GlobalEvent(Event):
    """
    GlobalEvent is not disconnectable by other event types
    """

    def __init__(self, ev_type: str, ev_callback: Callable) -> None:
        super().__init__(ev_type, ev_callback)


class UniqueEvent(Event):
    """
    UniqueEvent desconnects all other events
    """

    def __init__(self, ev_type: str, ev_callback: Callable) -> None:
        super().__init__(ev_type, ev_callback)


class SharedEvent(Event):
    """
    SharedEvent disconnects all previous events that don't belong to this event shared group
    """

    def __init__(self, ev_type: str, ev_callback: Callable) -> None:
        super().__init__(ev_type, ev_callback)

    
class EmptyEvent(Event):
    """
    EmptyEvent does nothing. It's used to divide SharedEvent groups
    """

    def __init__(self) -> None:
        pass

    def reconnect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass


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
        self.button_cid = self.button_ref.on_clicked(callback)
        
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
    

class BlockingButton(ViewButton):

    def __init__(self, parent_view: View, axes: list[float], label: str, callback: Callable[[Callable[..., None]], None]) -> None:

        def reconnect_callback(*args, **kwargs):
            self.button_cid = self.button_ref.on_clicked(blocking_callback)

        def blocking_callback(*args, **kwargs):
            self.button_ref.disconnect(self.button_cid)
            callback(reconnect_callback)

        super().__init__(parent_view, axes, label, blocking_callback)

    def remove(self):
        return super().remove()
    
    def refresh(self) -> None:
        return super().refresh()
    

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


class ViewRadioButtons(ViewElement):

    def __init__(self, parent_view: View, axes: list[float], labels: list[str], callback: Callable) -> None:
        super().__init__()
        self.pv = parent_view
        self.ax = parent_view.vm.fig.add_axes(axes, frameon=False)
        self.ref = RadioButtons(self.ax, labels=labels, active=0)
        self.ref.on_clicked(callback)

    def remove(self):
        super().remove()
        self.ref.disconnect_events()
        self.pv.vm.fig.delaxes(self.ax)

    def refresh(self) -> None:
        return super().refresh()


class ViewSlider(ViewElement):

    def __init__(self, parent_view: View, axes: list[float], label: str,
                 valmin: float, valmax: float, callback: Callable) -> None:
        super().__init__()
        self.pv = parent_view
        self.ax = parent_view.vm.fig.add_axes(axes, frameon=False)
        self.ref = Slider(ax=self.ax, label=label, valmin=valmin, valmax=valmax, initcolor=None)
        self.ref.on_changed(callback)

    def remove(self):
        super().remove()
        self.ref.disconnect_events()
        self.pv.vm.fig.delaxes(self.ax)

    def refresh(self) -> None:
        return super().refresh()
