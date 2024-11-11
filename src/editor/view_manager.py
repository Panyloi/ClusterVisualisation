from enum import Enum
from abc import ABC, abstractmethod
import logging
from typing import Callable

from matplotlib.figure import Figure
from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.axes._axes import Axes
from matplotlib.widgets import RadioButtons, Slider, Button, TextBox, CheckButtons

from .artists import *


# ----------------------------- TOP LEVEL CLASSES ---------------------------- #


class ViewsEnum(Enum):
    """
    Enumeration of different views in the application.
    """
    HOME   = 0
    LABELS = 1
    ARROWS = 2
    HULLS = 3
    CLUSTER = 4
    AGGLOMERATIVE = 5
    DBSCAN  = 6
    CREATEHULL = 7
    REMOVELINE = 8


class ViewManager:
    """
    Manages different views in the application.
    """

    def __init__(self, fig: Figure, ax: Axes) -> None:
        self.fig = fig
        self.ax  = ax
        self.views: list[View] = []
        self.list_manager = CheckListManager(self)

    def register_views(self, views: list['View']) -> None:
        """
        Registers views with the manager.

        Parameters
        ----------
        views : list
            List of View objects to register.
        """
        self.views = views

    def get_view(self, view_id: ViewsEnum) -> 'View':
        """
        Gets a specific view by its ID.

        Parameters
        ----------
        view_id : ViewsEnum
            The ID of the view to retrieve.

        Returns
        -------
        View
            The requested view object.
        """
        return self.views[view_id.value]
    
    def run(self) -> None:
        """
        Runs the application, displaying the initial view.
        """
        self.views[ViewsEnum.HOME.value].draw()
        self.views[ViewsEnum.HOME.value].state.draw(self.ax)


class ViewElementManager:
    """
    Manages view elements within a view.
    """
    
    def __init__(self) -> None:
        self.elements: list['ViewElement'] = []

    def add(self, el: 'ViewElement') -> 'ViewElement':
        """
        Adds a view element to the manager.

        Parameters
        ----------
        el : ViewElement
            The view element to add.

        Returns
        -------
        ViewElement
            The added view element.
        """
        self.elements.append(el)
        return el

    def remove(self, el: 'ViewElement') -> None:
        el.remove()
        self.elements.remove(el)

    def refresh_connect(self, fig: Figure) -> None:
        fig.canvas.mpl_connect('refresh_event', self.refresh)

    def refresh(self, *args, **kwargs) -> None:
        """
        Refreshes all view elements. Only needed for TextBox elements.
        """
        for view_element in self.elements:
            view_element.refresh()

    def deconstruct(self) -> None:
        """
        Removes all view elements from the manager.
        """
        for view_element in self.elements:
            view_element.remove()
        self.elements.clear()

    def hide(self) -> None:
        """
        Hides all view elements
        """
        for view_element in self.elements:
            view_element.hide()

    def show(self) -> None:
        """
        Shows all view elements
        """
        for view_element in self.elements:
            view_element.show()

class CanvasEventManager:
    """
    Manages events on the canvas.
    """

    def __init__(self, canvas: FigureCanvasBase) -> None:
        self.canvas = canvas
        self.events_stack: list['Event'] = []

    def add(self, event: 'UniqueEvent | SharedEvent | GlobalEvent | EmptyEvent') -> None:
        """
        Adds an event to the event stack. Disables previous event group based
        on the new event type.

        Parameters
        ----------
        event : Union[UniqueEvent, SharedEvent, GlobalEvent, EmptyEvent]
            The event to add.
        """

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
            # globals don't disconnect anything, but they are inserted under the stack
            self.events_stack.insert(0, event)
            return

        self.events_stack.append(event)

    def _reconnect_events(self) -> None:
        # find next event group to activate
        shared_group = False
        for event in self.events_stack[::-1]:

            if isinstance(event, EmptyEvent):
                break

            # if no shared group activate just this event
            if isinstance(event, UniqueEvent):
                if not shared_group:
                    event.reconnect()
                    return
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
        """
        Disconnects latest unique event and reconnects next event group.
        """

        if not self.events_stack:
            return None

        # if not maching just call disconnect
        if not isinstance(self.events_stack[-1], UniqueEvent):
            return self.disconnect()

        self.events_stack.pop().disconnect()
        self._clear_empty_events()
        self._reconnect_events()

    def disconnect_shared(self) -> None:
        """
        Disconnects latest shared event group and reconnects next event group.
        """

        if not self.events_stack:
            return None

        if not isinstance(self.events_stack[-1], SharedEvent):
            return self.disconnect()

        while self.events_stack:
            if isinstance(self.events_stack[-1], SharedEvent):
                self.events_stack.pop().disconnect()
                continue
            break

        self._clear_empty_events()
        self._reconnect_events()

    def disconnect(self) -> None:
        """
        Disconnects all events.
        """
        for event in self.events_stack:
            event.disconnect()
        self.events_stack.clear()


class Event(ABC):
    """
    Base class for canvas events.
    """

    canvas: FigureCanvasBase

    @abstractmethod
    def __init__(self, ev_type: str, ev_callback: Callable) -> None:
        self.ev_type = ev_type
        self.ev_callback = ev_callback
        self.id = self.canvas.mpl_connect(ev_type, ev_callback)

    @classmethod
    def set_canvas(cls, canvas: FigureCanvasBase) -> None:
        """
        Sets the canvas for the event class.

        Parameters
        ----------
        canvas : FigureCanvasBase
            The canvas to set.
        """
        cls.canvas = canvas

    def reconnect(self) -> None:
        """
        Reconnects the event.
        """
         # source code says there is no error if self.id does not exist c:
        self.canvas.mpl_disconnect(self.id)
        self.id = self.canvas.mpl_connect(self.ev_type, self.ev_callback)

    def disconnect(self) -> None:
        """
        Disconnects the event.
        """
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
    UniqueEvent disconnects all other events
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
    """
    Base class for views.
    """

    def __init__(self, view_manager: ViewManager) -> None:
        self.vm = view_manager
        self.vem = ViewElementManager()
        self.cem = CanvasEventManager(self.vm.fig.canvas)
        self.change_button_y = 0.936
        self.change_button_length = 0.1
        self.change_button_height = 0.06
        self.home_ax = [0.05, self.change_button_y, self.change_button_length, self.change_button_height]
        self.clusters_ax = [0.15, self.change_button_y, self.change_button_length, self.change_button_height]
        self.hulls_ax = [0.25, self.change_button_y, self.change_button_length, self.change_button_height]
        self.labels_ax = [0.35, self.change_button_y, self.change_button_length, self.change_button_height]

    @abstractmethod
    def draw(self, *args, **kwargs) -> None:
        """
        Draws the view.

        Parameters
        ----------
        args : tuple
            Additional arguments.
        kwargs : dict
            Additional keyword arguments.

        """
        logging.info(f"{self.__class__} is drawing.")
        self.vem.show()

    @abstractmethod
    def hide(self) -> None:
        """
        Hides the view.
        """
        logging.info(f"{self.__class__} is hiding.")
        self.vem.hide()
        self.cem.disconnect()
        self.vm.fig.canvas.flush_events()

    def remove(self) -> None:
        """
        Fully deconstructs the view. Used to be called "undraw"
        """
        logging.info(f"{self.__class__} is removed.")
        self.cem.disconnect()
        self.vem.deconstruct()
        self.vm.fig.canvas.flush_events()

    def change_view(self, view_id: ViewsEnum, *args, **kwargs):
        """
        Changes the current view.

        Parameters
        ----------
        view_id : ViewsEnum
            The ID of the view to change to.
        args : tuple
            Additional arguments.
        kwargs : dict
            Additional keyword arguments.
        """
        import time
        s = time.time()
        self.hide()
        self.vm.get_view(view_id).draw(*args, **kwargs)
        e = time.time()
        print(f"{e-s}s")


class ViewElement(ABC, StateLinker):
    """
    Abstract base class representing a generic view element.
    """

    def __init__(self) -> None:
        """Initialize the ViewElement."""
        logging.info(f"{self.__class__} is creating.")

    @abstractmethod
    def remove(self) -> None:
        """Abstract method to remove the view element from the display."""
        logging.info(f"{self.__class__} is removing.")

    @abstractmethod
    def refresh(self) -> None:
        """Abstract method to refresh the view element."""
        logging.info(f"{self.__class__} is refreshing.")

    @abstractmethod
    def hide(self) -> None:
        """Hide the view element. Makes the element inactive and invisible"""
        logging.info(f"{self.__class__} is hiding")

    @abstractmethod
    def show(self) -> None:
        """Show the view element. Makes the element active and visible"""
        logging.info(f"{self.__class__} is showing")


# --------------------------- MIDDLE LEVEL CLASSES --------------------------- #


class ViewButton(ViewElement):
    """
    Class representing an abstract button view element.
    """

    def __init__(self, parent_view: View, axes: list[float],
                 label: str, callback: Callable) -> None:
        """Initialize the ViewButton.

        Parameters
        ----------
        parent_view: View
            The parent view to which the button belongs.
        axes: list[float]
            The position of the button on the view.
        label: str
            The label text of the button.
        callback: Callable
            The callback function to be executed when the button is clicked.
        """
        super().__init__()
        self.pv         = parent_view
        self.button_ax  = parent_view.vm.fig.add_axes(axes)
        self.button_ref = Button(self.button_ax, label)
        self.button_cid = self.button_ref.on_clicked(callback)
        
    @abstractmethod
    def remove(self):
        """Remove the button from the display."""
        super().remove()
        self.button_ref.disconnect_events()     
        self.pv.vm.fig.delaxes(self.button_ax)

    @abstractmethod
    def refresh(self) -> None:
        """Refresh the button."""
        return super().refresh()

    def hide(self) -> None:
        super().hide()
        self.button_ref.active = False
        self.button_ax.set_visible(False)

    def show(self) -> None:
        super().show()
        self.button_ref.active = True
        self.button_ax.set_visible(True)


class ViewTextBox(ViewElement):
    """
    Class representing an abstract text box view element.
    """

    def __init__(self, parent_view: View, axes: list[float], label: str, description="") -> None:
        """Initialize the ViewTextBox.

        Parameters
        ----------
        parent_view: View
            The parent view to which the text box belongs.
        axes: list[float]
            The position of the text box on the view.
        label: str
            The label text of the text box.
        """
        super().__init__()
        self.pv       = parent_view
        self.label_ax = parent_view.vm.fig.add_axes(axes)
        self.box_ref  = TextBox(self.label_ax, description, initial=label)

    @abstractmethod
    def remove(self):
        """Remove the text box from the display."""
        super().remove()
        self.pv.vm.fig.delaxes(self.label_ax)

    @abstractmethod
    def refresh(self) -> None:
        """Refresh the text box."""
        return super().refresh()

    def hide(self) -> None:
        """Hide the view element. Makes the element inactive and invisible"""
        super().hide()
        self.box_ref.active = False
        self.label_ax.set_visible(False)

    def show(self) -> None:
        super().show()
        self.box_ref.active = True
        self.label_ax.set_visible(True)

# ------------------------------ OUTPUT CLASSES ------------------------------ #

class ChangeViewButton(ViewButton):
    """Class representing a button for changing views."""
    
    def __init__(self, parent_view: View, axes: list[float], 
                 label: str, new_view: ViewsEnum) -> None:
        """Initialize the ChangeViewButton.

        Parameters
        ----------
        parent_view: View
            The parent view to which the button belongs.
        axes: list[float]
            The position of the button on the view.
        label: str
            The label text of the button.
        new_view: ViewsEnum
            The new view to switch to when the button is clicked.
        """
        super().__init__(parent_view, axes, label, 
                         lambda ev: parent_view.change_view(new_view, ev))
        self.button_ax.spines[['top', 'bottom', 'right', 'left']].set_visible(False)
        self.button_ref.color = "white"
        self.button_ref.hovercolor = "gainsboro"
        self.button_ax.set_facecolor("white")

    def remove(self):
        """Remove the button from the display."""
        return super().remove()

    def refresh(self) -> None:
        """Refresh the button."""
        return super().refresh()

    def highlight(self) -> None:
        self.button_ax.set_facecolor("whitesmoke")
        self.button_ref.color = "whitesmoke"
        self.button_ref.label.set_alpha(0.25)


class NormalButton(ViewButton):
    """
    A standard button for user interaction in a graphical view.
    """

    def __init__(self, parent_view: View, axes: list[float], 
                 label: str, callback: Callable) -> None:
        """init
        
        Parameters
        ----------
        parent_view : View
            The parent view to which the button belongs.
        axes : list[float]
            The position of the button in normalized coordinates [left, bottom, width, height].
        label : str
            The text displayed on the button.
        callback : Callable
            The function to be called when the button is clicked.
        
        """
        super().__init__(parent_view, axes, label, 
                         lambda ev: callback())

    def remove(self):
        """Remove the button from the view."""
        return super().remove()
    
    def refresh(self) -> None:
        """Refresh the appearance of the button."""
        return super().refresh()
    

class BlockingButton(ViewButton):
    """
    A button that blocks other interactions with itself until its action is completed.
    """

    def __init__(self, parent_view: View, axes: list[float], label: str, 
                 callback: Callable[[Callable[..., None]], None]) -> None:
        """init

        Parameters
        ----------
        parent_view : View
            The parent view to which the button belongs.
        axes : list[float]
            The position of the button in normalized coordinates [left, bottom, width, height].
        label : str
            The text displayed on the button.
        callback : Callable[[Callable[..., None]], None]
            The function to be called when the button is clicked, with an argument to reconnect the button.
        """

        def reconnect_callback(event = ...):
            self.button_cid = self.button_ref.on_clicked(blocking_callback)

        def blocking_callback(event = ...):
            self.button_ref.disconnect(self.button_cid)
            callback(reconnect_callback)

        super().__init__(parent_view, axes, label, blocking_callback)

    def remove(self):
        """Remove the button from the view."""
        return super().remove()
    
    def refresh(self) -> None:
        """Refresh the appearance of the button."""
        return super().refresh()
    

class ShiftingTextBox(ViewTextBox):
    """A text box that can be updated dynamically."""

    def __init__(self, parent_view: View, axes: list[float], 
                 update: Callable, submit: Callable, description: str = '', label: str = '') -> None:
        """init
        
        Parameters
        ----------
        parent_view : View
            The parent view to which the text box belongs.
        axes : list[float]
            The position of the text box in normalized coordinates [left, bottom, width, height].
        label : str
            The label of the text box.
        update : Callable
            The function to update the content of the text box.
        submit : Callable
            The function to be called when text is submitted.

        """
        super().__init__(parent_view, axes, label, description)

        old_kp = TextBox._keypress
        old_c = TextBox._click
        TextBox._keypress = self._custom_keypress
        TextBox._click = self._custom_click
        self.box_ref = TextBox(self.label_ax, description, initial=label)
        TextBox._keypress = old_kp
        TextBox._click = old_c

        self.update = update
        self.box_ref.on_submit(submit)

        # inject custom wrap to disp text
        self.box_ref.text_disp.set_wrap(True)
        gwt_type = type(self.box_ref.text_disp._get_wrapped_text)
        self.box_ref.text_disp._get_wrapped_text = gwt_type(
            lambda ref: self._custom_get_wrapped_text(ref, self.box_ref.ax), self.box_ref.text_disp
        )

    def remove(self) -> None:
        """Remove the text box from the view."""
        return super().remove()
    
    def refresh(self) -> None:
        """Refresh the content of the text box."""
        super().refresh()
        self.box_ref.set_val(self.update())

    @staticmethod
    def _custom_get_wrapped_text(self, ax):
        # lib
        if not self.get_wrap():
            return self.get_text()

        if self.get_usetex():
            return self.get_text()

        # custom
        line = self.get_text()
        line_width = ax.get_window_extent().width - 7 # TODO: make this responsive?
        current_width = self._get_rendered_text_width(line)

        while current_width > line_width:
            line = line[1:]
            current_width = self._get_rendered_text_width(line)

        return line
    
    @staticmethod
    def _custom_keypress(self, event):
        if self.ignore(event):
            return
        if self.capturekeystrokes:
            key = event.key
            text = self.text
            if len(key) == 1:
                text = (text[:self.cursor_index] + key +
                        text[self.cursor_index:])
                self.cursor_index += 1
            elif key == "end":
                self.cursor_index = len(text)
            elif key == "backspace":
                if self.cursor_index != 0:
                    text = (text[:self.cursor_index - 1] +
                            text[self.cursor_index:])
                    self.cursor_index -= 1
            self.text_disp.set_text(text)
            self._rendercursor()
            if self.eventson:
                self._observers.process('change', self.text)
                if key in ["enter", "return"]:
                    self._observers.process('submit', self.text)

    @staticmethod
    def _custom_click(self, event):
        if self.ignore(event):
            return
        if event.inaxes != self.ax:
            self.stop_typing()
            return
        if not self.eventson:
            return
        if event.canvas.mouse_grabber != self.ax:
            event.canvas.grab_mouse(self.ax)
        if not self.capturekeystrokes:
            self.begin_typing()
        self.cursor_index = len(self.text_disp.get_text())
        self._rendercursor()


class LimitedTextBox(ViewTextBox):
    """A text box that can be updated dynamically."""

    def __init__(self, parent_view: View, axes: list[float], 
                 update: Callable, submit: Callable, description = '', label: str = '') -> None:
        """init
        
        Parameters
        ----------
        parent_view : View
            The parent view to which the text box belongs.
        axes : list[float]
            The position of the text box in normalized coordinates [left, bottom, width, height].
        label : str
            The label of the text box.
        update : Callable
            The function to update the content of the text box.
        submit : Callable
            The function to be called when text is submitted.

        """
        super().__init__(parent_view, axes, label, description)
        self.update = update
        self.box_ref.on_submit(submit)
        
        # inject custom wrap to disp text
        self.box_ref.text_disp.set_wrap(True)
        gwt_type = type(self.box_ref.text_disp._get_wrapped_text)
        self.box_ref.text_disp._get_wrapped_text = gwt_type(
            lambda ref: self._custom_get_wrapped_text(ref, self.box_ref.ax), self.box_ref.text_disp
        )

    def remove(self) -> None:
        """Remove the text box from the view."""
        return super().remove()
    
    def refresh(self) -> None:
        """Refresh the content of the text box."""
        super().refresh()
        self.box_ref.set_val(self.update())

    @staticmethod
    def _custom_get_wrapped_text(self, ax):
        # lib
        if not self.get_wrap():
            return self.get_text()

        if self.get_usetex():
            return self.get_text()

        # custom
        line = self.get_text()
        line_width = ax.get_window_extent().width
        current_width = self._get_rendered_text_width(line)

        while current_width > line_width:
            line = line[:-1]
            current_width = self._get_rendered_text_width(line)

        self.set_text(line)

        return line


class ViewText(ViewElement):

    def __init__(self, ax: Axes, x: float, y: float, label: str) -> None:
        super().__init__()
        self.text_ref = Text(x, y, label)
        ax.add_artist(self.text_ref)

    def remove(self):
        super().remove()
        self.text_ref.remove()

    def refresh(self) -> None:
        return super().refresh()

    def hide(self) -> None:
        super().hide()
        self.text_ref.set_visible(False)

    def show(self) -> None:
        super().show()
        self.text_ref.set_visible(True)


class ViewRadioButtons(ViewElement):

    def __init__(self, parent_view: View, axes: list[float],
                 labels: list[str], callback: Callable, active: int = 0) -> None:
        super().__init__()
        self.pv = parent_view
        self.ax = parent_view.vm.fig.add_axes(axes, frameon=False)
        self.ref = RadioButtons(self.ax, labels=labels, active=active)
        self.ref.on_clicked(callback)

    def remove(self):
        super().remove()
        self.ref.disconnect_events()
        self.pv.vm.fig.delaxes(self.ax)

    def refresh(self) -> None:
        return super().refresh()

    def hide(self) -> None:
        super().hide()
        self.ref.active = False
        self.ax.set_visible(False)

    def show(self) -> None:
        super().show()
        self.ref.active = True
        self.ax.set_visible(True)


class ViewSlider(ViewElement):

    def __init__(self, parent_view: View, axes: list[float], label: str,
                 valmin: float, valmax: float, callback: Callable) -> None:
        super().__init__()
        self.pv = parent_view
        self.ax = parent_view.vm.fig.add_axes(axes, frameon=False)
        self.ref = Slider(ax=self.ax, label=label, valmin=valmin, valmax=valmax, valinit=2, initcolor=None)
        self.ref.on_changed(callback)

    def remove(self):
        super().remove()
        self.ref.disconnect_events()
        self.pv.vm.fig.delaxes(self.ax)

    def refresh(self) -> None:
        return super().refresh()

    def hide(self) -> None:
        super().hide()
        self.ref.active = False
        self.ax.set_visible(False)

    def show(self) -> None:
        super().show()
        self.ref.active = True
        self.ax.set_visible(True)


class CheckList(ViewElement):
    def __init__(self, vm, axes: list[float],
                 labels: list[str], callback: Callable) -> None:
        super().__init__()
        self.ax = vm.fig.add_axes(axes, frameon=False)
        self.ref = CheckButtons(self.ax, labels)
        self.callback = callback
        self.ref.on_clicked(self.callback)
        self.len = len(labels)
        self.check_all()
        self.shown = False

    def check_all(self):
        for i in range(self.len):
            self.ref.set_active(i)

    def update(self, labels: list[str]) -> None:
        label_to_actives = dict(zip(self.ref.labels, self.ref.get_status()))
        new_actives = [label_to_actives[label] if label in label_to_actives.keys() else True for label in labels]
        self.remove()
        super().__init__() # added for logs
        self.ref = CheckButtons(self.ax, labels, new_actives)
        self.ref.on_clicked(self.callback)
        plt.draw() #not sure if draw should be called automatically or not

    def remove(self):
        super().remove()
        self.ref.disconnect_events()
        self.ax.clear() # keeps axes for reuse

    def refresh(self) -> None:
        return super().refresh()

    def hide(self) -> None:
        super().hide()
        plt.subplots_adjust(bottom=0.15, left=0.01, right=0.99, top=0.935)
        self.ref.active = False
        self.ax.set_visible(False)
        self.shown = False
        plt.draw()

    def show(self) -> None:
        super().show()
        plt.subplots_adjust(bottom=0.15, left=0.25, right=0.99, top=0.935)
        self.ref.active = True
        self.ax.set_visible(True)
        self.shown = True
        plt.draw()

    def toggle(self):
        if self.shown:
            self.hide()
        else:
            self.show()

class CheckListManager(StateLinker):
    def __init__(self, vm: ViewManager):
        self.check_list = CheckList(
            vm, [0.01, 0.14, 0.1, 0.78], sorted(list(self.state.get_all_clusters().keys())), lambda x: print(x)
        )
        self.check_list.hide()

        self.button_ax = vm.fig.add_axes([0.832, 0.85, 0.15, 0.075], frameon=False)
        self.button = Button(self.button_ax, "Toggle list")
        self.button.on_clicked(lambda x: self.check_list.toggle())

        # self.test_button = Button(vm.fig.add_axes([0.6, 0.05, 0.1, 0.075], frameon=False), "Test")
        # self.test_button.on_clicked(lambda x: self.check_list.update(["x", "y"]))

    def hide_button(self):
        self.button.active = False
        self.button_ax.set_visible(False)
        self.check_list.hide()

    def show_button(self):
        self.button.active = True
        self.button_ax.set_visible(True)
        self.check_list.hide()
