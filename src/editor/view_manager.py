import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from enum import Enum
from abc import ABC, abstractmethod
import logging

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

    def get_raw(self) -> dict:
        return self.data
    
    def set_raw(self) -> None:
        return self.data

# ----------------------------- TOP LEVEL CLASSES ---------------------------- #


class ViewsEnum(Enum):
    HOME  = 0
    ADDP  = 1


class ViewManager:

    def __init__(self, fig: Figure, ax: Axes) -> None:
        self.fig = fig
        self.ax  = ax
        self.views: list[View] = []
    
    def register_views(self, views: list) -> None:
        self.views = views

    def get_view(self, view_id: ViewsEnum) -> "View":
        return self.views[view_id.value]
    
    def run(self) -> None:
        self.views[ViewsEnum.HOME.value].draw()


class View(ABC):

    state: State = None
    
    def __init__(self, view_manager: ViewManager) -> None:
        self.vm = view_manager

    @abstractmethod
    def undraw(self) -> None:
        logging.info(f"{self.__class__} is undrawing.")

    @abstractmethod
    def draw(self) -> None:
        logging.info(f"{self.__class__} is drawing.")

    def change_view(self, view_id: ViewsEnum, *args):
        self.undraw()
        self.vm.get_view(view_id).draw()

    @classmethod
    def link_state(cls, lstate: State) -> None:
        cls.state = lstate


class ViewElement(ABC):

    state: State = None

    def __init__(self) -> None:
        logging.info(f"{self.__class__} is createing.")
    
    @abstractmethod
    def remove(self) -> None:
        logging.info(f"{self.__class__} is removeing.")

    @abstractmethod
    def refresh(self) -> None:
        logging.info(f"{self.__class__} is refreshing.")

    @classmethod
    def link_state(cls, lstate: State) -> None:
        cls.state = lstate
        
        
class ViewElementManager:
    
    def __init__(self) -> None:
        self.elements: list[ViewElement] = []

    def add(self, el: ViewElement) -> None:
        self.elements.append(el)

    def refresh(self) -> None:
        for view_element in self.elements:
            view_element.refresh()
        
    def deconstruct(self) -> None:
        for view_element in self.elements:
            view_element.remove()

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
        self.pv        = parent_view
        self.label_ax  = parent_view.vm.fig.add_axes(axes)
        self.label_ref = TextBox(self.label_ax, label)

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


class Home(View):
    
    def __init__(self, view_manager: ViewManager) -> None:
        super().__init__(view_manager)

    def draw(self) -> None:
        super().draw()
        for culture_name in self.state.get_raw()['data'].keys():
            self.vm.ax.scatter(self.state.get_raw()['data'][culture_name]['x'], self.state.get_raw()['data'][culture_name]['y'], color="blue")

        self.vem = ViewElementManager()
        
        self.vem.add(ChangeViewButton(self, [0.1, 0.05, 0.05, 0.05], "Home", ViewsEnum.HOME))
        self.vem.add(ChangeViewButton(self, [0.81, 0.05, 0.1, 0.075], "Add", ViewsEnum.ADDP))

        plt.draw()
    
    def undraw(self) -> None:
        super().undraw()
        
        self.vem.deconstruct()
        
        self.vm.ax.clear()
        self.vm.fig.canvas.flush_events()


class AddPoints(View):

    def __init__(self, view_manager: ViewManager) -> None:
        super().__init__(view_manager)

    def draw(self) -> None:
        super().draw()
        for culture_name in self.state.get_raw()['data'].keys():
            self.vm.ax.scatter(self.state.get_raw()['data'][culture_name]['x'], self.state.get_raw()['data'][culture_name]['y'], color="black")

        self.vem = ViewElementManager()
        
        self.vem.add(ChangeViewButton(self, [0.1, 0.05, 0.05, 0.05], "Home", ViewsEnum.HOME))

        self.bpe = self.vm.fig.canvas.mpl_connect('button_press_event', lambda ev : self.add_point_event(ev))

        plt.draw()
        
    def undraw(self) -> None:
        super().undraw()
        
        self.vem.deconstruct()
        
        self.vm.fig.canvas.mpl_disconnect(self.bpe)
               
        self.vm.ax.clear()
        self.vm.fig.canvas.flush_events()

    def add_point_event(self, event):
        logging.info(f"{self.__class__} EVENT: {event}")

# -------------------------------- MAIN EDITOR ------------------------------- #

class Editor:
    def __init__(self, state: State) -> None:
        self.state = state

        # make injections
        View.link_state(state)
        ViewElement.link_state(state)
    
    def run(self) -> None:

        fig, ax = plt.subplots()
        fig.add_axes(ax)
        fig.subplots_adjust(bottom=0.2)

        vm = ViewManager(fig, ax)
        vm.register_views([Home(vm), AddPoints(vm)]) # must be the same as ViewsEnum
        vm.run()

        # dispalay
        plt.show()
