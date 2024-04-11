import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from enum import Enum
from abc import ABC, abstractmethod
import logging

# ----------------------------- TOP LEVEL CLASSES ---------------------------- #

class ViewsEnum(Enum):
    HOME  = 0
    ADDP  = 1


class ViewElement(ABC):
    
    def __init__(self) -> None:
        logging.info(f"{self.__class__} is createing.")
    
    def remove(self) -> None:
        logging.info(f"{self.__class__} is removeing.")
        
        
class ViewElementManager:
    
    def __init__(self) -> None:
        self.elements: list[ViewElement] = []

    def add(self, el: ViewElement):
        self.elements.append(el)
        
    def deconstruct(self):
        for view_element in self.elements:
            view_element.remove()


class ViewButton(ViewElement):
    
    def __init__(self, parent_view, axes, label: str, callback: callable) -> None:
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


class ChangeViewButton(ViewButton):
    
    def __init__(self, parent_view, axes, label: str, new_view: ViewsEnum) -> None:
        super().__init__(parent_view, axes, label, lambda ev: parent_view.change_view(new_view, ev))
        
    def remove(self):
        super().remove()


class ViewManager:

    def __init__(self, fig, ax, data) -> None:
        self.fig   = fig
        self.ax    = ax
        self.data  = data
        self.views = []
    
    def register_views(self, views: list) -> None:
        self.views = views

    def get_view(self, view_id: ViewsEnum) -> "View":
        return self.views[view_id.value]
    
    def run(self) -> None:
        self.views[ViewsEnum.HOME.value].draw()


class View(ABC):
    
    def __init__(self, view_manager: ViewManager) -> None:
        self.vm = view_manager

    @abstractmethod
    def undraw(self) -> None:
        logging.info(f"{self.__class__} is undrawing.")

    @abstractmethod
    def draw(self) -> None:
        logging.info(f"{self.__class__} is drawing.")

    def change_view(self, view_id: ViewsEnum, event):
        self.undraw()
        self.vm.get_view(view_id).draw()


class Home(View):
    
    def __init__(self, view_manager: ViewManager) -> None:
        super().__init__(view_manager)

    def draw(self) -> None:
        super().draw()
        for culture_name in self.vm.data['data'].keys():
            self.vm.ax.scatter(self.vm.data['data'][culture_name]['x'], self.vm.data['data'][culture_name]['y'], color="blue")

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
        for culture_name in self.vm.data['data'].keys():
            self.vm.ax.scatter(self.vm.data['data'][culture_name]['x'], self.vm.data['data'][culture_name]['y'], color="black")

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


class Editor:
    def __init__(self, data) -> None:
        self.data = data
    
    def run(self) -> None:

        fig, ax = plt.subplots()
        fig.add_axes(ax)
        fig.subplots_adjust(bottom=0.2)

        vm = ViewManager(fig, ax, self.data)
        vm.register_views([Home(vm), AddPoints(vm)]) # must be the same as ViewsEnum
        vm.run()

        # dispalay
        plt.show()
