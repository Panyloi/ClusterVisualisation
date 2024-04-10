import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from enum import Enum
from abc import ABC, abstractmethod
import logging
import sys

class ViewsEnum(Enum):
    HOME = 0
    ADDP  = 1


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

    name = "AbstractView"
    
    def __init__(self, view_manager: ViewManager) -> None:
        self.vm = view_manager

    @abstractmethod
    def undraw(self) -> None:
        logging.info(f"{self.name} is undrawing.")

    @abstractmethod
    def draw(self) -> None:
        logging.info(f"{self.name} is drawing.")

    def change_view(self, view_id: ViewsEnum, event):
        self.undraw()
        self.vm.get_view(view_id).draw()


class Home(View):

    name = "HomeView"
    
    def __init__(self, view_manager: ViewManager) -> None:
        super().__init__(view_manager)

    def draw(self) -> None:
        super().draw()
        self.vm.ax.scatter(self.vm.data[:, 0], self.vm.data[:, 1], color="blue")

        self.axhome = self.vm.fig.add_axes([0.1, 0.05, 0.05, 0.05])
        self.bhome = Button(self.axhome, "H")
        self.bhome.on_clicked(lambda ev : self.change_view(ViewsEnum.HOME, ev))

        self.axtmp = self.vm.fig.add_axes([0.81, 0.05, 0.1, 0.075])
        self.btmp = Button(self.axtmp, "Add")
        self.btmp.on_clicked(lambda ev : self.change_view(ViewsEnum.ADDP, ev))

        plt.draw()
    
    def undraw(self) -> None:
        super().undraw()
        self.vm.fig.delaxes(self.axhome)
        self.vm.fig.delaxes(self.axtmp)
        self.vm.ax.clear()
        self.vm.fig.canvas.flush_events()


class AddPoints(View):

    name = "AddPointsView"

    def __init__(self, view_manager: ViewManager) -> None:
        super().__init__(view_manager)

    def draw(self) -> None:
        super().draw()
        self.vm.ax.scatter(self.vm.data[:, 0], self.vm.data[:, 1], color="black")

        self.axhome = self.vm.fig.add_axes([0.1, 0.05, 0.05, 0.05])
        self.bhome = Button(self.axhome, "H")
        self.bhome.on_clicked(lambda ev : self.change_view(ViewsEnum.HOME, ev))

        self.bpe = self.vm.fig.canvas.mpl_connect('button_press_event', lambda ev : self.add_point_event(ev))

        plt.draw()
        
    def undraw(self) -> None:
        super().undraw()
        self.vm.fig.delaxes(self.axhome)
        self.vm.ax.clear()
        self.vm.fig.canvas.mpl_disconnect(self.bpe)
        self.vm.fig.canvas.flush_events()

    #CUSTOM

    def add_point_event(self, event):
        logging.info(f"{self.name} EVENT: {event}")

class Plot:
    def __init__(self, path: str) -> None:
        # ./points/kamada_l1-mutual_attraction_2d.csv
        self.path = path
    
    def draw(self) -> None:

        fig, ax = plt.subplots()
        fig.add_axes(ax)
        fig.subplots_adjust(bottom=0.2)

        rnd_points = np.random.uniform(0, 10, (10, 2))
        vm = ViewManager(fig, ax, rnd_points)
        vm.register_views([Home(vm), AddPoints(vm)])

        vm.run()
        plt.show()

if __name__ == "__main__":
    FORMAT = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
    plot = Plot("kk_swap_2d.csv")
    plot.draw()

    
