import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from copy import deepcopy

class Button_Manager:
    def __init__(self, fig, ax,  points) -> None:
        self.fig = fig
        self.menu = _Menu_Callback(fig, ax, points, self)
        self.points = points
        
        self.bhome = None
        self.brestart = None
        self.bnext = None
        self.bprev = None
        self.badd_points = None
        self.bdone = None

        self.axadd_points = None

        self.fig.canvas.mpl_connect('button_press_event', self.menu.on_click)

        self.buttons = [
            self.bhome,
            self.brestart,
            self.bnext,
            self.bprev,
            self.badd_points,
            self.bdone
        ]
        
        self.baxes = []
        self.menu.draw()

    def _show_home(self) -> None:
        self.axhome = self.fig.add_axes(self.menu.axhome_pos)
        self.baxes.append(self.axhome)

        self.bhome = Button(self.axhome, "H")
        self.bhome.on_clicked(self.menu.home_button)
    
    def _show_restart(self) -> None:
        self.axrestart = self.fig.add_axes(self.menu.axrestart_pos)
        self.baxes.append(self.axrestart)

        self.brestart = Button(self.axrestart, "R")
        self.brestart.on_clicked(self.menu.restart_button)
    
    def _show_next(self) -> None:
        self.axnext = self.fig.add_axes(self.menu.axnext_pos)
        self.baxes.append(self.axnext)

        self.bnext = Button(self.axnext, "Next")
        self.bnext.on_clicked(self.menu.next_button)
    
    def _show_prev(self) -> None:
        self.axprev = self.fig.add_axes(self.menu.axprev_pos)
        self.baxes.append(self.axprev)

        self.bprev = Button(self.axprev, "Prev")
        self.bprev.on_clicked(self.menu.prev_button)

    def _show_add_points(self) -> None:
        self.axadd_points = self.fig.add_axes(self.menu.axadd_points_pos)
        self.baxes.append(self.axadd_points)

        self.badd_points = Button(self.axadd_points, "Add Points")
        self.badd_points.on_clicked(self.menu.add_points_button)

    def _show_done(self) -> None:
        self.axdone = self.fig.add_axes(self.menu.axdone_pos)
        self.baxes.append(self.axdone)

        self.bdone = Button(self.axdone, "Done")
        self.bdone.on_clicked(self.menu.done_button)

    def _delete_buttons(self) -> None:
        for bax in self.baxes:
            self.fig.delaxes(bax)
            
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.baxes = []
                        

    def show_buttons_menu(self) -> None:
        self._delete_buttons()

        self._show_home()
        self._show_restart()
        self._show_next()
        self._show_prev()
        self._show_add_points()
        plt.draw()
    
    def show_adding_points(self) -> None:
        self._delete_buttons()

        self._show_home()
        self._show_restart()        
        self._show_done()
        plt.draw()
    

class _Menu_Callback:
    def __init__(self, fig, ax, points, manager: Button_Manager) -> None:
        self.fig = fig
        self.ax = ax
        self.points = points
        self.manager = manager
        self.adding_points = False
        # init functions
        self._configurate_buttons_position()
    
    """
        Private method for setting position for buttons
    """
    def _configurate_buttons_position(self) -> None:
        # left, bottom, width, height
        self.axhome_pos = [0.1, 0.05, 0.05, 0.05]
        self.axrestart_pos = [0.165, 0.05, 0.05, 0.05]
        self.axnext_pos = [0.81, 0.05, 0.1, 0.075]
        self.axprev_pos = [0.7, 0.05, 0.1, 0.075]
        self.axadd_points_pos =[0.55, 0.05, 0.14, 0.075]
        self.axdone_pos = [0.81, 0.05, 0.1, 0.075]
    
    def home_button(self, event) -> None:
        ...
    
    def restart_button(self, event) -> None:
        ...
    
    def next_button(self, event) -> None:
        ...

    def prev_button(self, event) -> None:
        ...
    
    def select_set_button(self, event) -> None:
        ...
    
    def add_points_button(self, event) -> None:
        self.adding_points = True
        self.manager.show_adding_points()

    def done_button(self, event) -> None:
        self.adding_points = False
        self.manager.show_buttons_menu()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        new_point = (event.xdata, event.ydata)
        if self.adding_points:
            self.points = np.vstack([self.points, new_point])

        self.draw()

    def draw(self) -> None:
        self.ax.clear()
        self.ax.scatter(self.points[:, 0], self.points[:, 1])
        plt.draw()


class Plot:
    def __init__(self, path = str) -> None:
        # ./points/kamada_l1-mutual_attraction_2d.csv
        self.path = path
    
    def draw(self) -> None:

        fig, ax = plt.subplots()
        fig.add_axes(ax)
        fig.subplots_adjust(bottom=0.2)

        rnd_points = np.random.uniform(0, 10, (10, 2))
        

        button_manager = Button_Manager(fig, ax, rnd_points)
        
        button_manager.show_buttons_menu()

        plt.show()

if __name__ == "__main__":

    plot = Plot("kk_swap_2d.csv")
    plot.draw()

    
