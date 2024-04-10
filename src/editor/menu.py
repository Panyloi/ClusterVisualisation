import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

class Button_Manager:
    def __init__(self, fig) -> None:
        self.fig = fig
        self.menu = _Menu_Callback(fig)
        self.bhome = None
        self.brestart = None
        self.bnext = None
        self.bprev = None

    def _show_home(self) -> None:
        self.axhome = self.fig.add_axes(self.menu.axhome_pos)
        self.bhome = Button(self.axhome, "H")
        self.bhome.on_clicked(self.menu.home_button)
    
    def _show_restart(self) -> None:
        self.axrestart = self.fig.add_axes(self.menu.axrestart_pos)
        self.brestart = Button(self.axrestart, "R")
        self.brestart.on_clicked(self.menu.restart_button)
    
    def _show_next(self) -> None:
        self.axnext = self.fig.add_axes(self.menu.axnext_pos)
        self.bnext = Button(self.axnext, "Next")
        self.bnext.on_clicked(self.menu.next_button)
    
    def _show_prev(self) -> None:
        self.axprev = self.fig.add_axes(self.menu.axprev_pos)
        self.bprev = Button(self.axprev, "Prev")
        self.bprev.on_clicked(self.menu.prev_button)

    def show_buttons(self) -> None:
        self._show_home()
        self._show_restart()
        self._show_next()
        self._show_prev()
        plt.draw()
        

class _Menu_Callback:
    def __init__(self, fig) -> None:
        self.fig = fig

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
    


class Plot:
    def __init__(self, path = str) -> None:
        # ./points/kamada_l1-mutual_attraction_2d.csv
        self.path = path
    
    def draw(self) -> None:
        # df = pd.read_csv(self.path, sep=';')
        # df["instance_id"] = df["instance_id"].apply(lambda x: x.split(sep="_")[0])

        # x_points = df["x"].to_numpy()
        # y_points = df["y"].to_numpy()


        # categorized_names = df["instance_id"].unique()
        # gruped_points = {}

        # for i in range(df["instance_id"].nunique()):
        #     selected_points = df.loc[df["instance_id"] == categorized_names[i]]
        #     gruped_points[categorized_names[i]] = {"x": selected_points["x"].to_numpy(),
        #                                         "y": selected_points["y"].to_numpy()}

        # fig, ax = plt.subplots()
        # ax.scatter(x_points, y_points, s=10)
        # ax.set_title('Simple plot')

        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.2)

        rnd_points = np.random.uniform(0, 10, (10, 2))
        

        button_manager = Button_Manager(fig)
        button_manager.show_buttons()

        plt.show()

if __name__ == "__main__":

    plot = Plot("kk_swap_2d.csv")
    plot.draw()

    
