from src.editor.view_manager import ViewManager
from .view_manager import *

import matplotlib.pyplot as plt
from matplotlib.backend_bases import FigureCanvasBase, PickEvent, MouseEvent, KeyEvent

class Home(View):
    
    def __init__(self, view_manager: ViewManager) -> None:
        super().__init__(view_manager)

    def draw(self) -> None:
        super().draw()
        
        self.vem.add(ChangeViewButton(self, [0.05, 0.05, 0.1, 0.075], "Home", ViewsEnum.HOME))
        self.vem.add(ChangeViewButton(self, [0.85, 0.05, 0.1, 0.075], "Labels", ViewsEnum.LABELS))

        plt.draw()
    
    def undraw(self) -> None:
        super().undraw()
        
        
class LabelsView(View):

    def __init__(self, view_manager: ViewManager) -> None:
        super().__init__(view_manager)
        self.dragged_item: LabelArtist = None
        self.picked_item: LabelArtist  = None
        self.events_stack = []
        
    def draw(self) -> None:
        super().draw()
        self.events_stack.clear()

        # buttons
        self.vem.add(ChangeViewButton(self, [0.05, 0.05, 0.1, 0.075], "Home", ViewsEnum.HOME))
        self.vem.add(NormalButton(self, [0.15, 0.05, 0.05, 0.075], "+", self.add_label))
        self.vem.add(NormalButton(self, [0.20, 0.05, 0.05, 0.075], "-", self.delete_label))

        # displays
        self.vem.add(UpdateableTextBox(self, [0.30, 0.05, 0.15, 0.075], "...", self.label_name_update, self.label_name_submit))
        self.vem.add(UpdateableTextBox(self, [0.45, 0.05, 0.15, 0.075], "...", self.label_x_shift_update, self.label_x_shift_submit))
        self.vem.add(UpdateableTextBox(self, [0.60, 0.05, 0.15, 0.075], "...", self.label_y_shift_update, self.label_y_shift_submit))

        self.cem.add(self.vm.fig.canvas.mpl_connect('pick_event', self.pick_event))
        self.cem.add(self.vm.fig.canvas.mpl_connect('button_release_event', self.release_event))
        self.cem.add(self.vm.fig.canvas.mpl_connect('key_press_event', self.key_press_event))

        plt.draw()
    
    def undraw(self) -> None:
        super().undraw()
        
    def pick_event(self, event: PickEvent) -> None:
        if isinstance(event.artist, LabelArtist):
            logging.info(f"{self.__class__} EVENT: {event} ARTIST: {event.artist} ID: {event.artist.id}")
            self.events_stack.append(event.artist.get_state())
            self.dragged_item = event.artist
            self.picked_item  = event.artist
            self.pick_pos = (event.mouseevent.xdata, event.mouseevent.ydata)

            # update fields
            self.vem.refresh()

    def release_event(self, event: MouseEvent) -> None:
        if self.dragged_item is not None:
            logging.info(f"{self.__class__} EVENT: {event} ID: {self.dragged_item.id}")
            old_pos = self.dragged_item.get_position()
            new_pos = (old_pos[0] + event.xdata - self.pick_pos[0],
                       old_pos[1] + event.ydata - self.pick_pos[1])
            self.dragged_item.set_position(new_pos)

            # update state info
            self.state.set_label_pos(self.dragged_item.id, new_pos[0], new_pos[1])

            self.dragged_item = None
            plt.draw()

    def add_label(self) -> None:
        pass

    def delete_label(self) -> None:
        self.picked_item.remove()
        self.picked_item = None
        self.vem.refresh()

    def ctrlz(self) -> None:
        if self.events_stack:
            id, *state = self.events_stack.pop()
            LabelArtist.get_by_id(self.vm.ax, id).set_state(state)
            plt.draw()

    def key_press_event(self, event: KeyEvent):
        logging.info(f"KEY PRESS: {event.key}")
        if event.key == "ctrl+z":
            return self.ctrlz()
        
    def label_name_update(self) -> str:
        if self.picked_item is None:
            return "..."
        return self.picked_item.get_text()
    
    def label_name_submit(self, nname) -> None:
        if self.picked_item is None:
            return
        self.picked_item.set_text(nname)
        plt.draw()

    def label_x_shift_update(self) -> float:
        if self.picked_item is None:
            return "..."
        return self.picked_item.

    def label_x_shift_submit(self, x) -> None:
        pass

    def label_y_shift_update(self) -> float:
        if self.picked_item is None:
            return "..."
        return self.picked_item.get_text()

    def label_y_shift_submit(self, y) -> None:
        pass


# -------------------------------- MAIN EDITOR ------------------------------- #

class Editor:
    def __init__(self, state: State) -> None:
        self.state = state

        # inject state
        StateLinker.link_state(state)
    
    def run(self) -> None:

        fig, ax = plt.subplots()
        fig.add_axes(ax)
        fig.subplots_adjust(bottom=0.2)

        vm = ViewManager(fig, ax)
        vm.register_views([Home(vm), LabelsView(vm)]) # must be the same as ViewsEnum
        vm.run()

        # dispalay
        plt.show()
