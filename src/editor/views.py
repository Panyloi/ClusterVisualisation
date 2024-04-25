from src.editor.view_manager import ViewManager
from .view_manager import *

import matplotlib.pyplot as plt
import matplotlib.artist as artist
from matplotlib.backend_bases import FigureCanvasBase, PickEvent, MouseEvent, KeyEvent

class Home(View):
    
    def __init__(self, view_manager: ViewManager) -> None:
        super().__init__(view_manager)

    def draw(self, *args, **kwargs) -> None:
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
        
    def draw(self, *args, **kwargs) -> None:
        super().draw()
        self.events_stack.clear()

        # get picked arrow if exists
        self.picked_item = kwargs.get('picked_item', None)

        # buttons
        self.vem.add(ChangeViewButton(self, [0.05, 0.05, 0.1, 0.075], "Home", ViewsEnum.HOME))
        self.vem.add(NormalButton(self, [0.15, 0.05, 0.05, 0.075], "+", self.add_label))
        self.vem.add(NormalButton(self, [0.20, 0.05, 0.05, 0.075], "-", self.delete_label))
        self.vem.add(NormalButton(self, [0.50, 0.05, 0.10, 0.075], "+arrow", self.add_arrow))

        # displays
        self.vem.add(UpdateableTextBox(self, [0.30, 0.05, 0.15, 0.075], "...", self.label_name_update, self.label_name_submit))

        self.cem.add(self.vm.fig.canvas.mpl_connect('pick_event', self.pick_event))
        self.cem.add(self.vm.fig.canvas.mpl_connect('button_release_event', self.release_event))
        self.cem.add(self.vm.fig.canvas.mpl_connect('key_press_event', self.key_press_event))

        self.vem.refresh()
        plt.draw()
    
    def undraw(self) -> None:
        super().undraw()
        
    def pick_event(self, event: PickEvent) -> None:
        logging.info(f"{self.__class__} EVENT: {event} ARTIST: {event.artist} ID: {event.artist.id}")
        if isinstance(event.artist, LabelArtist):
            self.events_stack.append(event.artist.get_state())
            self.dragged_item = event.artist
            self.picked_item  = event.artist
            self.pick_pos = (event.mouseevent.xdata, event.mouseevent.ydata)

            # update fields
            self.vem.refresh()
            
        if isinstance(event.artist, ArrowArtist):
            # check if click overlaps with other artists
            artists = get_artists_by_type(self.vm.ax, LabelArtist)
            for artist in artists:
                if artist.contains(event.mouseevent)[0]:
                    logging.info(f"{self.__class__} EVENT: {event} canceled due to overlapping Label: {artist}")
                    return
            
            # pass current artist as kwarg for new view to start initiated
            return self.change_view(ViewsEnum.ARROWS, picked_item=event.artist)

    def release_event(self, event: MouseEvent) -> None:
        if self.dragged_item is not None:
            logging.info(f"{self.__class__} EVENT: {event} ID: {self.dragged_item.id}")
            old_pos = self.dragged_item.get_position()
            new_pos = (old_pos[0] + event.xdata - self.pick_pos[0],
                       old_pos[1] + event.ydata - self.pick_pos[1])
            self.dragged_item.set_position(new_pos)
            self.dragged_item = None
            plt.draw()

    def add_label(self) -> None:
        nid = self.state.add_empty_label()
        LabelArtist.text(self.vm.ax, nid)
        plt.draw()
        
    def add_arrow(self) -> None:
        nid = self.state.add_empty_arrow(self.picked_item.id)
        
        x, y = self.state.get_label_pos(self.picked_item.id)
        atx, aty = self.state.get_arrow_att_point(self.picked_item.id, nid)
        rfx, rfy = self.state.get_arrow_ref_point(self.picked_item.id, nid)
        val = self.state.get_arrow_val(self.picked_item.id, nid)
        self.picked_item.arrows[nid] = ArrowArtist.arrow(self.vm.ax, nid, x, y, rfx, rfy, atx-x, aty-y, self.picked_item, val)
        
        plt.draw()

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


class ArrowsView(View):

    def __init__(self, view_manager: ViewManager) -> None:
        super().__init__(view_manager)
        self.picked_item: ArrowArtist  = None

    def draw(self, *args, **kwargs) -> None:
        super().draw()

        # get picked arrow if exists
        self.picked_item = kwargs.get('picked_item', None)

        # buttons
        self.vem.add(ChangeViewButton(self, [0.05, 0.05, 0.1, 0.075], "Home", ViewsEnum.HOME))
        self.vem.add(NormalButton(self, [0.15, 0.05, 0.05, 0.075], "-", self.delete_arrow))
        self.vem.add(NormalButton(self, [0.60, 0.05, 0.05, 0.075], "p", self.point_picker))

        # displays
        self.vem.add(UpdateableTextBox(self, [0.30, 0.05, 0.15, 0.075], "...", self.arrow_shx_update, self.arrow_shx_submit))
        self.vem.add(UpdateableTextBox(self, [0.45, 0.05, 0.15, 0.075], "...", self.arrow_shy_update, self.arrow_shy_submit))

        self.cem.add(self.vm.fig.canvas.mpl_connect('pick_event', self.pick_event))
        # self.cem.add(self.vm.fig.canvas.mpl_connect('key_press_event', self.key_press_event))

        self.vem.refresh()
        plt.draw()
    
    def undraw(self) -> None:
        return super().undraw()
    
    def arrow_shx_update(self) -> float:
        if self.picked_item is None:
            return "..."
        return self.picked_item.get_shs()[0]

    def arrow_shx_submit(self, nshx) -> None:
        if self.picked_item is None:
            return
        try:
            self.picked_item.set(shx=float(nshx))
            plt.draw()
        except ValueError:
            pass
    
    def arrow_shy_update(self) -> float:
        if self.picked_item is None:
            return "..."
        return self.picked_item.get_shs()[1]

    def arrow_shy_submit(self, nshy) -> None:
        if self.picked_item is None:
            return
        try:
            self.picked_item.set(shy=float(nshy))
            plt.draw()
        except ValueError:
            pass

    def pick_event(self, event) -> None:
        logging.info(f"{self.__class__} EVENT: {event} ARTIST: {event.artist} ID: {event.artist.id}")
        if isinstance(event.artist, ArrowArtist):
            self.picked_item  = event.artist
            self.vem.refresh()

        if isinstance(event.artist, LabelArtist):
            return self.change_view(ViewsEnum.LABELS, picked_item=event.artist)

    def delete_arrow(self) -> None:
        self.picked_item.remove()
        self.picked_item = None
        self.vem.refresh()

    def point_picker(self) -> None:
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
        vm.register_views([Home(vm), LabelsView(vm), ArrowsView(vm)]) # must be the same as ViewsEnum
        vm.run()

        # dispalay
        plt.show()
