from .view_manager import *

import matplotlib.pyplot as plt
from matplotlib.backend_bases import PickEvent, MouseEvent, KeyEvent

from sklearn.cluster import AgglomerativeClustering
import numpy as np


class Home(View):
    
    def __init__(self, view_manager: ViewManager) -> None:
        super().__init__(view_manager)

    def draw(self, *args, **kwargs) -> None:
        super().draw()
        
        self.vem.add(ChangeViewButton(self, [0.05, 0.05, 0.1, 0.075], "Home", ViewsEnum.HOME))
        self.vem.add(ChangeViewButton(self, [0.15, 0.05, 0.1, 0.075], "Labels", ViewsEnum.LABELS))
        self.vem.add(ChangeViewButton(self, [0.25, 0.05, 0.1, 0.075], "Cluster", ViewsEnum.CLUSTER))

        plt.draw()
    
    def undraw(self) -> None:
        super().undraw()
        
        
class LabelsView(View):

    def __init__(self, view_manager: ViewManager) -> None:
        super().__init__(view_manager)
        self.dragged_item: LabelArtist | None = None
        self.picked_item:  LabelArtist | None = None
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

        self.cem.add(SharedEvent('pick_event', self.pick_event))
        self.cem.add(SharedEvent('button_release_event', self.release_event))
        self.cem.add(SharedEvent('key_press_event', self.key_press_event))

        self.vem.refresh()
        plt.draw()
    
    def undraw(self) -> None:
        super().undraw()
        
    def pick_event(self, event: PickEvent) -> None:
        logging.info(f"{self.__class__} EVENT: {event} ARTIST: {event.artist} ID: {getattr(event.artist, 'id', None)}")
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
            new_pos = (old_pos[0] + subtract_with_default(event.xdata, self.pick_pos[0], 0),
                       old_pos[1] + subtract_with_default(event.ydata, self.pick_pos[1], 0))
            self.dragged_item.set_position(new_pos)
            self.dragged_item = None
            plt.draw()

    def add_label(self) -> None:
        nid = self.state.add_empty_label()
        LabelArtist.text(self.vm.ax, nid)
        plt.draw()
        
    def add_arrow(self) -> None:
        if self.picked_item is None:
            return
        
        # create empty arrow in state and get it's id
        nid = self.state.add_empty_arrow(self.picked_item.id)
        
        # create artist and place arrow on plot
        x, y = self.state.get_label_pos(self.picked_item.id)
        atx, aty = self.state.get_arrow_att_point(self.picked_item.id, nid)
        rfx, rfy = self.state.get_arrow_ref_point(self.picked_item.id, nid)
        val = self.state.get_arrow_val(self.picked_item.id, nid)
        self.picked_item.arrows[nid] = ArrowArtist.arrow(self.vm.ax, nid, x, y, rfx, rfy, atx-x, aty-y, self.picked_item, val)
        
        plt.draw()

    def delete_label(self) -> None:
        if self.picked_item is None:
            return
        
        self.picked_item.remove()
        self.picked_item = None
        self.vem.refresh()

    def ctrlz(self) -> None:
        if self.events_stack:
            id, *state = self.events_stack.pop()
            l = LabelArtist.get_by_id(self.vm.ax, id)
            if l is not None:
                l.set_state(state)
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
        self.picked_item: ArrowArtist | None = None

    def draw(self, *args, **kwargs) -> None:
        super().draw()

        # get picked arrow if exists
        self.picked_item = kwargs.get('picked_item', None)

        # buttons
        self.vem.add(ChangeViewButton(self, [0.05, 0.05, 0.1, 0.075], "Home", ViewsEnum.HOME))
        self.vem.add(NormalButton(self, [0.15, 0.05, 0.05, 0.075], "-", self.delete_arrow))
        self.vem.add(BlockingButton(self, [0.50, 0.05, 0.05, 0.075], "p", self.sh_point_picker))
        self.vem.add(BlockingButton(self, [0.80, 0.05, 0.05, 0.075], "p", self.rf_point_picker))

        # displays
        self.vem.add(UpdateableTextBox(self, [0.30, 0.05, 0.10, 0.075], "...", self.arrow_shx_update, self.arrow_shx_submit))
        self.vem.add(UpdateableTextBox(self, [0.40, 0.05, 0.10, 0.075], "...", self.arrow_shy_update, self.arrow_shy_submit))
        self.vem.add(UpdateableTextBox(self, [0.60, 0.05, 0.10, 0.075], "...", self.arrow_rfx_update, self.arrow_rfx_submit))
        self.vem.add(UpdateableTextBox(self, [0.70, 0.05, 0.10, 0.075], "...", self.arrow_rfy_update, self.arrow_rfy_submit))

        self.cem.add(SharedEvent('pick_event', self.pick_event))
        # self.cem.add(self.vm.fig.canvas.mpl_connect('key_press_event', self.key_press_event))

        self.vem.refresh()
        plt.draw()
    
    def undraw(self) -> None:
        return super().undraw()
    
    def arrow_shx_update(self) -> float:
        if self.picked_item is None:
            return 0
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
            return 0
        return self.picked_item.get_shs()[1]

    def arrow_shy_submit(self, nshy) -> None:
        if self.picked_item is None:
            return
        try:
            self.picked_item.set(shy=float(nshy))
            plt.draw()
        except ValueError:
            pass

    def arrow_rfx_update(self) -> float:
        if self.picked_item is None:
            return 0
        return self.picked_item.get_rfs()[0]

    def arrow_rfx_submit(self, nrfx) -> None:
        if self.picked_item is None:
            return
        try:
            self.picked_item.set(rfx=float(nrfx))
            plt.draw()
        except ValueError:
            pass
    
    def arrow_rfy_update(self) -> float:
        if self.picked_item is None:
            return 0
        return self.picked_item.get_rfs()[1]

    def arrow_rfy_submit(self, nrfy) -> None:
        if self.picked_item is None:
            return
        try:
            self.picked_item.set(rfy=float(nrfy))
            plt.draw()
        except ValueError:
            pass

    def pick_event(self, event, *args, **kwargs) -> None:
        logging.info(f"{self.__class__} EVENT: {event} ARTIST: {event.artist} ID: {event.artist.id}")
        if isinstance(event.artist, ArrowArtist):
            self.picked_item  = event.artist
            self.vem.refresh()

        if isinstance(event.artist, LabelArtist):
            return self.change_view(ViewsEnum.LABELS, picked_item=event.artist)

    def delete_arrow(self, *args, **kwargs) -> None:
        if self.picked_item is None:
            return
        
        self.picked_item.remove()
        self.picked_item = None
        self.vem.refresh()

    def sh_point_picker(self, reconect_callback: Callable[..., None], *args, **kwargs) -> None:
        self.cem.add(UniqueEvent('button_press_event', lambda event, *args, **kwargs : self.sh_point_pick_event(reconect_callback, event, *args, **kwargs)))

    def sh_point_pick_event(self, reconect_callback: Callable[..., None], event: MouseEvent, *args, **kwargs) -> None:
        if event.inaxes is not self.vm.ax:
            logging.info(f"Point pick event out of chart.")
            return

        self.picked_item.set_sh_by_raw(event.xdata, event.ydata)
        
        self.cem.disconnect_unique()
        reconect_callback()
        self.vem.refresh()

    def rf_point_picker(self, reconect_callback: Callable[..., None], *args, **kwargs) -> None:
        self.cem.add(UniqueEvent('button_press_event', lambda event, *args, **kwargs : self.rf_point_pick_event(reconect_callback, event, *args, **kwargs)))

    def rf_point_pick_event(self, reconect_callback: Callable[..., None], event: MouseEvent, *args, **kwargs) -> None:
        if event.inaxes is not self.vm.ax:
            logging.info(f"Point pick event out of chart.")
            return

        self.picked_item.set(rfx=event.xdata, rfy=event.ydata)
        
        self.cem.disconnect_unique()
        reconect_callback()
        self.vem.refresh()


    
class ClusterView(View):

    def __init__(self, view_manager: ViewManager) -> None:
        super().__init__(view_manager)
        self.type = None
        self.linkage = None
        self.scalar = None
        self.clusters = {}
        self.cmap = plt.cm.get_cmap("hsv", len(self.state.get_raw()['data'].keys()))
        self.removed = {"x": [], "y": []}

    def draw(self, *args, **kwargs) -> None:
        super().draw()
        for culture_name in self.state.get_raw()['data'].keys():
            self.vm.ax.scatter(
                self.state.get_raw()['data'][culture_name]['x'],
                self.state.get_raw()['data'][culture_name]['y'],
                color="blue", s=3
            )

        self.widgetType    = ViewRadioButtons(self, [0.05, 0.15, 0.3, 0.75], sorted(list(self.state.get_raw()['data'].keys())), self._update_args)
        self.widgetLinkage = ViewRadioButtons(self, [0.4, 0.15, 0.1, 0.1], ["ward", "complete", "average", "single"], self._update_args)
        self.widgetScalar  = ViewSlider(self, [0.55, 0.17, 0.3, 0.05], "", 0.01, 2.5, self._update_args)

        self.type    = self.widgetType.ref.value_selected
        self.linkage = self.widgetLinkage.ref.value_selected
        self.scalar  = self.widgetScalar.ref.val

        self.vem.add(self.widgetType)
        self.vem.add(self.widgetLinkage)
        self.vem.add(self.widgetScalar)

        self.toggle()
        self.toggle()

        self.vem.add(NormalButton(self, [0.1, 0.05, 0.17, 0.075], "Toggle options", self.toggle))
        self.vem.add(NormalButton(self, [0.28, 0.05, 0.15, 0.075], "Save cluster", self.save_cluster))
        self.vem.add(NormalButton(self, [0.44, 0.05, 0.17, 0.075], "Remove points", self.remove_points))
        self.vem.add(NormalButton(self, [0.62, 0.05, 0.09, 0.075], "Reset", self.reset))
        self.vem.add(ChangeViewButton(self, [0.85, 0.05, 0.09, 0.075], "Home", ViewsEnum.HOME))
        self.vem.add(ChangeViewButton(self, [0.75, 0.05, 0.09, 0.075], "Save", ViewsEnum.CLUSTER))

        self.draw_cluster()

    def undraw(self) -> None:
        super().undraw()
        plt.subplots_adjust(bottom=0.2)
        plt.subplots_adjust(left=0.125)

    def _update_args(self, *args, **kwargs):
            self.type = self.widgetType.ref.value_selected
            self.linkage = self.widgetLinkage.ref.value_selected
            self.scalar = self.widgetScalar.ref.val
            self.draw_cluster()

    def toggle(self, *args, **kwargs) -> None:
        if self.widgetType.ax.get_visible():
            self.widgetType.ref._buttons.set_visible(False)
            self.widgetLinkage.ref._buttons.set_visible(False)
            self.widgetType.ax.set_visible(False)
            self.widgetLinkage.ax.set_visible(False)
            self.widgetScalar.ax.set_visible(False)
            plt.subplots_adjust(bottom=0.2)
            plt.subplots_adjust(left=0.125)
        else:
            self.widgetType.ref._buttons.set_visible(True)
            self.widgetLinkage.ref._buttons.set_visible(True)
            self.widgetType.ax.set_visible(True)
            self.widgetLinkage.ax.set_visible(True)
            self.widgetScalar.ax.set_visible(True)
            plt.subplots_adjust(bottom=0.3)
            plt.subplots_adjust(left=0.4)

        plt.draw()

    def draw_cluster(self):
        if self.type is None or self.linkage is None or self.scalar is None: return
        cluster = self.state.get_raw()['data'][self.type]
        sth = np.column_stack([cluster["x"], cluster["y"]])
        if len(cluster["x"]) > 1:
            dist = AgglomerativeClustering(n_clusters=1, compute_distances=True).fit(sth).distances_
            mean = np.mean(dist)
            clustering = AgglomerativeClustering(n_clusters=None, linkage=self.linkage, distance_threshold=mean * self.scalar).fit(sth)
            color = clustering.labels_
        else:
            color = 'k'

        self.vm.ax.clear()
        for culture_name in self.state.get_raw()['data'].keys():
            self.vm.ax.scatter(
                self.state.get_raw()['data'][culture_name]['x'],
                self.state.get_raw()['data'][culture_name]['y'],
                color="blue", s=3
            )
        self.show_clusters()
        self.vm.ax.scatter(cluster["x"], cluster["y"], c=color)
        plt.draw()

    def save_cluster(self, *args, **kwargs):
        self.clusters[self.type] = (self.linkage, self.scalar, self.cmap(list(self.state.get_raw()['data'].keys()).index(self.type)))
        print(self.clusters)

    def show_clusters(self, *args, **kwargs):
        for key, value in self.clusters.items():
            self.vm.ax.scatter(
                self.state.get_raw()['data'][key]['x'],
                self.state.get_raw()['data'][key]['y'],
                color=value[2], s=10
            )
        for i in range(len(self.removed["x"])):
            self.vm.ax.annotate("x", (self.removed["x"][i]-3, self.removed["y"][i]-3))

    def reset(self, *args, **kwargs):
        self.clusters = {}
        self.removed = {"x": [], "y": []}
        self.draw_cluster()

    def remove_points(self, *args, **kwargs):
        if self.type in self.clusters.keys():
            cluster = self.state.get_raw()['data'][self.type]
            sth = np.column_stack([cluster["x"], cluster["y"]])
            dist = AgglomerativeClustering(n_clusters=1, compute_distances=True).fit(sth).distances_
            mean = np.mean(dist)
            clustering = AgglomerativeClustering(n_clusters=None, linkage=self.linkage,
                                                 distance_threshold=mean * self.scalar).fit(sth)
            for i in range(len(clustering.labels_)):
                if clustering.labels_[i] != 0:
                    self.removed["x"].append(cluster["x"][i])
                    self.removed["y"].append(cluster["y"][i])
            print(self.removed)

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

        # init event canvas
        Event.set_canvas(fig.canvas)

        vm = ViewManager(fig, ax)
        vm.register_views([Home(vm), LabelsView(vm), ArrowsView(vm), ClusterView(vm)]) # must be the same as ViewsEnum
        vm.run()

        # dispalay
        plt.show()
