import matplotlib.pyplot as plt
from matplotlib.backend_bases import PickEvent, MouseEvent, KeyEvent, ResizeEvent
from sklearn.cluster import AgglomerativeClustering, DBSCAN

from .view_manager import *
from ..generator.hull_generator import calc_hull
from ..generator.labels_generator import calc, parse_solution_to_editor
from scipy import interpolate


class Home(View):

    def __init__(self, view_manager: ViewManager) -> None:
        super().__init__(view_manager)
        view_button = ChangeViewButton(self, self.home_ax, "Home", ViewsEnum.HOME)
        view_button.highlight()
        self.vem.add(view_button)
        self.vem.add(ChangeViewButton(self, self.labels_ax, "Labels", ViewsEnum.LABELS))
        self.vem.add(ChangeViewButton(self, self.clusters_ax, "Cluster", ViewsEnum.CLUSTER))
        self.vem.add(ChangeViewButton(self, self.hulls_ax, "Hulls", ViewsEnum.HULLS))

    def draw(self, *args, **kwargs) -> None:
        super().draw()

        self.state.show_labels_and_hulls(self.vm.ax)
        self.vm.list_manager.hide_button()
        self.vm.ax.set_xlim(-190, 190)
        self.vm.ax.set_ylim(-150, 150)

        # connect auto refreshing
        self.vem.refresh_connect(self.vm.fig)

        plt.draw()

    def hide(self) -> None:
        self.vm.list_manager.show_button()
        super().hide()


class LabelsView(View):

    def __init__(self, view_manager: ViewManager) -> None:
        super().__init__(view_manager)
        self.dragged_item: LabelArtist | None = None
        self.picked_item: LabelArtist | None = None
        self.pick_pos: tuple[float, float] | None = None
        self.events_stack = []

        # buttons
        view_button = ChangeViewButton(self, self.labels_ax, "Labels", ViewsEnum.LABELS)
        view_button.highlight()
        self.vem.add(view_button)
        self.vem.add(ChangeViewButton(self, self.home_ax, "Home", ViewsEnum.HOME))
        self.vem.add(ChangeViewButton(self, self.clusters_ax, "Cluster", ViewsEnum.CLUSTER))
        self.vem.add(ChangeViewButton(self, self.hulls_ax, "Hulls", ViewsEnum.HULLS))
        self.vem.add(NormalButton(self, [0.075, 0.05, 0.05, 0.075], "+", self.add_label))
        self.vem.add(NormalButton(self, [0.125, 0.05, 0.05, 0.075], "-", self.delete_label))
        self.vem.add(NormalButton(self, [0.575, 0.0875, 0.05, 0.0375], "^", self.font_size_up))
        self.vem.add(NormalButton(self, [0.575, 0.05, 0.05, 0.0375], "v", self.font_size_down))
        self.vem.add(NormalButton(self, [0.75, 0.0875, 0.05, 0.0375], "^", self.arrow_size_up))
        self.vem.add(NormalButton(self, [0.75, 0.05, 0.05, 0.0375], "v", self.arrow_size_down))
        self.vem.add(NormalButton(self, [0.825, 0.05, 0.10, 0.075], "+arrow", self.add_arrow))

        # displays
        self.vem.add(ShiftingTextBox(self, [0.2, 0.05, 0.25, 0.075],
                                     self.label_name_update, self.label_name_submit))
        self.vem.add(LimitedTextBox(self, [0.475, 0.05, 0.10, 0.075],
                                    self.font_size_update, self.font_size_submit))
        self.vem.add(LimitedTextBox(self, [0.65, 0.05, 0.10, 0.075],
                                    self.arrow_size_update, self.arrow_size_submit))

        self.vem.hide()

    def draw(self, *args, **kwargs) -> None:
        super().draw()

        self.state.show_labels_and_hulls(self.vm.ax)
        self.vm.ax.set_xlim(-190, 190)
        self.vm.ax.set_ylim(-150, 150)

        self.events_stack.clear()
        # get picked arrow if exists
        event = kwargs.get('event', None)
        self.picked_item = kwargs.get('picked_item', None)
        if event is not None:
            self.pick_event(event)

        # events
        self.cem.add(SharedEvent('pick_event', self.pick_event))
        self.cem.add(SharedEvent('button_release_event', self.release_event))
        self.cem.add(SharedEvent('key_press_event', self.key_press_event))
        self.cem.add(SharedEvent('resize_event', self.resize_label_update))

        self.vem.refresh()
        plt.draw()

    def hide(self) -> None:
        super().hide()

    def pick_event(self, event: PickEvent) -> None:
        logging.info(f"""{self.__class__} EVENT: {event} ARTIST: {event.artist} 
ID: {getattr(event.artist, 'id', None)}""")
        if isinstance(event.artist, LabelArtist):
            self.events_stack.append(event.artist.get_state())
            self.dragged_item = event.artist
            self.picked_item = event.artist
            self.pick_pos = (event.mouseevent.xdata, event.mouseevent.ydata)

            # update fields
            self.vem.refresh()

        if isinstance(event.artist, ArrowArtist):
            # check if click overlaps with other artists
            artists = get_artists_by_type(self.vm.ax, LabelArtist)
            for artist in artists:
                if artist.contains(event.mouseevent)[0]:
                    logging.info(f"""{self.__class__} EVENT: {event} 
canceled due to overlapping Label: {artist}""")
                    return

            # pass current artist as kwarg for new view to start initiated
            self.change_view(ViewsEnum.ARROWS, picked_item=event.artist)

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
        self.picked_item.arrows[nid] = ArrowArtist.arrow(self.vm.ax, nid, x, y, rfx, rfy,
                                                         atx - x, aty - y, self.picked_item, val)

        plt.draw()

    def delete_label(self) -> None:
        if self.picked_item is None:
            return

        self.picked_item.remove()
        self.picked_item = None
        self.vem.refresh()

    def ctrlz(self) -> None:
        if self.events_stack:
            sid, *state = self.events_stack.pop()
            l = LabelArtist.get_by_id(self.vm.ax, sid)
            if l is not None:
                l.set_state(state)
            plt.draw()

    def key_press_event(self, event: KeyEvent) -> None:
        logging.info(f"KEY PRESS: {event.key}")
        if event.key == "ctrl+z":
            self.ctrlz()

    def label_name_update(self) -> str:
        if self.picked_item is None:
            return ''
        return self.picked_item.get_text()

    def label_name_submit(self, nname) -> None:
        if self.picked_item is None:
            return
        if nname != "":
            self.picked_item.set_text(nname)
            plt.draw()

    def font_size_update(self) -> int:
        return self.state.get_label_size()

    def font_size_submit(self, size: str) -> None:
        try:
            fsize = float(size)
            if not 0 < fsize <= 50:
                return
            LabelArtist.update_all_labels_fontsize(self.vm.ax, fsize)
            plt.draw()
        except ValueError:
            return

    def font_size_up(self) -> None:
        size = self.state.get_label_size()
        size += 1
        if not 0 < size <= 50:
            return
        self.state.set_label_size(size)
        LabelArtist.update_all_labels_fontsize(self.vm.ax, size)
        self.vem.refresh()
        plt.draw()

    def font_size_down(self) -> None:
        size = self.state.get_label_size()
        size -= 1
        if not 0 < size <= 50:
            return
        self.state.set_label_size(size)
        LabelArtist.update_all_labels_fontsize(self.vm.ax, size)
        self.vem.refresh()
        plt.draw()

    def resize_label_update(self, event: ResizeEvent) -> None:
        # print(self.vm.fig.get_window_extent())
        ...

    def arrow_size_update(self) -> int:
        return self.state.get_arrow_size()

    def arrow_size_submit(self, size: str) -> None:
        try:
            fsize = float(size)
            if not 0 < fsize <= 10:
                return
            ArrowArtist.update_all_arrows_size(self.vm.ax, fsize)
            plt.draw()
        except ValueError:
            return

    def arrow_size_up(self) -> None:
        size = self.state.get_arrow_size()
        size += 0.1
        if not 0 < size <= 10:
            return
        self.state.set_arrow_size(size)
        ArrowArtist.update_all_arrows_size(self.vm.ax, size)
        self.vem.refresh()
        plt.draw()

    def arrow_size_down(self) -> None:
        size = self.state.get_arrow_size()
        size -= 0.1
        if not 0 < size <= 10:
            return
        self.state.set_arrow_size(size)
        ArrowArtist.update_all_arrows_size(self.vm.ax, size)
        self.vem.refresh()
        plt.draw()

    def resize_arrow_update(self, event: ResizeEvent) -> None:
        # print(self.vm.fig.get_window_extent())
        ...


class ArrowsView(View):

    def __init__(self, view_manager: ViewManager) -> None:
        super().__init__(view_manager)
        self.picked_item: ArrowArtist | None = None

        # buttons
        view_button = ChangeViewButton(self, self.labels_ax, "Labels", ViewsEnum.LABELS)
        view_button.highlight()
        self.vem.add(view_button)
        self.vem.add(ChangeViewButton(self, self.home_ax, "Home", ViewsEnum.HOME))
        self.vem.add(ChangeViewButton(self, self.clusters_ax, "Cluster", ViewsEnum.CLUSTER))
        self.vem.add(ChangeViewButton(self, self.hulls_ax, "Hulls", ViewsEnum.HULLS))
        self.vem.add(NormalButton(self, [0.8, 0.05, 0.1, 0.075], "Delete", self.delete_arrow))
        self.vem.add(BlockingButton(self, [0.35, 0.05, 0.05, 0.075], "p", self.sh_point_picker))
        self.vem.add(BlockingButton(self, [0.7, 0.05, 0.05, 0.075], "p", self.rf_point_picker))

        # displays
        self.vem.add(LimitedTextBox(self, [0.15, 0.05, 0.10, 0.075],
                                    self.arrow_shx_update, self.arrow_shx_submit, "Att:"))
        self.vem.add(LimitedTextBox(self, [0.25, 0.05, 0.10, 0.075],
                                    self.arrow_shy_update, self.arrow_shy_submit))
        self.vem.add(LimitedTextBox(self, [0.5, 0.05, 0.10, 0.075],
                                    self.arrow_rfx_update, self.arrow_rfx_submit, "Ref:"))
        self.vem.add(LimitedTextBox(self, [0.6, 0.05, 0.10, 0.075],
                                    self.arrow_rfy_update, self.arrow_rfy_submit))

        self.vem.hide()

    def draw(self, *args, **kwargs) -> None:
        super().draw()

        # get picked arrow if exists
        self.picked_item = kwargs.get('picked_item', None)

        self.cem.add(SharedEvent('pick_event', self.pick_event))

        self.vem.refresh()
        plt.draw()

    def hide(self) -> None:
        return super().hide()

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

    def pick_event(self, event: PickEvent) -> None:
        logging.info(f"""{self.__class__} EVENT: {event} ARTIST: {event.artist} 
ID: {event.artist.id}""")
        if isinstance(event.artist, ArrowArtist):

            artists = get_artists_by_type(self.vm.ax, LabelArtist)
            for artist in artists:
                if artist.contains(event.mouseevent)[0]:
                    logging.info(f"""{self.__class__} EVENT: {event} 
canceled due to overlapping Label: {artist}""")
                    return

            self.picked_item = event.artist
            self.vem.refresh()

        if isinstance(event.artist, LabelArtist):
            self.change_view(ViewsEnum.LABELS, picked_item=event.artist, event=event)
            return

    def delete_arrow(self) -> None:
        if self.picked_item is None:
            return

        self.picked_item.remove()
        self.picked_item = None
        self.vem.refresh()

    def sh_point_picker(self, reconect_callback: Callable[..., None]) -> None:
        self.cem.add(UniqueEvent('button_press_event',
                                 lambda event, *args, **kwargs: self.sh_point_pick_event(
                                     reconect_callback, event, *args, **kwargs)))

    def sh_point_pick_event(self,
                            reconect_callback: Callable[..., None],
                            event: MouseEvent) -> None:
        if event.inaxes is not self.vm.ax:
            logging.info("Point pick event out of chart.")
            return

        self.picked_item.set_sh_by_raw(event.xdata, event.ydata)

        self.cem.disconnect_unique()
        reconect_callback()
        self.vem.refresh()

    def rf_point_picker(self, reconect_callback: Callable[..., None]) -> None:
        self.cem.add(UniqueEvent('button_press_event', lambda event, *args, **kwargs: \
            self.rf_point_pick_event(reconect_callback,
                                     event, *args, **kwargs)))

    def rf_point_pick_event(self,
                            reconect_callback: Callable[..., None],
                            event: MouseEvent) -> None:
        if event.inaxes is not self.vm.ax:
            logging.info("Point pick event out of chart.")
            return

        self.picked_item.set(rfx=event.xdata, rfy=event.ydata)

        self.cem.disconnect_unique()
        reconect_callback()
        self.vem.refresh()


class ClusterMainView(View):
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

    def __init__(self, view_manager: ViewManager) -> None:
        super().__init__(view_manager)
        self.picked_item: PointArtist | None = None
        self.pick_pos: tuple[float, float] | None = None
        self.events_stack = []
        self.info_text = None
        self.hulls_off = True

        # buttons
        view_button = ChangeViewButton(self, self.clusters_ax, "Cluster", ViewsEnum.CLUSTER)
        view_button.highlight()
        self.vem.add(view_button)
        self.vem.add(ChangeViewButton(self, self.home_ax, "Home", ViewsEnum.HOME))
        self.vem.add(ChangeViewButton(self, self.labels_ax, "Labels", ViewsEnum.LABELS))
        self.vem.add(ChangeViewButton(self, self.hulls_ax, "Hulls", ViewsEnum.HULLS))
        self.vem.add(ChangeViewButton(self, [0.75, self.change_button_y, self.change_button_length, self.change_button_height], "Agglo", ViewsEnum.AGGLOMERATIVE))
        self.vem.add(ChangeViewButton(self, [0.85, self.change_button_y, self.change_button_length, self.change_button_height], "DBSCAN", ViewsEnum.DBSCAN))
        self.vem.add(NormalButton(self, [0.05, 0.05, 0.1, 0.075], "Remove", self.remove_point))
        self.vem.add(NormalButton(self, [0.17, 0.05, 0.15, 0.075], "Toggle hulls", self.draw_hull))
        reset_b = NormalButton(self, [0.85, 0.05, 0.1, 0.075], "Reset", self.reset_clusters)
        reset_b.button_ax.set_facecolor("lightcoral")
        reset_b.button_ref.color = "lightcoral"
        reset_b.button_ref.hovercolor = "crimson"
        # todo fix color being updated only after movement
        self.vem.add(reset_b)

        self.vem.hide()

    def draw(self, *args, **kwargs) -> None:
        super().draw()

        # clear ax by hiding elements
        self.state.hide_labels_and_hulls(self.vm.ax)
        self.hulls_off = True

        # make points more transparent
        for artist in self.state.data['clusters_data']['artists']:
            artist.set_alpha(0.3)

        # add pick event
        self.cem.add(SharedEvent('pick_event', self.pick_event))

        # make main plot larger
        df = self.state.get_all_points()
        # setting lims manually since relim and autoscale don't perform well
        self.vm.ax.set_xlim(df['x'].min()-10, df['x'].max()+10)
        self.vm.ax.set_ylim(df['y'].min()-10, df['y'].max()+10)

        # todo make Text a view element
        self.info_text = Text(0, 0, "Info")
        self.info_text.set_position((self.vm.ax.get_xlim()[0] + 3, self.vm.ax.get_ylim()[1] - 10))
        self.vm.ax.add_artist(self.info_text)

        self.vem.refresh()
        plt.draw()

    def hide(self) -> None:
        super().hide()
        self.reset_pick_event()
        self.info_text.remove()
        for artist in self.state.data['clusters_data']['artists']:
            artist.set_alpha(1)
        self.vm.ax.set_xlim(-190, 190)
        self.vm.ax.set_ylim(-150, 150)

    def reset_clusters(self):
        self.state.reset_clusters()
        for artist in self.state.data['clusters_data']['artists']:
            artist.set_color(self.state.get_point_color(artist.id))
        plt.draw()

    def remove_point(self):
        if self.picked_item:
            self.reset_pick_event()
            point = self.state.get_point(self.picked_item.id)
            point_hull = point['type']
            self.state.set_hull_to_undraw(point_hull)

            self.state.set_cluster("Removed", [self.picked_item.id])
            self.info_text.set_text("Removed")

            self.state.set_hull_to_change(point_hull, self.state.get_cluster(point_hull))

            artist = self.state.data['clusters_data']['artists'][self.picked_item.id]
            artist.set_color(self.state.get_point_color(self.picked_item.id))

            plt.draw()

    def reset_pick_event(self):
        if self.picked_item:
            if self.picked_item.axes is not None:
                self.picked_item.remove()
            point = self.state.get_point(self.picked_item.id)
            for point_id in self.state.get_cluster(point['type']).index:
                artist = self.state.data['clusters_data']['artists'][point_id]
                artist.set_alpha(0.3)
                artist.set_edgecolor(self.state.get_point_color(point_id))

    def pick_event(self, event: PickEvent) -> None:
        self.reset_pick_event()
        self.picked_item = PointArtist.point(self.vm.ax, event.artist.id, facecolor="red", edgecolor="red")
        point = self.state.get_point(self.picked_item.id)

        for point_id in self.state.get_cluster(point['type']).index:
            artist = self.state.data['clusters_data']['artists'][point_id]
            artist.set_alpha(1)
            artist.set_edgecolor("black")

        self.info_text.set_text(point['type'])
        plt.draw()

    def draw_hull(self):
        # pretty sure hulls don't get removed properly
        # todo after merge with hulls make sure we don't have old hulls stay in memory
        if self.hulls_off:
            self.hulls_off = False
            for child in self.vm.ax.get_children():
                if type(child) is LineCollection:
                    child.remove()
            self.state.update_hulls()

            for hull_name in self.state.data['hulls_data']['hulls'].keys():
                HullArtist.hull(self.vm.ax, hull_name)
        else:
            self.hulls_off = True
            HullArtist.hide_hulls(self.vm.ax)
        plt.draw()


class ClusteringSubViewBase(View):
    @abstractmethod
    def __init__(self, view_manager: ViewManager) -> None:
        """Adds buttons and widgets common to all sub views"""
        super().__init__(view_manager)
        self.info_text = None
        self.previous_cluster_name = None
        self.current_labels = None
        self.widget_cluster_name = None

        self.vem.add(ChangeViewButton(self, self.home_ax, "Home", ViewsEnum.HOME))
        self.vem.add(ChangeViewButton(self, self.labels_ax, "Labels", ViewsEnum.LABELS))
        self.vem.add(ChangeViewButton(self, self.clusters_ax, "Cluster", ViewsEnum.CLUSTER))
        self.vem.add(ChangeViewButton(self, self.hulls_ax, "Hulls", ViewsEnum.HULLS))
        self.vem.add(ChangeViewButton(self, [0.85, 0.936, 0.1, 0.06], "Back", ViewsEnum.CLUSTER))
        self.vem.add(NormalButton(self, [0.05, 0.05, 0.15, 0.075], "Save cluster", self.save_cluster))
        reset_b = NormalButton(self, [0.85, 0.05, 0.1, 0.075], "Reset", self.reset)
        reset_b.button_ax.set_facecolor("lightcoral")
        reset_b.button_ref.color = "lightcoral"
        reset_b.button_ref.hovercolor = "crimson"
        self.vem.add(reset_b)
        # In concrete class: add to vem the view specific elements and call vem.hide()

    @abstractmethod
    def count_clustering(self, points_xy):
        """Return the fit results of a clustering algorithm"""
        pass

    def draw(self, *args, **kwargs) -> None:
        super().draw()

        self.widget_cluster_name = ViewRadioButtons(
            self,
            [0.05, 0.15, 0.3, 0.75],
            sorted(list(self.state.get_all_clusters().keys())),
            self.update_plot,
        )
        self.vem.add(self.widget_cluster_name)

        # clear ax by hiding elements
        self.state.hide_labels_and_hulls(self.vm.ax)
        self.vm.list_manager.hide_button()

        # make points more transparent
        for artist in self.state.data['clusters_data']['artists']:
            artist.set_alpha(0.3)

        # move plot to right top corner
        plt.subplots_adjust(bottom=0.3, left=0.4, top=0.9, right=0.9)
        # make plot larger
        df = self.state.get_all_points()
        # setting lims manually since relim and autoscale don't perform well
        self.vm.ax.set_xlim(df['x'].min() - 10, df['x'].max() + 10)
        self.vm.ax.set_ylim(df['y'].min() - 10, df['y'].max() + 10)

        # todo make Text a view element
        self.info_text = Text(self.vm.ax.get_xlim()[0] + 3, self.vm.ax.get_ylim()[1] - 10, "Info")
        self.vm.ax.add_artist(self.info_text)

        self.update_plot(None)

    def hide(self) -> None:
        super().hide()
        self.dehighlight_previous_cluster()
        self.info_text.remove()
        plt.subplots_adjust(bottom=0.15, left=0.01, right=0.99, top=0.935)
        for artist in self.state.data['clusters_data']['artists']:
            artist.set_alpha(1)
        self.vm.ax.set_xlim(-190, 190)
        self.vm.ax.set_ylim(-150, 150)
        self.widget_cluster_name.remove()
        self.vm.list_manager.show_button()

    def dehighlight_previous_cluster(self):
        """Resets currently picked cluster points to their original look"""
        cluster_name = self.previous_cluster_name
        for point_id in self.state.get_cluster(cluster_name).index:
            artist = self.state.data['clusters_data']['artists'][point_id]
            artist.set_color(self.state.get_point_color(point_id))
            artist.set_alpha(0.3)
            artist.set_radius(1.5)
            artist.set_zorder(1)

    def highlight_current_cluster(self, cluster_name, colors):
        """Makes currently picked cluster points more visible"""
        idx = 0
        for point_id in self.state.get_cluster(cluster_name).index:
            artist = self.state.data['clusters_data']['artists'][point_id]
            artist.set_color(colors[idx])
            artist.set_alpha(1)
            artist.set_radius(2.5)
            artist.set_zorder(10)
            idx += 1

    def update_plot(self, widget_value):
        """Updates the plot to match currently picked values"""
        cluster_name = self.widget_cluster_name.ref.value_selected
        current_cluster = self.state.get_cluster(cluster_name)

        # get colors to match labels of clustering
        if current_cluster.shape[0] > 1:
            clustering = self.count_clustering(current_cluster[['x', 'y']])
            colors = mpl.colormaps["tab10"](clustering.labels_)
            colors = ["black" if clustering.labels_[idx] == -1 else x for idx, x in enumerate(colors)]
        else:
            colors = ["black"]

        # update visuals
        self.info_text.set_text("Info")
        self.dehighlight_previous_cluster()
        self.highlight_current_cluster(cluster_name, colors)
        plt.draw()

        self.previous_cluster_name = cluster_name
        self.current_labels = clustering.labels_

    def save_cluster(self):
        """"Saves the currently picked cluster"""
        # dehighlight first as we're gonna break the cluster
        self.dehighlight_previous_cluster()

        cluster_name = self.widget_cluster_name.ref.value_selected
        current_cluster = self.state.get_cluster(cluster_name)

        label_map = {key: [] for key in set(self.current_labels)}

        for idx, point_id in enumerate(current_cluster.index):
            label_map[self.current_labels[idx]].append(point_id)

        for key, points in label_map.items():
            if key == -1:
                continue
            new_cluster_name = cluster_name + str(key)
            self.state.set_cluster(new_cluster_name, points)
            self.state.set_hull_to_undraw(cluster_name)
            self.state.set_hull_to_change(new_cluster_name, self.state.get_cluster(new_cluster_name))

            for point_id in points:
                artist = self.state.data['clusters_data']['artists'][point_id]
                artist.set_color(self.state.get_point_color(point_id))

        self.widget_cluster_name.remove()
        self.widget_cluster_name = ViewRadioButtons(self, [0.05, 0.15, 0.3, 0.75],
                                                    sorted(list(self.state.get_all_clusters().keys())),
                                                    self.update_plot)
        self.vem.add(self.widget_cluster_name)

        self.info_text.set_text("Saved")
        plt.draw()

    def reset(self):
        """Resets clusters back to the original state"""
        self.state.reset_clusters()
        self.widget_cluster_name.remove()
        self.widget_cluster_name = ViewRadioButtons(self, [0.05, 0.15, 0.3, 0.75],
            sorted(list(self.state.get_all_clusters().keys())), self.update_plot)
        self.vem.add(self.widget_cluster_name)
        self.update_plot(None)


class DBSCANView(ClusteringSubViewBase):
    def __init__(self, view_manager: ViewManager) -> None:
        super().__init__(view_manager)

        self.widget_scalar = ViewSlider(
            self, [0.55, 0.17, 0.3, 0.05], "", 0.01, 2.5, self.update_plot
        )
        self.vem.add(self.widget_scalar)
        self.vem.hide()

    def count_clustering(self, points_xy):
        return DBSCAN(eps=self.widget_scalar.ref.val*10, min_samples=2).fit(points_xy)

class AgglomerativeView(ClusteringSubViewBase):
    def __init__(self, view_manager: ViewManager) -> None:
        super().__init__(view_manager)
        self.widget_linkage = ViewRadioButtons(self, [0.4, 0.15, 0.1, 0.1],
            ["ward", "complete", "average", "single"], self.update_plot, 3)
        self.widget_scalar = ViewSlider(self, [0.55, 0.17, 0.3, 0.05],
            "", 0.01, 2.5, self.update_plot)

        self.vem.add(self.widget_linkage)
        self.vem.add(self.widget_scalar)
        self.vem.hide()

    def count_clustering(self, points_xy):
        dist = AgglomerativeClustering(n_clusters=1, compute_distances=True).fit(points_xy).distances_
        mean = np.mean(dist)
        return AgglomerativeClustering(
                n_clusters=None,
                linkage=self.widget_linkage.ref.value_selected,
                distance_threshold=mean * self.widget_scalar.ref.val
            ).fit(points_xy)

class HullView(View):

    def __init__(self, view_manager: ViewManager) -> None:
        super().__init__(view_manager)
        self.dragged_item: HullArtist | None = None
        self.picked_item: HullArtist | None = None
        self.events_stack = []
        self.seq_remove_line_1 = 0
        self.seq_remove_line_2 = 0
        self.event_xdata = 0
        self.event_ydata = 0
        self.line_1_id = 0
        self.line_2_id = 0

        view_button = ChangeViewButton(self, self.hulls_ax, "Hulls", ViewsEnum.HULLS)
        # view_button.highlight()
        self.vem.add(view_button)
        self.vem.add(ChangeViewButton(self, self.home_ax, "Home", ViewsEnum.HOME))
        self.vem.add(ChangeViewButton(self, self.labels_ax, "Labels", ViewsEnum.LABELS))
        self.vem.add(ChangeViewButton(self, self.clusters_ax, "Cluster", ViewsEnum.CLUSTER))
        self.vem.add(ChangeViewButton(self, [0.3, 0.05, 0.15, 0.075], "Create hulls", ViewsEnum.CREATEHULL))
        self.vem.add(ChangeViewButton(self, [0.50, 0.05, 0.15, 0.075], "Remove line", ViewsEnum.REMOVELINE))
        # self.vem.add(NormalButton(self, [0.3, 0.05, 0.15, 0.075], "Remove Line", self.remove_line))
        # self.vem.add(NormalButton(self, [0.50, 0.05, 0.15, 0.075], "Remove Hull", self.remove_hull))
        self.vem.hide()

    def draw(self, *args, **kwargs) -> None:
        super().draw()

        # HullArtist.hide_hulls(self.vm.ax)
        self.state.show_labels_and_hulls(self.vm.ax)
        # print("WITAM")
        self.vm.ax.set_xlim(-190, 190)
        self.vm.ax.set_ylim(-150, 150)
        self.events_stack.clear()
        self.picked_item = kwargs.get('picked_item', None)

        for child in self.vm.ax.get_children():
            if type(child) is LineCollection:
                child.remove()

        for hull_name in self.state.data['hulls_data']['hulls'].keys():
                HullArtist.hull(self.vm.ax, hull_name)

        # events
        self.cem.add(SharedEvent('pick_event', self.pick_event))

        self.vem.refresh()
        plt.draw()

    def pick_event(self, event: PickEvent) -> None:
        logging.info(f"""{self.__class__} EVENT: {event} ARTIST: {event.artist} 
                     ID: {getattr(event.artist, 'id', None)}""")
        
        if isinstance(event.artist, HullArtist):
            
            if self.seq_remove_line_1 == 1:
                self.line_1_id = getattr(event.artist, 'id', None)
                print("ID_1")
                print(self.line_1_id)
                self.seq_remove_line_1 += 1

            elif self.seq_remove_line_1 == 3:
                self.line_2_id = getattr(event.artist, 'id', None)
                print("ID_2")
                print(self.line_2_id)
                self.seq_remove_line_1 = -1

            self.events_stack.append(event.artist.get_state())
            self.picked_item  = event.artist
            self.pick_pos = (event.mouseevent.xdata, event.mouseevent.ydata)

            # update fields
            self.vem.refresh()

    def press_event(self, event: MouseEvent) -> None:
        logging.info(f"""{self.__class__} POS: {event.xdata}, {event.ydata}""")

    def remove_line(self) -> None:
        self.seq_remove_line_1 = 0

    def _exec_remove_line(self):
        ...

    def get_closest_hull(self, coordx, coordy):
        ...


    def remove_hull(self) -> None:
        if self.picked_item is None:
            return

        self.pick_event.remove()
        self.picked_item = None
        self.vem.refresh()

    def hide(self) -> None:
        super().hide()

class CreateNewHullView(View):

    def __init__(self, view_manager: ViewManager) -> None:
        super().__init__(view_manager)
        self.pick_pos: tuple[float, float] | None = None
        self.events_stack = []
        self.is_adding_points: bool = False
        self.points_to_add = []
        self.pointer_points = []

        # buttons
        view_button = ChangeViewButton(self, self.labels_ax, "Labels", ViewsEnum.LABELS)
        # view_button.highlight()
        self.vem.add(view_button)
        self.vem.add(ChangeViewButton(self, self.home_ax, "Home", ViewsEnum.HOME))
        self.vem.add(ChangeViewButton(self, self.clusters_ax, "Cluster", ViewsEnum.CLUSTER))
        self.vem.add(ChangeViewButton(self, [0.85, 0.936, 0.1, 0.06], "Back", ViewsEnum.HULLS))

        self.pointer_end_add_points = NormalButton(self, [0.075, 0.05, 0.2, 0.075], "End adding", self.end_add_points)
        self.vem_pointer_end_add_points = self.vem.add(self.pointer_end_add_points)

        self.pointer_add_points = NormalButton(self, [0.075, 0.05, 0.2, 0.075], "Add points", self.create_from_existing)
        self.vem_pointer_add_points = self.vem.add(self.pointer_add_points)


        # self.vem.add(NormalButton(self, [0.125, 0.05, 0.05, 0.075], "-", self.delete_label))
        # self.vem.add(NormalButton(self, [0.575, 0.0875, 0.05, 0.0375], "^", self.font_size_up))
        # self.vem.add(NormalButton(self, [0.575, 0.05, 0.05, 0.0375], "v", self.font_size_down))
        # self.vem.add(NormalButton(self, [0.75, 0.0875, 0.05, 0.0375], "^", self.arrow_size_up))
        # self.vem.add(NormalButton(self, [0.75, 0.05, 0.05, 0.0375], "v", self.arrow_size_down))
        # self.vem.add(NormalButton(self, [0.825, 0.05, 0.10, 0.075], "+arrow", self.add_arrow))


        self.vem.hide()

    def draw(self, *args, **kwargs) -> None:

        super().draw()

        self.state.show_labels_and_hulls(self.vm.ax)
        self.vm.ax.set_xlim(-190, 190)
        self.vm.ax.set_ylim(-150, 150)
        self.events_stack.clear()
        self.picked_item = kwargs.get('picked_item', None)

        # events
        self.cem.add(SharedEvent('pick_event', self.pick_event))
        self.cem.add(SharedEvent('button_press_event', self.press_event))

        for child in self.vm.ax.get_children():
            if type(child) is LineCollection:
                child.remove()

        for hull_name in self.state.data['hulls_data']['hulls'].keys():
                HullArtist.hull(self.vm.ax, hull_name)

        self.vem.refresh()
        plt.draw()

    def hide(self) -> None:
        super().hide()

    def pick_event(self, event: PickEvent) -> None:
        logging.info(f"""{self.__class__} EVENT: {event} ARTIST: {event.artist} 
                     ID: {getattr(event.artist, 'id', None)}""")
        
        if isinstance(event.artist, HullArtist):
            
            if self.seq_remove_line_1 == 1:
                self.line_1_id = getattr(event.artist, 'id', None)
                print("ID_1")
                print(self.line_1_id)
                self.seq_remove_line_1 += 1

            elif self.seq_remove_line_1 == 3:
                self.line_2_id = getattr(event.artist, 'id', None)
                print("ID_2")
                print(self.line_2_id)
                self.seq_remove_line_1 = -1

            self.events_stack.append(event.artist.get_state())
            self.picked_item  = event.artist
            self.pick_pos = (event.mouseevent.xdata, event.mouseevent.ydata)

            # update fields
            self.vem.refresh()

    def press_event(self, event: MouseEvent) -> None:
        if self.is_adding_points:
            logging.info(f"""{self.__class__} POS: {event.xdata}, {event.ydata}""")
            self.points_to_add.append((event.xdata, event.ydata))

            self.state._hull_set_point(len(self.points_to_add) * (-1), event.xdata, event.ydata)
            self.pointer_points.append(PointArtist.point(self.vm.ax, len(self.points_to_add) * (-1), facecolor="red", edgecolor="red"))
            
            self.vem.refresh()
            plt.draw()


    def search_for_hull_name_in_hole(self, point, _hull_name = ""):
        hulls_name = self.state.get_all_hulls_name()
        best_dist = float("inf")
        best_hull_name = ""
        best_cord = (0,0)

        for hull_name in hulls_name:
            cords = self.state.get_hole_in_hulls(hull_name)
            print(f"CORDS: {cords}")
            if _hull_name != "" and _hull_name != hull_name:
                continue
            for cord in cords:
                dist_1 = np.sqrt((cord[0][0] - point[0]) ** 2 + (cord[0][1] - point[1]) ** 2)
                dist_2 = np.sqrt((cord[1][0] - point[0]) ** 2 + (cord[1][1] - point[1]) ** 2)

                dist = dist_1 if dist_1 < dist_2 else dist_2

                if dist < 10 and dist < best_dist:
                    best_dist = dist
                    best_hull_name = hull_name
                    best_cord = cord
        
        return best_hull_name, best_cord
    
    def distance(self, point_1, point_2):
        return np.sqrt((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2)

    def find_closest_edge_point_in_hull(self, hull_name, point):
        hull_cords = self.state.get_hull_interpolated_cords(hull_name)
        best_cords = (0,0)
        best_dist = float("inf")
        best_idx = -1

        for i, hull_cord in enumerate(hull_cords):
            dist = np.sqrt((hull_cord[0] - point[0]) ** 2 + (hull_cord[1] - point[1]) ** 2)
            if dist < best_dist:
                best_cords = hull_cord
                best_dist = dist
                best_idx = i
        
        return best_cords, best_idx

    def create_from_existing(self):
        self.is_adding_points = True

        self.pointer_add_points.remove()
        self.vem.elements.remove(self.vem_pointer_add_points)

        self.vem.refresh()
        
        self.pointer_end_add_points = NormalButton(self, [0.075, 0.05, 0.2, 0.075], "End adding", self.end_add_points)
        self.vem_pointer_end_add_points = self.vem.add(self.pointer_end_add_points)
        
        self.vem.refresh()
        plt.draw()

    def end_add_points(self):
        self.is_adding_points = False
        self.calculate_new_hull()

        self.pointer_end_add_points.remove()
        self.vem.elements.remove(self.vem_pointer_end_add_points)

        self.vem.refresh()

        self.pointer_add_points = NormalButton(self, [0.075, 0.05, 0.2, 0.075], "Add points", self.create_from_existing)
        self.vem_pointer_add_points = self.vem.add(self.pointer_add_points)
        
        self.vem.refresh()
        plt.draw()

    def interpolate_points(self, points, num=100):
        if len(points) < 3:
            return points
        u3 = np.linspace(0, 1, num, endpoint=True)

        np_points = np.array(points)
        x = np_points[:, 0]
        y = np_points[:, 1]

        tck, _ = interpolate.splprep([x, y], k=3, s=0)
        new_points = interpolate.splev(u3, tck)
        return list(zip(new_points[0], new_points[1]))

    def calculate_new_hull(self):
        # self.state -> doable
        print("POINTS TO ADD")
        self.points_to_add = self.points_to_add[:-1]
        print(self.points_to_add)

        hull_name, hole_cord = self.search_for_hull_name_in_hole(self.points_to_add[0])
        print(hull_name)
        print(hole_cord)

        final_cords = []
        final_lines = []

        if hull_name == "":

            interpolated_points = self.interpolate_points(self.points_to_add)
            polygon_lines = [
                (
                    interpolated_points[j % len(interpolated_points)],
                    interpolated_points[(j + 1) % len(interpolated_points)],
                )
                for j in range(len(interpolated_points))
            ]

            final_cords = interpolated_points
            final_lines = polygon_lines 
            
            next_name = self.state.get_hulls_render_name()
            _name = f'hull_{next_name}'
            print(_name)
            self.state.set_hulls_render_name = next_name
            
            self.state.add_hull(hull_name=_name, line_cords=final_lines, interpolate_points=final_cords)

            HullArtist.hull(self.vm.ax, _name)
        

            for i, pointer_point in enumerate(self.pointer_points):
                pointer_point.remove()
                self.state._hull_remove_point(i * (-1))

            self.pointer_points = []

            self.points_to_add = []

            self.vem.refresh()
            plt.draw()
            return
        hull_interpolated_points = self.state.get_hull_interpolated_cords(hull_name)
        lines = self.state.get_hull_lines_cords(hull_name)

        point_1, idx_1 = self.find_closest_edge_point_in_hull(hull_name, hole_cord[0])
        point_2, idx_2 = self.find_closest_edge_point_in_hull(hull_name, hole_cord[1])

        line_start_index = -1
        line_end_index = -1

        print(point_1, point_2)
        print(f"IDX: {idx_1} {idx_2}")
        interpolated_points = self.interpolate_points(self.points_to_add[1:-1])


        for index, (line_point_1, line_point_2) in enumerate(lines):
            if ((point_1[0] == line_point_1[0] and point_1[1] == line_point_1[1]) 
                or (point_1[0] == line_point_2[0] and point_1[1] == line_point_2[1])):
                line_start_index = index
            
        for index, (line_point_1, line_point_2) in enumerate(lines):
            if ((point_2[0] == line_point_1[0] and point_2[1] == line_point_1[1]) 
                or (point_2[0] == line_point_2[0] and point_2[1] == line_point_2[1])):
                line_end_index = index

        if idx_1 < idx_2:

            first_part_hull_points = hull_interpolated_points[:idx_1+1]
            second_part_hull_points = hull_interpolated_points[idx_2:]
            
            line_first_part_hull_points = lines[:line_start_index+1]
            line_second_part_hull_points = lines[line_end_index:]

            polygon_lines = [
                (
                    interpolated_points[j % len(interpolated_points)],
                    interpolated_points[(j + 1) % len(interpolated_points)],
                )
                for j in range(len(interpolated_points) - 1)
            ]

            print(polygon_lines)

            final_cords = first_part_hull_points + interpolated_points + second_part_hull_points
            final_lines = (line_first_part_hull_points 
                           + [((line_first_part_hull_points[-1][1]), (polygon_lines[0][0]))] 
                           + polygon_lines 
                           + [((polygon_lines[-1][1]), (line_second_part_hull_points[0][0]))] 
                           + line_second_part_hull_points)
            
                


        hull_artist = HullArtist.get_by_id(self.vm.ax, hull_name)
        hull_artist.remove()
        self.vem.refresh()

        self.state.remove_hole_in_hulls(hull_name, hole_cord)

        self.state.set_hull_interpolated_cords(hull_name, final_cords)
        self.state.set_hull_lines_cords(hull_name, final_lines)

        HullArtist.hull(self.vm.ax, hull_name)
        

        for i, pointer_point in enumerate(self.pointer_points):
            pointer_point.remove()
            self.state._hull_remove_point(i * (-1))

        self.pointer_points = []

        self.points_to_add = []

        self.vem.refresh()
        plt.draw()


    def hide(self) -> None:
        super().hide()


class RemoveHullLineView(View):

    def __init__(self, view_manager: ViewManager) -> None:
        super().__init__(view_manager)
        self.pick_pos: tuple[float, float] | None = None
        self.events_stack = []
        self.is_adding_points: bool = False
        self.points_to_add = []
        self.pointer_points = []
        self.remove_line_state = 0

        self.removed_lines = []

        # buttons
        view_button = ChangeViewButton(self, self.labels_ax, "Labels", ViewsEnum.LABELS)
        # view_button.highlight()
        self.vem.add(view_button)
        self.vem.add(ChangeViewButton(self, self.home_ax, "Home", ViewsEnum.HOME))
        self.vem.add(ChangeViewButton(self, self.clusters_ax, "Cluster", ViewsEnum.CLUSTER))
        self.vem.add(ChangeViewButton(self, [0.85, 0.936, 0.1, 0.06], "Back", ViewsEnum.HULLS))

        self.pointer_remove_line = NormalButton(self, [0.075, 0.05, 0.2, 0.075], "Remove line", self.remove_line)
        self.vem_pointer_remove_line = self.vem.add(self.pointer_remove_line)

        reset_b = NormalButton(self, [0.85, 0.05, 0.1, 0.075], "Reset", self.reset_removing_line)
        reset_b.button_ax.set_facecolor("lightcoral")
        reset_b.button_ref.color = "lightcoral"
        reset_b.button_ref.hovercolor = "crimson"
        # todo fix color being updated only after movement
        self.vem.add(reset_b)


        self.vem.hide()

    def draw(self, *args, **kwargs) -> None:

        super().draw()

        self.state.show_labels_and_hulls(self.vm.ax)
        self.vm.ax.set_xlim(-190, 190)
        self.vm.ax.set_ylim(-150, 150)
        self.events_stack.clear()
        self.picked_item = kwargs.get('picked_item', None)

        for child in self.vm.ax.get_children():
            if type(child) is LineCollection:
                child.remove()

        for hull_name in self.state.data['hulls_data']['hulls'].keys():
                HullArtist.hull(self.vm.ax, hull_name)
        # events
        self.cem.add(SharedEvent('pick_event', self.pick_event))
        self.cem.add(SharedEvent('button_press_event', self.press_event))
        self.vem.refresh()
        plt.draw()

    def hide(self) -> None:
        super().hide()

    def pick_event(self, event: PickEvent) -> None:
        logging.info(f"""{self.__class__} EVENT: {event} ARTIST: {event.artist} 
                     ID: {getattr(event.artist, 'id', None)}""")
        
        if isinstance(event.artist, HullArtist):
            
            if self.seq_remove_line_1 == 1:
                self.line_1_id = getattr(event.artist, 'id', None)
                print("ID_1")
                print(self.line_1_id)
                self.seq_remove_line_1 += 1

            elif self.seq_remove_line_1 == 3:
                self.line_2_id = getattr(event.artist, 'id', None)
                print("ID_2")
                print(self.line_2_id)
                self.seq_remove_line_1 = -1

            self.events_stack.append(event.artist.get_state())
            self.picked_item  = event.artist
            self.pick_pos = (event.mouseevent.xdata, event.mouseevent.ydata)

            # update fields
            self.vem.refresh()

    def press_event(self, event: MouseEvent) -> None:
        logging.info(f"""{self.__class__} POS: {event.xdata}, {event.ydata}""")

        if self.is_adding_points:
            if self.remove_line_state <= 2:
                
                self.remove_line_state += 1

                self.points_to_add.append((event.xdata, event.ydata))

                self.state._hull_set_point(len(self.points_to_add) * (-1), event.xdata, event.ydata)
                self.pointer_points.append(PointArtist.point(self.vm.ax, len(self.points_to_add) * (-1), facecolor="red", edgecolor="red"))
                
                self.vem.refresh()
                plt.draw()

            if self.remove_line_state == 3:
                self.remove_line_state = 0
                self.is_adding_points = False
                
                self.remove_line_from_hull()
    

    def remove_line(self):
        self.is_adding_points = True
        self.remove_line_state = 1

    def reset_removing_line(self):
        ...

    def search_for_hull_name(self, point, _hull_name = ""):
        hulls_name = self.state.get_all_hulls_name()
        best_dist = float("inf")
        best_hull_name = ""
        best_cord = (0,0)

        for hull_name in hulls_name:
            cords = self.state.get_hull_interpolated_cords(hull_name)
            if _hull_name != "" and _hull_name != hull_name:
                continue
            for cord in cords:
                dist = np.sqrt((cord[0] - point[0]) ** 2 + (cord[1] - point[1]) ** 2)
                if dist < 10 and dist < best_dist:
                    best_dist = dist
                    best_hull_name = hull_name
                    best_cord = cord
        
        return best_hull_name, best_cord


    def remove_line_from_hull(self):
        point1 = self.points_to_add[0]
        point2 = self.points_to_add[1]

        self.points_to_add = []

        for i, pointer_point in enumerate(self.pointer_points):
            pointer_point.remove()
            self.state._hull_remove_point(i * (-1))

        self.pointer_points = []

        hull_name_1, cord_1 = self.search_for_hull_name(point1)
        hull_name_2, cord_2 = self.search_for_hull_name(point2, hull_name_1)

        if hull_name_1 == hull_name_2:
            self.exec_remove_line_from_hull(hull_name_1, cord_1, cord_2)

        self.vem.refresh()
        plt.draw()

    def exec_remove_line_from_hull(self, hull_name, point1, point2):

        cords = self.state.get_hull_interpolated_cords(hull_name)
        lines = self.state.get_hull_lines_cords(hull_name)

        final_cords = []
        final_lines = []

        start_index = cords.index(point1)
        end_index = cords.index(point2)


        cord_start = cords[start_index]
        cord_end = cords[end_index]

        self.state.set_hole_in_hulls(hull_name, (cord_start, cord_end))

        line_start_index = -1
        line_end_index = -1

        for index, (point1, point2) in enumerate(lines):
            if ((cord_start[0] == point1[0] and cord_start[1] == point1[1]) 
                or (cord_start[0] == point2[0] and cord_start[1] == point2[1])):
                line_start_index = index
            
        for index, (point1, point2) in enumerate(lines):
            if ((cord_end[0] == point1[0] and cord_end[1] == point1[1]) 
                or (cord_end[0] == point2[0] and cord_end[1] == point2[1])):
                line_end_index = index

        if (start_index + 1) % len(cords) < (end_index - 1) % len(cords):
            final_cords = cords[:start_index] + cords[end_index+1:]
            final_lines = lines[:line_start_index] + lines[line_end_index+1:]

        else:
            final_cords = cords[end_index + 1 : start_index]
            final_lines = lines[line_end_index + 1 : line_start_index]

        hull_artist = HullArtist.get_by_id(self.vm.ax, hull_name)
        hull_artist.remove()
        self.vem.refresh()

        self.state.set_hull_interpolated_cords(hull_name, final_cords)
        self.state.set_hull_lines_cords(hull_name, final_lines)

        HullArtist.hull(self.vm.ax, hull_name)
        self.vem.refresh()
        plt.draw()


    def hide(self) -> None:
        super().hide()


# -------------------------------- MAIN EDITOR ------------------------------- #


class Editor:
    def __init__(self, state: State) -> None:
        self.state = state

        # inject state
        StateLinker.link_state(state)

    def run(self) -> None:
        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.2)

        # setup costom toolbar buttons and events
        SaveLoadPltLinker.link_ax_fig(ax, fig)
        custom_buttons_setup(self.state)


        # init event canvas
        Event.set_canvas(fig.canvas)

        vm = ViewManager(fig, ax)
        vm.register_views([Home(vm),
                           LabelsView(vm),
                           ArrowsView(vm),
                           HullView(vm),
                           ClusterMainView(vm),
                           AgglomerativeView(vm),
                           DBSCANView(vm),
                           CreateNewHullView(vm),
                           RemoveHullLineView(vm)])  # must be the same as ViewsEnum
        vm.run()

        # display
        # plt.ion()
        plt.show(block=True)
