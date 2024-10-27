import matplotlib.pyplot as plt
from matplotlib.backend_bases import PickEvent, MouseEvent, KeyEvent, ResizeEvent
from sklearn.cluster import AgglomerativeClustering, DBSCAN

from .view_manager import *
from ..generator.hull_generator import calc_hull
from ..generator.labels_generator import calc, parse_solution_to_editor


class Home(View):

    def __init__(self, view_manager: ViewManager) -> None:
        super().__init__(view_manager)
        self.vem.add(ChangeViewButton(self, [0.05, 0.05, 0.1, 0.075], "Home", ViewsEnum.HOME))
        self.vem.add(ChangeViewButton(self, [0.15, 0.05, 0.1, 0.075], "Labels", ViewsEnum.LABELS))
        self.vem.add(ChangeViewButton(self, [0.25, 0.05, 0.1, 0.075], "Cluster", ViewsEnum.CLUSTER))
        self.vem.add(ChangeViewButton(self, [0.35, 0.05, 0.1, 0.075], "Hulls", ViewsEnum.HULLS))

    def draw(self, *args, **kwargs) -> None:
        super().draw()

        # connect auto refreshing
        self.vem.refresh_connect(self.vm.fig)

        plt.draw()

    def undraw(self) -> None:
        super().undraw()


class LabelsView(View):

    def __init__(self, view_manager: ViewManager) -> None:
        super().__init__(view_manager)
        self.dragged_item: LabelArtist | None = None
        self.picked_item: LabelArtist | None = None
        self.pick_pos: tuple[float, float] | None = None
        self.events_stack = []

        # buttons
        self.vem.add(ChangeViewButton(self, [0.025, 0.05, 0.1, 0.075], "Home", ViewsEnum.HOME))
        self.vem.add(NormalButton(self, [0.125, 0.05, 0.05, 0.075], "+", self.add_label))
        self.vem.add(NormalButton(self, [0.175, 0.05, 0.05, 0.075], "-", self.delete_label))
        self.vem.add(NormalButton(self, [0.55, 0.05, 0.10, 0.075], "+arrow", self.add_arrow))
        self.vem.add(NormalButton(self, [0.775, 0.0875, 0.05, 0.0375], "^", self.font_size_up))
        self.vem.add(NormalButton(self, [0.775, 0.05, 0.05, 0.0375], "v", self.font_size_down))
        self.vem.add(NormalButton(self, [0.925, 0.0875, 0.05, 0.0375], "^", self.arrow_size_up))
        self.vem.add(NormalButton(self, [0.925, 0.05, 0.05, 0.0375], "v", self.arrow_size_down))

        # displays
        self.vem.add(ShiftingTextBox(self, [0.275, 0.05, 0.25, 0.075],
                                     self.label_name_update, self.label_name_submit))
        self.vem.add(LimitedTextBox(self, [0.675, 0.05, 0.10, 0.075],
                                    self.font_size_update, self.font_size_submit))
        self.vem.add(LimitedTextBox(self, [0.825, 0.05, 0.10, 0.075],
                                    self.arrow_size_update, self.arrow_size_submit))

        self.vem.hide()

    def draw(self, *args, **kwargs) -> None:
        super().draw()
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

    def undraw(self) -> None:
        super().undraw()

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
        self.vem.add(ChangeViewButton(self, [0.05, 0.05, 0.1, 0.075], "Home", ViewsEnum.HOME))
        self.vem.add(NormalButton(self, [0.15, 0.05, 0.05, 0.075], "-", self.delete_arrow))
        self.vem.add(BlockingButton(self, [0.50, 0.05, 0.05, 0.075], "p", self.sh_point_picker))
        self.vem.add(BlockingButton(self, [0.80, 0.05, 0.05, 0.075], "p", self.rf_point_picker))

        # displays
        self.vem.add(LimitedTextBox(self, [0.30, 0.05, 0.10, 0.075],
                                    self.arrow_shx_update, self.arrow_shx_submit, "Att:"))
        self.vem.add(LimitedTextBox(self, [0.40, 0.05, 0.10, 0.075],
                                    self.arrow_shy_update, self.arrow_shy_submit))
        self.vem.add(LimitedTextBox(self, [0.60, 0.05, 0.10, 0.075],
                                    self.arrow_rfx_update, self.arrow_rfx_submit, "Ref:"))
        self.vem.add(LimitedTextBox(self, [0.70, 0.05, 0.10, 0.075],
                                    self.arrow_rfy_update, self.arrow_rfy_submit))

        self.vem.hide()

    def draw(self, *args, **kwargs) -> None:
        super().draw()

        # get picked arrow if exists
        self.picked_item = kwargs.get('picked_item', None)

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


class ClusterView(View):
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

    def __init__(self, view_manager: ViewManager) -> None:
        super().__init__(view_manager)
        self.picked_item: PointArtist | None = None
        self.pick_pos: tuple[float, float] | None = None
        self.events_stack = []
        self.info_text = None

        # buttons
        self.vem.add(ChangeViewButton(self, [0.1, 0.05, 0.1, 0.075], "Home", ViewsEnum.HOME))
        self.vem.add(ChangeViewButton(self, [0.2, 0.05, 0.1, 0.075], "Agglo", ViewsEnum.AGGLOMERATIVE))
        self.vem.add(ChangeViewButton(self, [0.3, 0.05, 0.1, 0.075], "DBSCAN", ViewsEnum.DBSCAN))
        self.vem.add(NormalButton(self, [0.4, 0.05, 0.1, 0.075], "Remove", self.remove_point))
        self.vem.add(NormalButton(self, [0.56, 0.05, 0.12, 0.075], "Hulls", self.draw_hull))
        self.vem.add(NormalButton(self, [0.68, 0.05, 0.12, 0.075], "Labels", self.draw_labels))
        self.vem.add(NormalButton(self, [0.8, 0.05, 0.1, 0.075], "Reset", self.reset))

        self.vem.hide()

    def draw(self, *args, **kwargs) -> None:
        super().draw()

        # self.events_stack.clear()
        # event = kwargs.get('event', None)
        # self.picked_item = kwargs.get('picked_item', None)
        # if event is not None:
        #     self.pick_event(event)

        self.info_text = Text(0, 0, "Info")

        # events
        self.cem.add(SharedEvent('pick_event', self.pick_event))

        self.draw_points()
        self.info_text.set_position((self.vm.ax.get_xlim()[0] + 3, self.vm.ax.get_ylim()[1] - 10))
        self.vm.ax.add_artist(self.info_text)

        self.vem.refresh()
        plt.draw()

    def draw_points(self):
        self.vm.ax.clear()
        for artist in self.state.data['clusters_data']['artists']:
            self.vm.ax.add_artist(artist)
            artist.set_alpha(0.3)

        df = self.state.get_all_points()
        self.vm.ax.set_xlim(df['x'].min()-10, df['x'].max()+10)
        self.vm.ax.set_ylim(df['y'].min()-10, df['y'].max()+10)

        # self.vm.ax.relim()
        # self.vm.ax.autoscale_view()
        # autoscale doesn't work when labels are out of axis
        # relim fixes that but takes a lot of time, setting lims manually seems to work for now

    def undraw(self) -> None:
        super().undraw()
        self.reset_pick_event()
        self.info_text.remove()

    def reset(self):
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

        # logging.info(f"""{self.__class__} EVENT: {event} ARTIST: {event.artist} ID: {event.artist.id}""")

        self.picked_item = PointArtist.point(self.vm.ax, event.artist.id, color="red")
        point = self.state.get_point(self.picked_item.id)
        for point_id in self.state.get_cluster(point['type']).index:
            artist = self.state.data['clusters_data']['artists'][point_id]
            artist.set_alpha(1)
            artist.set_edgecolor("black")

        self.info_text.set_text(point['type'])

        plt.draw()

    def draw_hull(self):
        # hulls = calc_hull(self.state.get_normalised_clusters(), 2, 10, 20)
        # for i in hulls.keys():
        #     self.state.data['hulls_data'][i] = {
        #         'name': hulls[i]['name'],
        #         'cords': hulls[i]['polygon_points'],
        #         'line_cords': hulls[i]['polygon_lines']
        #     }

        self.state.update_hulls()
        
        for hull_name in self.state.data['hulls_data']['hulls'].keys():
            HullArtist.hull(self.vm.ax, hull_name)
        plt.draw()

    def draw_labels(self):
        labels = calc(self.state.get_normalised_clusters(), self.state.get_all_points()[['x', 'y']].values, 10, 2)
        self.state.data = parse_solution_to_editor(labels, self.state.data)

        for label_id in self.state.data['labels_data'].keys():
            if isinstance(label_id, int):
                LabelArtist.text(self.vm.ax, label_id)
        plt.draw()



class DBSCANView(View):
    def __init__(self, view_manager: ViewManager) -> None:
        super().__init__(view_manager)
        self.widget_cluster_name = None
        self.widget_scalar = None
        self.args = {"cluster_name": None, "scalar": None}
        self.current = {"cluster": None, "labels": None}
        self.info_text = None

        self.vem.add(ChangeViewButton(self, [0.1, 0.05, 0.1, 0.075], "Home", ViewsEnum.HOME))
        self.vem.add(ChangeViewButton(self, [0.2, 0.05, 0.1, 0.075], "Back", ViewsEnum.CLUSTER))
        self.vem.add(NormalButton(self, [0.65, 0.05, 0.15, 0.075], "Save cluster", self.save_cluster))
        self.vem.add(NormalButton(self, [0.8, 0.05, 0.1, 0.075], "Save all", self.reset))

        self.vem.hide()

    def draw(self, *args, **kwargs) -> None:
        super().draw()

        plt.subplots_adjust(bottom=0.3, left=0.4, top=0.9, right=0.9)

        self.info_text = Text(self.vm.ax.get_xlim()[0] + 3, self.vm.ax.get_ylim()[1] - 10, "Info")
        self.vm.ax.add_artist(self.info_text)

        self.widget_cluster_name = ViewRadioButtons(
            self,
            [0.05, 0.15, 0.3, 0.75],
            sorted(list(self.state.get_all_clusters().keys())),
            self._update_args,
        )

        self.widget_scalar = ViewSlider(
            self, [0.55, 0.17, 0.3, 0.05], "", 0.01, 2.5, self._update_args
        )

        self.args["cluster_name"] = self.widget_cluster_name.ref.value_selected
        self.args["scalar"] = self.widget_scalar.ref.val

        # adding and removing manually todo add remove to vem
        # self.vem.add(self.widget_cluster_name)
        self.vem.add(self.widget_scalar)

        self.draw_cluster()
        plt.draw()

    def reset_cluster(self):
        for point_id in self.state.get_cluster(self.args["cluster_name"]).index:
            artist = self.state.data['clusters_data']['artists'][point_id]
            artist.set_color(self.state.get_point_color(point_id))
            artist.set_alpha(0.3)
            artist.set_radius(1.5)
            artist.set_zorder(1)

    def draw_cluster(self):
        self.current["cluster"] = self.state.get_cluster(self.args["cluster_name"])
        X = self.current["cluster"][['x', 'y']].values
        if self.current["cluster"].shape[0] > 1:
            clustering = DBSCAN(eps=self.args["scalar"]*10, min_samples=2).fit(X)
            self.current["labels"] = clustering.labels_
            colors = mpl.colormaps["tab10"](self.current["labels"])
            colors = ["black" if self.current["labels"][idx] == -1 else x for idx, x in enumerate(colors)]
        else:
            colors = self.current["labels"] = ["black"]

        idx = 0
        for point_id in self.state.get_cluster(self.args["cluster_name"]).index:
            artist = self.state.data['clusters_data']['artists'][point_id]
            artist.set_color(colors[idx])
            artist.set_alpha(1)
            artist.set_radius(2.5)
            artist.set_zorder(10)
            idx += 1

    def _update_args(self, widget_value):
        self.reset_cluster()
        self.info_text.set_text("Info")
        self.args["cluster_name"] = self.widget_cluster_name.ref.value_selected
        self.args["scalar"] = self.widget_scalar.ref.val
        self.draw_cluster()
        plt.draw()

    def undraw(self) -> None:
        super().undraw()
        self.reset_cluster()
        self.widget_cluster_name.remove()
        self.info_text.remove()
        plt.subplots_adjust(bottom=0.15, left=0.01, right=0.99, top=0.99)

    def recreate_cluster_widget(self):
        self.widget_cluster_name.remove()
        self.widget_cluster_name = ViewRadioButtons(
            self,
            [0.05, 0.15, 0.3, 0.75],
            sorted(list(self.state.get_all_clusters().keys())),
            self._update_args,
        )

    def save_cluster(self):
        self.reset_cluster()

        label_map = {key: [] for key in set(self.current["labels"])}

        for idx, point_id in enumerate(self.current["cluster"].index):
            label_map[self.current["labels"][idx]].append(point_id)

        for key, points in label_map.items():
            if key == -1: continue
            self.state.set_cluster(self.args["cluster_name"] + str(key), points)
            self.state.set_hull_to_undraw(self.args["cluster_name"])
            self.state.set_hull_to_change(self.args["cluster_name"] + str(key), self.state.get_cluster(self.args["cluster_name"] + str(key)))


            for point_id in points:
                artist = self.state.data['clusters_data']['artists'][point_id]
                artist.set_color(self.state.get_point_color(point_id))

        self.recreate_cluster_widget()
        self.info_text.set_text("Saved")
        plt.draw()

        print(self.state.get_all_clusters())

    #todo I'm using this func for testing, gotta change it back later
    def reset(self):
        print(self.state.get_normalised_clusters())
        # self.state.set_clusters_empty()
        # self.draw_cluster()


class AgglomerativeView(View):
    def __init__(self, view_manager: ViewManager) -> None:
        super().__init__(view_manager)
        self.type = None
        self.linkage = None
        self.scalar = None
        self.widget_type = None
        self.widget_linkage = None
        self.widget_scalar = None
        self.cmap = plt.cm.get_cmap("hsv", len(self.state.get_raw()['data'].keys()))
        self.removed = {"x": [], "y": []}
        self.current_cluster = None
        self.current_labels = None
        self.save_index = 0

        self.vem.add(ChangeViewButton(self, [0.1, 0.05, 0.1, 0.075],
                                      "Home", ViewsEnum.HOME))
        self.vem.add(ChangeViewButton(self, [0.2, 0.05, 0.1, 0.075],
                                      "Back", ViewsEnum.CLUSTER))
        self.vem.add(NormalButton(self, [0.65, 0.05, 0.15, 0.075],
                                  "Save cluster", self.save_cluster))

        self.vem.hide()

    def draw(self, *args, **kwargs) -> None:
        super().draw()

        self.widget_type = ViewRadioButtons(self, [0.05, 0.15, 0.3, 0.75],
                                            sorted(list(self.state.get_raw()['data'].keys())),
                                            self._update_args)
        self.widget_linkage = ViewRadioButtons(self, [0.4, 0.15, 0.1, 0.1],
                                               ["ward", "complete", "average", "single"],
                                               self._update_args, 3)
        self.widget_scalar = ViewSlider(self, [0.55, 0.17, 0.3, 0.05], "", 0.01, 2.5,
                                        self._update_args)

        self.type = self.widget_type.ref.value_selected
        self.linkage = self.widget_linkage.ref.value_selected
        self.scalar = self.widget_scalar.ref.val

        self.vem.add(self.widget_type)
        self.vem.add(self.widget_linkage)
        self.vem.add(self.widget_scalar)

        plt.subplots_adjust(bottom=0.3, left=0.4, top=0.9, right=0.9)

        # self.vem.add(NormalButton(self, [0.44, 0.05, 0.17, 0.075],
        #                           "Remove points", self.remove_points))
        # self.vem.add(NormalButton(self, [0.8, 0.05, 0.1, 0.075],
        #                           "Reset", self.reset))

        self.draw_cluster()

    def undraw(self) -> None:
        super().undraw()
        plt.subplots_adjust(bottom=0.15, left=0.01, right=0.99, top=0.99)

    def _update_args(self, sth):
        self.type = self.widget_type.ref.value_selected
        self.linkage = self.widget_linkage.ref.value_selected
        self.scalar = self.widget_scalar.ref.val
        self.draw_cluster()

    def draw_cluster(self):
        if self.type is None or self.linkage is None or self.scalar is None: return
        self.current_cluster = self.state.get_raw()['data'][self.type]
        sth = np.column_stack([self.current_cluster["x"], self.current_cluster["y"]])
        if len(self.current_cluster["x"]) > 1:
            dist = AgglomerativeClustering(n_clusters=1, compute_distances=True).fit(sth).distances_
            mean = np.mean(dist)
            clustering = AgglomerativeClustering(n_clusters=None,
                                                 linkage=self.linkage,
                                                 distance_threshold=mean * self.scalar).fit(sth)
            self.current_labels = clustering.labels_
        else:
            self.current_labels = 'k'

        self.vm.ax.clear()
        for point_id in range(len(self.state.get_all_points())):
            PointArtist.point(self.vm.ax, point_id, alpha=0.3)
        self.vm.ax.plot()

        self.vm.ax.scatter(self.current_cluster["x"], self.current_cluster["y"], c=self.current_labels)
        plt.draw()

    def save_cluster(self):
        tuples = list(zip(self.current_cluster["x"], self.current_cluster["y"]))
        label_map = {key: [] for key in set(self.current_labels)}
        for point_id in range(len(self.state.get_all_points())):
            point = self.state.get_point_pos(point_id)
            if point in tuples:
                idx = tuples.index(point)
                label_map[self.current_labels[idx]].append(point_id)
        for points in label_map.values():
            self.state.set_cluster(str(self.save_index), points)
            self.state.set_hull_to_undraw(self.type)
            self.state.set_hull_to_change(str(self.save_index), self.state.get_cluster(str(self.save_index)))

            self.save_index += 1
        print(self.state.get_all_clusters())

    def reset(self):
        self.state.set_clusters_empty()

    # def remove_points(self):
    #     if self.type in self.clusters.keys():
    #         cluster = self.state.get_raw()['data'][self.type]
    #         sth = np.column_stack([cluster["x"], cluster["y"]])
    #         dist = AgglomerativeClustering(n_clusters=1, compute_distances=True).fit(sth).distances_
    #         mean = np.mean(dist)
    #         clustering = AgglomerativeClustering(n_clusters=None, linkage=self.linkage,
    #                                              distance_threshold=mean * self.scalar).fit(sth)
    #         for i in range(len(clustering.labels_)):
    #             if clustering.labels_[i] != 0:
    #                 self.removed["x"].append(cluster["x"][i])
    #                 self.removed["y"].append(cluster["y"][i])
    #         print(self.removed)


class HullView(View):

    def __init__(self, view_manager: ViewManager) -> None:
        super().__init__(view_manager)
        self.dragged_item: HullArtist | None = None
        self.picked_item: HullArtist | None = None
        self.events_stack = []

        self.vem.add(ChangeViewButton(self, [0.05, 0.05, 0.1, 0.075], "Home", ViewsEnum.HOME))
        self.vem.add(NormalButton(self, [0.15, 0.05, 0.15, 0.075], "Remove Line", self.remove_line))
        self.vem.add(NormalButton(self, [0.30, 0.05, 0.15, 0.075], "Remove Hull", self.remove_hull))
        self.vem.hide()

    def draw(self, *args, **kwargs) -> None:
        super().draw()

        self.events_stack.clear()

        self.picked_item = kwargs.get('picked_item', None)


        # events
        self.cem.add(SharedEvent('pick_event', self.pick_event))

        self.vem.refresh()
        plt.draw()

    def pick_event(self, event: PickEvent) -> None:
        logging.info(f"""{self.__class__} EVENT: {event} ARTIST: {event.artist} 
                     ID: {getattr(event.artist, 'id', None)}""")
        if isinstance(event.artist, HullArtist):
            self.events_stack.append(event.artist.get_state())
            self.picked_item  = event.artist
            self.pick_pos = (event.mouseevent.xdata, event.mouseevent.ydata)

            # update fields
            self.vem.refresh()

    def remove_line(self) -> None:
        ...
    
    def remove_hull(self) -> None:
        if self.picked_item is None:
            return
        
        self.pick_event.remove()
        self.picked_item = None
        self.vem.refresh()

    def undraw(self) -> None:
        super().undraw()

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
                           ClusterView(vm),
                           AgglomerativeView(vm),
                           DBSCANView(vm)])  # must be the same as ViewsEnum
        vm.run()

        # dispalay
        # plt.ion()
        plt.show(block=True)
