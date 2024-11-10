from matplotlib.backend_bases import PickEvent, MouseEvent, KeyEvent, ResizeEvent

from ..view_manager import *

class LabelsView(View):

    def __init__(self, view_manager: ViewManager) -> None:
        super().__init__(view_manager)
        self.dragged_item: LabelArtist | None = None
        self.picked_item: LabelArtist | None = None
        self.pick_pos: tuple[float, float] | None = None
        self.events_stack = []

        # buttons
        self.vem.add(ChangeViewButton(self, self.home_ax, "Home", ViewsEnum.HOME))
        self.vem.add(ChangeViewButton(self, self.clusters_ax, "Cluster", ViewsEnum.CLUSTER))
        self.vem.add(ChangeViewButton(self, self.hulls_ax, "Hulls", ViewsEnum.HULLS))
        view_button = self.vem.add(ChangeViewButton(self, self.labels_ax, "Labels", ViewsEnum.LABELS))
        self.vem.add(NormalButton(self, [0.075, 0.05, 0.05, 0.075], "+", self.add_label))
        self.vem.add(NormalButton(self, [0.125, 0.05, 0.05, 0.075], "-", self.delete_label))
        self.vem.add(NormalButton(self, [0.575, 0.0875, 0.05, 0.0375], "^", self.font_size_up))
        self.vem.add(NormalButton(self, [0.575, 0.05, 0.05, 0.0375], "v", self.font_size_down))
        self.vem.add(NormalButton(self, [0.75, 0.0875, 0.05, 0.0375], "^", self.arrow_size_up))
        self.vem.add(NormalButton(self, [0.75, 0.05, 0.05, 0.0375], "v", self.arrow_size_down))
        self.vem.add(NormalButton(self, [0.825, 0.05, 0.10, 0.075], "+arrow", self.add_arrow))

        view_button.highlight()

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
        logging.info(
            f"""{self.__class__} EVENT: {event} ARTIST: {event.artist} 
            ID: {getattr(event.artist, 'id', None)}"""
        )
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
                    logging.info(f"""{self.__class__} EVENT: {event} canceled due to overlapping Label: {artist}""")
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
        self.vem.add(ChangeViewButton(self, self.home_ax, "Home", ViewsEnum.HOME))
        self.vem.add(ChangeViewButton(self, self.clusters_ax, "Cluster", ViewsEnum.CLUSTER))
        self.vem.add(ChangeViewButton(self, self.hulls_ax, "Hulls", ViewsEnum.HULLS))
        view_button = self.vem.add(ChangeViewButton(self, self.labels_ax, "Labels", ViewsEnum.LABELS))
        self.vem.add(NormalButton(self, [0.8, 0.05, 0.1, 0.075], "Delete", self.delete_arrow))
        self.vem.add(BlockingButton(self, [0.35, 0.05, 0.05, 0.075], "p", self.sh_point_picker))
        self.vem.add(BlockingButton(self, [0.7, 0.05, 0.05, 0.075], "p", self.rf_point_picker))

        view_button.highlight()

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
