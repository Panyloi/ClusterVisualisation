from matplotlib.backend_bases import PickEvent
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from ..view_manager import *


class ClusterMainView(View):
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

    def __init__(self, view_manager: ViewManager) -> None:
        super().__init__(view_manager)
        self.picked_item: PointArtist | None = None
        self.pick_pos: tuple[float, float] | None = None
        self.events_stack = []
        self.hulls_off = True

        # buttons
        self.vem.add(ChangeViewButton(self, self.home_ax, "Home", ViewsEnum.HOME))
        view_button = self.vem.add(ChangeViewButton(self, self.clusters_ax, "Cluster", ViewsEnum.CLUSTER))
        self.vem.add(ChangeViewButton(self, self.labels_ax, "Labels", ViewsEnum.LABELS))
        self.vem.add(ChangeViewButton(self, self.hulls_ax, "Hulls", ViewsEnum.HULLS))
        self.vem.add(ChangeViewButton(self, [0.75, self.change_button_y, self.change_button_length, self.change_button_height], "Agglo", ViewsEnum.AGGLOMERATIVE))
        self.vem.add(ChangeViewButton(self, [0.85, self.change_button_y, self.change_button_length, self.change_button_height], "DBSCAN", ViewsEnum.DBSCAN))
        self.vem.add(NormalButton(self, [0.05, 0.05, 0.1, 0.075], "Remove", self.remove_point))
        self.vem.add(NormalButton(self, [0.17, 0.05, 0.15, 0.075], "Toggle hulls", self.draw_hull))
        reset_b = self.vem.add(NormalButton(self, [0.85, 0.05, 0.1, 0.075], "Reset", self.reset_clusters))
        self.info = self.vem.add(ViewText(self.vm.ax, 0, 0, "Info"))

        view_button.highlight()

        reset_b.button_ax.set_facecolor("lightcoral")
        reset_b.button_ref.color = "lightcoral"
        reset_b.button_ref.hovercolor = "crimson"

        self.vem.hide()

    def draw(self, *args, **kwargs) -> None:
        super().draw()

        # clear ax by hiding elements
        self.state.hide_labels_and_hulls(self.vm.ax)
        self.hulls_off = True
        self.vm.list_manager.clusters_view_hull_off = True
        self.vm.list_manager.hide_button()

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
        self.info.text_ref.set_position((self.vm.ax.get_xlim()[0] + 3, self.vm.ax.get_ylim()[1] - 10))

        plt.draw()

    def hide(self) -> None:
        super().hide()
        self.reset_pick_event()
        for artist in self.state.data['clusters_data']['artists']:
            artist.set_alpha(1)
        self.vm.ax.set_xlim(-190, 190)
        self.vm.ax.set_ylim(-150, 150)
        self.vm.list_manager.clusters_view_hull_off = False
        self.vm.list_manager.show_button()

    def reset_clusters(self): #todo really ugly, gotta clean up later
        self.state.reset_clusters()
        for artist in self.state.data['clusters_data']['artists']:
            artist.set_color(self.state.get_point_color(artist.id))

        HullArtist.remove_hulls(self.vm.ax)
        self.state.update_hulls()

        self.vm.list_manager.check_list.update(sorted(list(self.state.get_all_clusters().keys())))
        selected_hulls = self.vm.list_manager.get_only_active()
        for hull_name in self.state.get_normalised_clusters():
            artist = HullArtist.hull(self.vm.ax, hull_name)
            if self.hulls_off:
                artist.hide()
            elif artist.id not in selected_hulls:
                artist.hide()
        if self.hulls_off:
            self.vm.list_manager.check_list.hide()
        # todo doesnt work cause dunno how to use Hull Artist
        # the problem is that I remove only LineCollections, but when using algorithm I use
        # self.state.set_hull_to_undraw(cluster_name)
        # self.state.set_hull_to_change(new_cluster_name, self.state.get_cluster(new_cluster_name))
        # which presumably smites the original hulls from existance, after which HullArtist.hull doesnt work

        # what I would want is to be able to remove all hull artist when reseting and recalulate them again
        # unless it takes too much time ofc

    def remove_point(self):
        if self.picked_item:
            self.reset_pick_event()
            point = self.state.get_point(self.picked_item.id)
            point_hull = point['type']
            self.state.set_hull_to_undraw(point_hull)

            self.state.set_cluster("Removed", [self.picked_item.id])
            self.info.text_ref.set_text("Removed")

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

        self.info.text_ref.set_text(point['type'])
        plt.draw()

    def draw_hull(self):
        if self.hulls_off:
            self.hulls_off = False
            HullArtist.remove_hulls(self.vm.ax)
            self.state.update_hulls()
            self.vm.list_manager.show_button()

            self.vm.list_manager.check_list.update(sorted(list(self.state.get_all_clusters().keys())))
            selected_hulls = self.vm.list_manager.get_only_active()
            for hull_name in self.state.data['hulls_data']['hulls'].keys():
                artist: HullArtist = HullArtist.hull(self.vm.ax, hull_name)
                if artist.id not in selected_hulls:
                    artist.hide()

        else:
            self.hulls_off = True
            HullArtist.hide_hulls(self.vm.ax)
            self.vm.list_manager.hide_button()
        plt.draw()
        self.vm.list_manager.clusters_view_hull_off = self.hulls_off


class ClusteringSubViewBase(View):
    @abstractmethod
    def __init__(self, view_manager: ViewManager) -> None:
        """Adds buttons and widgets common to all sub views"""
        super().__init__(view_manager)
        self.previous_cluster_name = None
        self.current_labels = None
        self.widget_cluster_name = None

        self.vem.add(ChangeViewButton(self, self.home_ax, "Home", ViewsEnum.HOME))
        self.vem.add(ChangeViewButton(self, self.labels_ax, "Labels", ViewsEnum.LABELS))
        self.vem.add(ChangeViewButton(self, self.clusters_ax, "Cluster", ViewsEnum.CLUSTER))
        self.vem.add(ChangeViewButton(self, self.hulls_ax, "Hulls", ViewsEnum.HULLS))
        self.vem.add(ChangeViewButton(self, [0.85, 0.936, 0.1, 0.06], "Back", ViewsEnum.CLUSTER))
        self.vem.add(NormalButton(self, [0.05, 0.05, 0.15, 0.075], "Save cluster", self.save_cluster))
        reset_b = self.vem.add(NormalButton(self, [0.85, 0.05, 0.1, 0.075], "Reset", self.reset))
        self.info = self.vem.add(ViewText(self.vm.ax, 0, 0, ""))

        reset_b.button_ax.set_facecolor("lightcoral")
        reset_b.button_ref.color = "lightcoral"
        reset_b.button_ref.hovercolor = "crimson"
        # In concrete class: add to vem the view specific elements and call vem.hide()

    @abstractmethod
    def count_clustering(self, points_xy):
        """Return the fit results of a clustering algorithm"""
        NotImplementedError()

    def draw(self, *args, **kwargs) -> None:
        super().draw()

        self.widget_cluster_name = self.vem.add(
            ViewRadioButtons(
                self,
                [0.05, 0.15, 0.3, 0.75],
                sorted(list(self.state.get_all_clusters().keys())),
                self.update_plot
            )
        )

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
        self.info.text_ref.set_position((self.vm.ax.get_xlim()[0] + 3, self.vm.ax.get_ylim()[1] - 10))

        self.update_plot(None)

    def hide(self) -> None:
        super().hide()
        self.dehighlight_previous_cluster()
        plt.subplots_adjust(bottom=0.15, left=0.01, right=0.99, top=0.935)
        for artist in self.state.data['clusters_data']['artists']:
            artist.set_alpha(1)
        self.vm.ax.set_xlim(-190, 190)
        self.vm.ax.set_ylim(-150, 150)
        self.vem.remove(self.widget_cluster_name)
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
            self.current_labels = clustering.labels_
        else:
            colors = ["black"]
            self.current_labels = [0]

        # update visuals
        self.info.text_ref.set_text("")
        self.dehighlight_previous_cluster()
        self.highlight_current_cluster(cluster_name, colors)
        plt.draw()

        self.previous_cluster_name = cluster_name

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

        self.vem.remove(self.widget_cluster_name)
        self.widget_cluster_name = self.vem.add(
            ViewRadioButtons(
                self,
                [0.05, 0.15, 0.3, 0.75],
                sorted(list(self.state.get_all_clusters().keys())),
                self.update_plot
            )
        )

        self.info.text_ref.set_text("Saved")
        plt.draw()

    def reset(self):
        """Resets clusters back to the original state"""
        self.state.reset_clusters()
        self.vem.remove(self.widget_cluster_name)
        self.widget_cluster_name = self.vem.add(
            ViewRadioButtons(
                self,
                [0.05, 0.15, 0.3, 0.75],
                sorted(list(self.state.get_all_clusters().keys())),
                self.update_plot
            )
        )
        self.update_plot(None)


class DBSCANView(ClusteringSubViewBase):
    def __init__(self, view_manager: ViewManager) -> None:
        super().__init__(view_manager)

        self.widget_scalar = self.vem.add(
            ViewSlider(self, [0.55, 0.17, 0.3, 0.05], "", 0.01, 2.5, self.update_plot)
        )
        self.vem.hide()

    def count_clustering(self, points_xy):
        return DBSCAN(eps=self.widget_scalar.ref.val*10, min_samples=2).fit(points_xy)

class AgglomerativeView(ClusteringSubViewBase):
    def __init__(self, view_manager: ViewManager) -> None:
        super().__init__(view_manager)
        self.widget_linkage = self.vem.add(
            ViewRadioButtons(
                self,
                [0.4, 0.12, 0.1, 0.16],
                ["ward", "complete", "average", "single"],
                self.update_plot,
                3
            )
        )

        self.widget_scalar = self.vem.add(
            ViewSlider(self, [0.55, 0.17, 0.3, 0.05],"", 0.01, 2.5, self.update_plot)
        )

        self.vem.hide()

    def count_clustering(self, points_xy):
        dist = AgglomerativeClustering(n_clusters=1, compute_distances=True).fit(points_xy).distances_
        mean = np.mean(dist)
        return AgglomerativeClustering(
                n_clusters=None,
                linkage=self.widget_linkage.ref.value_selected,
                distance_threshold=mean * self.widget_scalar.ref.val
            ).fit(points_xy)