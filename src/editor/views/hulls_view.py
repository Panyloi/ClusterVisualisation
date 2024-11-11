from matplotlib.backend_bases import PickEvent, MouseEvent
from scipy import interpolate
from ..view_manager import *
from src.generator.hull_generator import interpolate_points

class HullView(View):

    def __init__(self, view_manager: ViewManager) -> None:
        super().__init__(view_manager)
        self.mode: str = 'main'
        self.pick_pos: tuple[float, float] | None = None
        self.is_adding_points: bool = False
        self.points_to_add = []
        self.pointer_points = []
        self.remove_line_state = 0
        self.removed_lines = []

        self.vem.add(ChangeViewButton(self, self.home_ax, "Home", ViewsEnum.HOME))
        self.vem.add(ChangeViewButton(self, self.labels_ax, "Labels", ViewsEnum.LABELS))
        view_button = self.vem.add(ChangeViewButton(self, self.hulls_ax, "Hulls", ViewsEnum.HULLS))
        self.vem.add(ChangeViewButton(self, self.clusters_ax, "Cluster", ViewsEnum.CLUSTER))

        self.add_line_button: NormalButton = self.vem.add(NormalButton(self, [0.05, 0.05, 0.2, 0.075],
                                                                       "Add line", lambda: self.switch_mode('add')))
        self.remove_line_button: NormalButton = self.vem.add(NormalButton(self, [0.25, 0.05, 0.2, 0.075],
                                                                          "Remove line", lambda: self.switch_mode('remove')))
        self.confirm_button: NormalButton = self.vem.add(NormalButton(self, [0.74, 0.05, 0.1, 0.075], "Confirm", self.confirm))
        self.reset_button: NormalButton = self.vem.add(NormalButton(self, [0.85, 0.05, 0.1, 0.075], "Reset", self.reset))
        self.cancel_button: NormalButton = self.vem.add(NormalButton(self, [0.85, 0.05, 0.1, 0.075], "Cancel", self.cancel))

        view_button.highlight()
        self.reset_button.button_ax.set_facecolor("lightcoral")
        self.reset_button.button_ref.color = "lightcoral"
        self.reset_button.button_ref.hovercolor = "crimson"

        self.vem.hide()

    def draw(self, *args, **kwargs) -> None:
        super().draw()

        self.state.hide_labels_and_hulls(self.vm.ax)
        HullArtist.show_hulls(self.vm.ax)
        self.confirm_button.hide()
        self.cancel_button.hide()

        df = self.state.get_all_points()
        self.vm.ax.set_xlim(df['x'].min() - 10, df['x'].max() + 10)
        self.vm.ax.set_ylim(df['y'].min() - 10, df['y'].max() + 10)

        # remove current hulls
        for child in self.vm.ax.get_children():
            if type(child) is LineCollection:
                child.remove()

        # add new hulls
        for hull_name in self.state.data['hulls_data']['hulls'].keys():
            HullArtist.hull(self.vm.ax, hull_name)

        # events
        self.cem.add(SharedEvent('pick_event', self.pick_event))
        self.cem.add(SharedEvent('button_press_event', self.press_event))

        plt.draw()

    def switch_mode(self, mode_name):
        """ Switch mode -> toggle buttons"""
        self.mode = mode_name
        if self.mode == 'main':
            self.add_line_button.show()
            self.remove_line_button.show()
            self.reset_button.show()
            self.confirm_button.hide()
            self.cancel_button.hide()
        else:
            self.add_line_button.hide()
            self.remove_line_button.hide()
            self.reset_button.hide()
            self.confirm_button.show()
            self.cancel_button.show()
        plt.draw()

    def cancel(self):
        """ Switch to main mode and cancel every action since last switch """
        # todo code action like resetting picked points etc
        self.switch_mode('main')

    def reset(self):
        """ Reset hulls to before changes"""
        # can be used for testing
        ...

    def pick_event(self, event: PickEvent) -> None:
        logging.info(
            f"""{self.__class__} EVENT: {event} ARTIST: {event.artist} 
            ID: {getattr(event.artist, 'id', None)}"""
        )
        """ 
            Hull artist doesn't inherit after Artist, so it can't ever be picked up 
            Instead LineCollection can be used
        """

        # if isinstance(event.artist, LineCollection):
        #   pass

    def press_event(self, event: MouseEvent) -> None:
        logging.info(f"""{self.__class__} POS: {event.xdata}, {event.ydata}""")
        print(event)
        if event.inaxes != self.vm.ax:
            return

        if self.mode == "add":
            print("creating mode")
            self.points_to_add.append((event.xdata, event.ydata))
            self.state.hull_set_point(-len(self.points_to_add), event.xdata, event.ydata)
            self.pointer_points.append(
                PointArtist.point(self.vm.ax, -len(self.points_to_add), facecolor="red", edgecolor="red")
            )
        elif self.mode == "remove":
            print("removing mode")
            if self.remove_line_state <= 2:
                self.remove_line_state += 1

                self.points_to_add.append((event.xdata, event.ydata))
                self.state.hull_set_point(-len(self.points_to_add), event.xdata, event.ydata)
                self.pointer_points.append(
                    PointArtist.point(self.vm.ax, -len(self.points_to_add), facecolor="red", edgecolor="red")
                )
            elif self.remove_line_state == 3:
                self.remove_line_state = 0
                self.is_adding_points = False
                self.remove_line_from_hull()
        elif self.mode == "main":
            return

        plt.draw()

    def confirm(self):
        if self.mode == "add":
            self.calculate_new_hull()
            plt.draw()
        elif self.mode == "remove":
            self.remove_line_state = 1
        self.switch_mode("main") # todo kinda stupid fix, gotta add a separate return button

    def search_for_hull_name_in_hole(self, point: tuple[float, float], _hull_name: str = "") -> (str, tuple[float, float]):
        hulls_name = self.state.get_all_hulls_name()
        best_dist = float("inf")
        best_hull_name = ""
        best_cord = (0, 0)

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

    def find_closest_edge_point_in_hull(self, hull_name: str, point: tuple[float, float]) -> (tuple[float, float], int):
        hull_cords = self.state.get_hull_interpolated_cords(hull_name)
        best_cords = (0, 0)
        best_dist = float("inf")
        best_idx = -1

        for i, hull_cord in enumerate(hull_cords):
            dist = np.sqrt((hull_cord[0] - point[0]) ** 2 + (hull_cord[1] - point[1]) ** 2)
            if dist < best_dist:
                best_cords = hull_cord
                best_dist = dist
                best_idx = i

        return best_cords, best_idx

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

            interpolated_points = interpolate_points(self.points_to_add)
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
                self.state.hull_remove_point(i * (-1))

            self.pointer_points = []

            self.points_to_add = []

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
        interpolated_points = interpolate_points(self.points_to_add[1:-1])

        for index, (line_point_1, line_point_2) in enumerate(lines):
            if ((point_1[0] == line_point_1[0] and point_1[1] == line_point_1[1])
                    or (point_1[0] == line_point_2[0] and point_1[1] == line_point_2[1])):
                line_start_index = index

        for index, (line_point_1, line_point_2) in enumerate(lines):
            if ((point_2[0] == line_point_1[0] and point_2[1] == line_point_1[1])
                    or (point_2[0] == line_point_2[0] and point_2[1] == line_point_2[1])):
                line_end_index = index

        if idx_1 < idx_2:
            first_part_hull_points = hull_interpolated_points[:idx_1 + 1]
            second_part_hull_points = hull_interpolated_points[idx_2:]

            line_first_part_hull_points = lines[:line_start_index + 1]
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

        self.state.remove_hole_in_hulls(hull_name, hole_cord)

        self.state.set_hull_interpolated_cords(hull_name, final_cords)
        self.state.set_hull_lines_cords(hull_name, final_lines)

        HullArtist.hull(self.vm.ax, hull_name)

        for i, pointer_point in enumerate(self.pointer_points):
            pointer_point.remove()
            self.state.hull_remove_point(i * (-1))

        self.pointer_points = []

        self.points_to_add = []

        plt.draw()

    def remove_line(self):
        self.is_adding_points = True
        self.remove_line_state = 1

    def reset_removing_line(self):
        ...

    def search_for_hull_name(self, point, _hull_name=""):
        hulls_name = self.state.get_all_hulls_name()
        best_dist = float("inf")
        best_hull_name = ""
        best_cord = (0, 0)

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
            self.state.hull_remove_point(i * (-1))

        self.pointer_points = []

        hull_name_1, cord_1 = self.search_for_hull_name(point1)
        hull_name_2, cord_2 = self.search_for_hull_name(point2, hull_name_1)

        if hull_name_1 == hull_name_2:
            self.exec_remove_line_from_hull(hull_name_1, cord_1, cord_2)

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
            final_cords = cords[:start_index] + cords[end_index + 1:]
            final_lines = lines[:line_start_index] + lines[line_end_index + 1:]

        else:
            final_cords = cords[end_index + 1: start_index]
            final_lines = lines[line_end_index + 1: line_start_index]

        hull_artist = HullArtist.get_by_id(self.vm.ax, hull_name)
        hull_artist.remove()

        self.state.set_hull_interpolated_cords(hull_name, final_cords)
        self.state.set_hull_lines_cords(hull_name, final_lines)

        HullArtist.hull(self.vm.ax, hull_name)
        plt.draw()

    def hide(self) -> None:
        super().hide()
        self.vm.ax.set_xlim(-190, 190)
        self.vm.ax.set_ylim(-150, 150)

    @staticmethod
    def distance(point_1: tuple[float, float], point_2: tuple[float, float]) -> float:
        return np.sqrt((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2)
