from matplotlib.backend_bases import PickEvent, MouseEvent
from scipy import interpolate
from ..view_manager import *

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

        self.vem.add(ChangeViewButton(self, self.home_ax, "Home", ViewsEnum.HOME))
        self.vem.add(ChangeViewButton(self, self.labels_ax, "Labels", ViewsEnum.LABELS))
        self.vem.add(ChangeViewButton(self, self.clusters_ax, "Cluster", ViewsEnum.CLUSTER))
        self.vem.add(ChangeViewButton(self, [0.3, 0.05, 0.15, 0.075], "Create hulls", ViewsEnum.CREATEHULL))
        self.vem.add(ChangeViewButton(self, [0.50, 0.05, 0.15, 0.075], "Remove line", ViewsEnum.REMOVELINE))

        view_button = ChangeViewButton(self, self.hulls_ax, "Hulls", ViewsEnum.HULLS)
        view_button.highlight()
        self.vem.add(view_button)

        # self.vem.add(NormalButton(self, [0.3, 0.05, 0.15, 0.075], "Remove Line", self.remove_line))
        # self.vem.add(NormalButton(self, [0.50, 0.05, 0.15, 0.075], "Remove Hull", self.remove_hull))
        self.vem.hide()

    def draw(self, *args, **kwargs) -> None:
        super().draw()

        # HullArtist.hide_hulls(self.vm.ax)
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

        self.vem.refresh()
        plt.draw()

    def pick_event(self, event: PickEvent) -> None:
        logging.info(
            f"""{self.__class__} EVENT: {event} ARTIST: {event.artist} 
            ID: {getattr(event.artist, 'id', None)}"""
        )

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
            self.picked_item = event.artist
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
        self.vem.add(ChangeViewButton(self, self.home_ax, "Home", ViewsEnum.HOME))
        self.vem.add(ChangeViewButton(self, self.clusters_ax, "Cluster", ViewsEnum.CLUSTER))
        self.vem.add(ChangeViewButton(self, self.hulls_ax, "Hulls", ViewsEnum.HULLS))
        self.vem.add(ChangeViewButton(self, self.labels_ax, "Labels", ViewsEnum.LABELS))
        self.vem.add(ChangeViewButton(self, [0.85, 0.936, 0.1, 0.06], "Back", ViewsEnum.HULLS))

        self.pointer_end_add_points = NormalButton(self, [0.075, 0.05, 0.2, 0.075], "End adding", self.end_add_points)
        self.vem_pointer_end_add_points = self.vem.add(self.pointer_end_add_points)

        self.pointer_add_points = NormalButton(self, [0.075, 0.05, 0.2, 0.075], "Add points", self.create_from_existing)
        self.vem_pointer_add_points = self.vem.add(self.pointer_add_points)

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
        logging.info(
            f"""{self.__class__} EVENT: {event} ARTIST: {event.artist} 
            ID: {getattr(event.artist, 'id', None)}"""
        )

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
            self.picked_item = event.artist
            self.pick_pos = (event.mouseevent.xdata, event.mouseevent.ydata)

            # update fields
            self.vem.refresh()

    def press_event(self, event: MouseEvent) -> None:
        if self.is_adding_points:
            logging.info(f"""{self.__class__} POS: {event.xdata}, {event.ydata}""")
            self.points_to_add.append((event.xdata, event.ydata))

            self.state._hull_set_point(len(self.points_to_add) * (-1), event.xdata, event.ydata)
            self.pointer_points.append(
                PointArtist.point(self.vm.ax, len(self.points_to_add) * (-1), facecolor="red", edgecolor="red"))

            self.vem.refresh()
            plt.draw()

    def search_for_hull_name_in_hole(self, point, _hull_name=""):
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

    def distance(self, point_1, point_2):
        return np.sqrt((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2)

    def find_closest_edge_point_in_hull(self, hull_name, point):
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
        self.vem.add(ChangeViewButton(self, self.home_ax, "Home", ViewsEnum.HOME))
        self.vem.add(ChangeViewButton(self, self.clusters_ax, "Cluster", ViewsEnum.CLUSTER))
        self.vem.add(ChangeViewButton(self, self.hulls_ax, "Hulls", ViewsEnum.HULLS))
        self.vem.add(ChangeViewButton(self, self.labels_ax, "Labels", ViewsEnum.LABELS))
        self.vem.add(ChangeViewButton(self, [0.85, 0.936, 0.1, 0.06], "Back", ViewsEnum.HULLS))

        self.pointer_remove_line = NormalButton(self, [0.075, 0.05, 0.2, 0.075], "Remove line", self.remove_line)
        self.vem_pointer_remove_line = self.vem.add(self.pointer_remove_line)

        reset_b = NormalButton(self, [0.85, 0.05, 0.1, 0.075], "Reset", self.reset_removing_line)
        reset_b.button_ax.set_facecolor("lightcoral")
        reset_b.button_ref.color = "lightcoral"
        reset_b.button_ref.hovercolor = "crimson"
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
        logging.info(
            f"""{self.__class__} EVENT: {event} ARTIST: {event.artist} 
            ID: {getattr(event.artist, 'id', None)}"""
        )

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
            self.picked_item = event.artist
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
                self.pointer_points.append(
                    PointArtist.point(self.vm.ax, len(self.points_to_add) * (-1), facecolor="red", edgecolor="red"))

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
            final_cords = cords[:start_index] + cords[end_index + 1:]
            final_lines = lines[:line_start_index] + lines[line_end_index + 1:]

        else:
            final_cords = cords[end_index + 1: start_index]
            final_lines = lines[line_end_index + 1: line_start_index]

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
