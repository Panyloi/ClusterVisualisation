from matplotlib import pyplot as plt

from .backend_customs import SaveLoadPltLinker, custom_buttons_setup
from .state import State, StateLinker
from .view_manager import Event, ViewManager
from .views.clusters_view import ClusterMainView, AgglomerativeView, DBSCANView
from .views.home_view import Home
from .views.hulls_view import HullView, CreateNewHullView, RemoveHullLineView
from .views.labels_view import LabelsView, ArrowsView


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
