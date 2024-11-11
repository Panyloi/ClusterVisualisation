from ..view_manager import *


class Home(View):

    def __init__(self, view_manager: ViewManager) -> None:
        super().__init__(view_manager)
        view_button = self.vem.add(ChangeViewButton(self, self.home_ax, "Home", ViewsEnum.HOME))
        self.vem.add(ChangeViewButton(self, self.labels_ax, "Labels", ViewsEnum.LABELS))
        self.vem.add(ChangeViewButton(self, self.clusters_ax, "Cluster", ViewsEnum.CLUSTER))
        self.vem.add(ChangeViewButton(self, self.hulls_ax, "Hulls", ViewsEnum.HULLS))
        view_button.highlight()

    def draw(self, *args, **kwargs) -> None:
        super().draw()

        self.state.show_labels_and_hulls(self.vm.ax)
        self.vm.list_manager.hide_button()
        self.vm.ax.set_xlim(-190, 190)
        self.vm.ax.set_ylim(-150, 150)

        plt.draw()

    def hide(self) -> None:
        self.vm.list_manager.show_button()
        super().hide()
