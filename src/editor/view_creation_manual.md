# How to create a view

## 1. Create an empty class with all deriviated methodes.
All Views should deriviate from base class `View`. View provides 2 abstract methodes, one
non-abstract fixed method for changeing between views and constructor. Additionaly there are few important attributes.
 + `self.vm` - View Manager. Provides transitions between views(only internall class not to mess around with).
 + `self.vem` - View Element Manager. Provides functionality for managing objects that deriviate from base class `ViewElement`.
 + `self.cem` - Canvas Event Manager. Provides functionality for managing events. There is no custom base class for events.
 + `self.state` - State. Because View deriviates from StateLinker it has access to global editor state

Empty View should look something like this:
```python
class EmptyView(View):

    def __init__(self, view_manager: ViewManager) -> None:
        super().__init__(view_manager)

    def draw(self) -> None:
        return super().draw()
    
    def undraw(self) -> None:
        return super().undraw()
```

## 2. Link the class to global editor
For View to be reachable it needs to create it's own enum in ViewsEnum. After that according to views enum order a class object needs to be registered in the main editor class. `vm.register_views([Home(vm), LabelsView(vm), ...])` in the ... space goes the object.

## 3. Populate the class
For example lets say we want to add an button and an event. Adding a button would look like this
```python
def draw(self) -> None:
    super().draw()
    
    self.vem.add(ChangeViewButton(self, [0.05, 0.05, 0.1, 0.075], "Home", ViewsEnum.HOME))

    plt.draw()
```
The button needs to deriviate directly or not from ViewElement base class. Adding event is very similar but we use canvas event manager instead.
```python
def draw(self) -> None:
    super().draw()
    
    self.vem.add(ChangeViewButton(self, [0.05, 0.05, 0.1, 0.075], "Home", ViewsEnum.HOME))
    self.cem.add(self.vm.fig.canvas.mpl_connect('pick_event', lambda ev: print("c:")))

    plt.draw()
```
Both events and view elements will be automaticly disconnected and eraseed from the canvase when the view changes by the vem and cem managers.

## 4. Refreshing
vem class implements additional refresh method. This will call refresh on all the connected view element objects. If an object is created in such way that it can refresh simple calling `self.vem.refresh()` will do the job.