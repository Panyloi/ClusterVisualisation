# How to create a view

## 1. Create an empty class with all derived methods.
All Views should be derived from the base class `View`. View provides two abstract methods, a non-abstract fixed method 
for changing between views and a constructor. 
Additionally, there are a few important attributes:
 + `self.vm` - View Manager. Provides transitions between views (only internal class, not to be messed around).
 + `self.vem` - View Element Manager. Provides functionality for managing objects that derive from the base class `ViewElement`.
 + `self.cem` - Canvas Event Manager. Provides functionality for managing events.
 + `self.state` - State. Since View derives from StateLinker it has access to the global editor state

Empty View looks like this:
```python
class EmptyView(View):

    def __init__(self, view_manager: ViewManager) -> None:
        super().__init__(view_manager)

    def draw(self, *args, **kwargs) -> None:
        return super().draw()
    
    def undraw(self) -> None:
        return super().undraw()
```

## 2. Link the class to the global editor
For View to be reachable, it must correspond to an enum in ViewsEnum. 
Then, according to the order of the views in the enum, the class object has to be registered in the main editor class. 
`vm.register_views([Home(vm), LabelsView(vm), ...])` "..." should be replaced with the object.

## 3. Populate the class
For example, lets say we want to add a button and an event. 

Adding the button looks like this:
```python
def draw(self, *args, **kwargs) -> None:
    super().draw()
    
    self.vem.add(ChangeViewButton(self, [0.05, 0.05, 0.1, 0.075], "Home", ViewsEnum.HOME))

    plt.draw()
```
The button must derive, directly or not, from the ViewElement base class. 

Adding the event looks the same, however Canvas Event Manager (cem) is used instead of the View Element Manager (vem).
```python
def draw(self, *args, **kwargs) -> None:
    super().draw()
    
    self.vem.add(ChangeViewButton(self, [0.05, 0.05, 0.1, 0.075], "Home", ViewsEnum.HOME))
    self.cem.add(SharedEvent('pick_event', lambda ev: print("c:")))

    plt.draw()
```
Both events and view elements will be automatically disconnected and removed from the canvas when the view is changed by the vem and cem managers. 
Additionally, events can be disconnected in several other ways such as (disconnect_unique, disconnect_shared, disconnect) for better manipulation of events exclusion.

## 4. Refreshing
vem class implements an additional refresh method. It calls refresh on all the connected view element objects. 
If an object is created in such a way that it can refresh, simply calling `self.vem.refresh()` will do the job.

## 5. Starting arguments
If needed a view can be drawn with some arguments (preferably kwargs). 
However, it is not possible to achieve this with a change view button! 
Just add needed kwargs when directly calling `view.change_view`, for example:
```python
self.change_view(ViewsEnum.ARROWS, picked_item=event.artist)
```
Then, retrival in the draw method of the view looks like this:
```python
    def draw(self, *args, **kwargs) -> None:
        super().draw()

        # get picked arrow if exists
        self.picked_item = kwargs.get('picked_item', None)
```