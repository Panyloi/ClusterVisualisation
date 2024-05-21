import logging
from typing import Callable, Type, TypeVar, ParamSpec, Union

from matplotlib.axes._axes import Axes

T = TypeVar('T')
P = ParamSpec('P')

def KeyErrorWrap(default) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Wrapper for the default value of return on KeyError exception.
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                result = func(*args, **kwargs)
                return result
            except KeyError as e:
                logging.error(f"{e}")
                return default
        return wrapper
    return decorator
  
def get_artists_by_type(ax: Axes, artist_type: Type[T]) -> list[T]:
    children = ax.get_children()
    artists = []
    for child in children:
        if isinstance(child, artist_type):
            artists.append(child)
    return artists

def subtract_with_default(value1: Union[T, None], value2: Union[T, None], default: T) -> T:
    if value1 is not None and value2 is not None:
        if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
            return value1 - value2
        raise ValueError("Unsupported types for subtraction")
    else:
        return default


class State:
    """ 
    Wrapper class for the editor data and state.
    Provides getters, setters and storage for options.
    """

    def __init__(self, data: dict) -> None:
        """ Data init
        
        Parameters
        ----------
        data: dict
            Data in proper editor format
        
        """
        
        self.data = data

    # ---------------------------------- GETTERS --------------------------------- #

    @KeyErrorWrap("")
    def get_label_text(self, label_id: int) -> str:
        return self.data['labels_data'][label_id]['text']
    
    @KeyErrorWrap((0, 0))
    def get_label_pos(self, label_id: int) -> tuple[float, float]:
        return self.data['labels_data'][label_id]['x'], self.data['labels_data'][label_id]['y']
    
    @KeyErrorWrap({})
    def get_label_arrows(self, label_id: int) -> dict:
        return self.data['labels_data'][label_id]['arrows']
    
    @KeyErrorWrap({})
    def get_label_arrow(self, label_id: int, arrow_id: int) -> dict:
        return self.data['labels_data'][label_id]['arrows'][arrow_id]
    
    @KeyErrorWrap((0, 0))
    def get_arrow_ref_point(self, label_id: int, arrow_id: int) -> tuple[float, float]:
        return self.data['labels_data'][label_id]['arrows'][arrow_id]['ref_x'],\
               self.data['labels_data'][label_id]['arrows'][arrow_id]['ref_y']

    @KeyErrorWrap((0, 0))
    def get_arrow_att_point(self, label_id: int, arrow_id: int) -> tuple[float, float]:
        return self.data['labels_data'][label_id]['arrows'][arrow_id]['att_x'],\
               self.data['labels_data'][label_id]['arrows'][arrow_id]['att_y']
    
    @KeyErrorWrap("")
    def get_arrow_val(self, label_id: int, arrow_id: int) -> str:
        return self.data['labels_data'][label_id]['arrows'][arrow_id]['val']
    
    @KeyErrorWrap(10)
    def get_label_size(self) -> float:
        return self.data['labels_data']['size']
    
    @KeyErrorWrap(1)
    def get_arrow_size(self) -> float:
        return self.data['labels_data']['arrow_size']
    
    # ---------------------------------- SETTERS --------------------------------- #

    @KeyErrorWrap(None)
    def set_label_text(self, label_id: int, text: str) -> None:
        self.data['labels_data'][label_id]['text'] = text
        
    @KeyErrorWrap(None)
    def set_label_pos(self, label_id: int, x: float, y: float) -> None:
        self.data['labels_data'][label_id]['x'] = x
        self.data['labels_data'][label_id]['y'] = y

    @KeyErrorWrap(None)
    def set_arrow_att_pos(self, label_id: int, arrow_id: int, att_x: float, att_y: float) -> None:
        self.data['labels_data'][label_id]['arrows'][arrow_id]['att_x'] = att_x
        self.data['labels_data'][label_id]['arrows'][arrow_id]['att_y'] = att_y

    @KeyErrorWrap(None)
    def set_arrow_ref_pos(self, label_id: int, arrow_id: int, ref_x: float, ref_y: float) -> None:
        self.data['labels_data'][label_id]['arrows'][arrow_id]['ref_x'] = ref_x
        self.data['labels_data'][label_id]['arrows'][arrow_id]['ref_y'] = ref_y

    @KeyErrorWrap(None)
    def set_arrow_val(self, label_id: int, arrow_id: int, val: str) -> None:
        self.data['labels_data'][label_id]['arrows'][arrow_id]['val'] = val

    @KeyErrorWrap(None)
    def set_label_size(self, size: float) -> None:
        self.data['labels_data']['size'] = size

    @KeyErrorWrap(None)
    def set_arrow_size(self, size: float) -> None:
        self.data['labels_data']['arrow_size'] = size
        
    # ------------------------------------ ADD ----------------------------------- #
    
    @KeyErrorWrap(-1)
    def add_empty_label(self) -> int:
        nid = max( filter(lambda x: True if isinstance(x, int) else False,
                          self.data['labels_data'].keys())) + 1
        self.data['labels_data'][nid] = \
        {
            'text': "...",
            'x': 0,
            'y': 0,
            'arrows': {}
        }
        return nid
    
    @KeyErrorWrap(-1)
    def add_empty_arrow(self, label_id: int) -> int:
        if self.get_label_arrows(label_id):
            nid = max(self.get_label_arrows(label_id).keys()) + 1
        else:
            nid = 0
        self.data['labels_data'][label_id]['arrows'][nid] = \
        {
            'ref_x': 0,
            'ref_y': 0,
            'att_x': self.get_label_pos(label_id)[0],
            'att_y': self.get_label_pos(label_id)[1],
            'val': ""
        }
        return nid
    
    # ---------------------------------- DELETE ---------------------------------- #
    
    @KeyErrorWrap(None)
    def delete_label(self, label_id: int) -> None:
        self.data['labels_data'].pop(label_id)
        
    @KeyErrorWrap(None)
    def delete_arrow(self, label_id: int, arrow_id: int) -> None:
        self.data['labels_data'][label_id]['arrows'].pop(arrow_id)

    # ----------------------------------- MISC ----------------------------------- #

    def get_raw(self) -> dict:
        return self.data
    
    def set_raw(self, data) -> None:
        self.data = data


class StateLinker:
    """ State class linker
    
    Every class that has access to the global editor state should derive
    from this class. After that class variable with state is accessible.

    """

    state: State

    @classmethod
    def link_state(cls, lstate):
        cls.state = lstate
