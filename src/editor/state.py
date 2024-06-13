import logging
import random
from typing import Callable, Type, TypeVar, ParamSpec, Union
import json
import numpy as np
import pandas as pd
from copy import deepcopy

from matplotlib.axes._axes import Axes
import matplotlib.pyplot as plt
from functools import wraps

T = TypeVar('T')
P = ParamSpec('P')

def KeyErrorWrap(default) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Wrapper for the default value of return on KeyError exception.
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                result = func(*args, **kwargs)
                return result
            except KeyError as e:
                logging.error(f"{e} func {func.__name__}")
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
        
        # pythonize data
        def pythonize_dict(d: dict):
            for key, value in d.items():
                
                # recurse for inside dictionaries
                if isinstance(value, dict):
                    pythonize_dict(value)
                    
                # recurse for inside lists
                if isinstance(value, list):
                    pythonize_list(value)
                    
                # prune
                if isinstance(value, np.ndarray):
                    d[key] = value.tolist()
                    
        def pythonize_list(l: list):
            for i, value in enumerate(l):
                
                # recurse for inside dictionaries
                if isinstance(value, dict):
                    pythonize_dict(value)
                    
                # recurse for inside lists
                if isinstance(value, list):
                    pythonize_list(value)
                    
                # prune
                if isinstance(value, np.ndarray):
                    l[i] = value.tolist()
        
        pythonize_dict(data)
        self.data = data

    def draw(self, ax: Axes):
        """This function should be overriten later after initialization of artists"""
        raise NotImplementedError

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

    # ------------------------------- HULLS_GETTERS ------------------------------ #

    @KeyErrorWrap([])
    def get_hull_polygon_cords(self, hull_id: int) -> list[tuple[float, float]]:
        return self.data['hulls_data'][hull_id]['cords']

    @KeyErrorWrap([])
    def get_hull_lines_cords(self, hull_id: int) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        return self.data['hulls_data'][hull_id]['line_cords']

    # ------------------------------- CLUSTER GETTERS ------------------------------ #
    @KeyErrorWrap(None)
    def get_cluster(self, cluster_name: str) -> pd.DataFrame:
        df = self.data['clusters_data']['points']
        return df[df['type'] == cluster_name]

    @KeyErrorWrap(None)
    def get_all_clusters(self) -> dict:
        df = self.data['clusters_data']['points']
        return df.groupby('type').apply(lambda x: x[['x', 'y']].values.tolist()).to_dict()

    @KeyErrorWrap(None)
    def get_all_points(self) -> pd.DataFrame:
        return self.data['clusters_data']['points']

    @KeyErrorWrap(None)
    def get_point(self, point_id) -> dict:
        df = self.data['clusters_data']['points'].loc[point_id]
        return df.to_dict()

    @KeyErrorWrap(None)
    def get_point_pos(self, point_id) -> tuple:
        df = self.data['clusters_data']['points']
        return df.loc[point_id, 'x'], df.loc[point_id, 'y']

    @KeyErrorWrap(None)
    def get_point_color(self, point_id) -> str:
        df = self.data['clusters_data']['points']
        cluster_name = df.loc[point_id, 'type']
        return self.data['clusters_data']['colors'][cluster_name]

    @KeyErrorWrap(None) #todo sometimes data is null, needs fix
    def get_normalised_clusters(self) -> dict:
        df = self.data['clusters_data']['points']
        data = {type_name: {'x': np.array(type_data['x']), 'y': np.array(type_data['y'])} for type_name, type_data in df.groupby('type')}
        if "Removed" in data.keys():
            data.pop("Removed")
        return data

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

    # ------------------------------- CLUSTER SETTERS ------------------------------ #
    @KeyErrorWrap(None)
    def set_cluster(self, cluster_name: str, points: list) -> None:
        if cluster_name not in self.data['clusters_data']['colors'].keys():
            self.data['clusters_data']['colors'][cluster_name] = f"#{random.randrange(0x1000000):06x}"
        df = self.data['clusters_data']['points']
        df.loc[df.index.isin(points), 'type'] = cluster_name
        self.data['clusters_data']['points'] = df

    @KeyErrorWrap(None)
    def reset_clusters(self) -> None:
        self.data['clusters_data']['points'] = pd.DataFrame([
            {
                "x": self.data['data'][culture_name]['x'][i],
                "y": self.data['data'][culture_name]['y'][i],
                "type": culture_name,
            }
            for culture_name in self.data['data'].keys()
            for i in range(len(self.data['data'][culture_name]['x']))
        ])

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
    
    @KeyErrorWrap(None)
    def delete_hull(self, hull_id: int) -> None:
        self.data['hulls_data'].pop(hull_id)

    # ----------------------------------- MISC ----------------------------------- #

    @staticmethod
    def _retype_state(state: dict) -> dict:
        state_cp = deepcopy(state)

        for key in state['labels_data']:
            try:
                int(key) # prune strings early
                label_data = state_cp['labels_data'].pop(key)

                label_data['x'] = float(label_data['x'])
                label_data['y'] = float(label_data['y'])

                label_data_cp = deepcopy(label_data)
                for arrow_key in label_data['arrows']:
                    arrow_data = label_data_cp['arrows'].pop(arrow_key)

                    arrow_data['ref_x'] = float(arrow_data['ref_x'])
                    arrow_data['ref_y'] = float(arrow_data['ref_y'])
                    arrow_data['att_x'] = float(arrow_data['att_x'])
                    arrow_data['att_y'] = float(arrow_data['att_y'])

                    label_data_cp['arrows'][int(arrow_key)] = arrow_data

                state_cp['labels_data'][int(key)] = label_data_cp
            except ValueError:
                state_cp['labels_data'][key] = float(state_cp['labels_data'][key])
        
        return state_cp

    def get_raw(self) -> dict:
        return self.data
    
    def set_raw(self, data) -> None:
        self.data = data

    def save_state_to_file(self, fpath: str) -> None:
        with open(fpath, 'w') as f:
            json.dump(self.data, f)

    def load_state_from_file(self, fpath: str) -> 'State':
        with open(fpath, 'r') as f:
            data = json.load(f)
        self.data = self._retype_state(data)


class StateLinker:
    """ State class linker
    
    Every class that needs access to the global editor state should derive
    from this class. Basically works as a shared global static variable.

    """

    state: State

    @classmethod
    def link_state(cls, lstate):
        cls.state = lstate
