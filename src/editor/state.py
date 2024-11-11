import logging
import random
from typing import Callable, Type, TypeVar, ParamSpec, Union
import json
import numpy as np
import pandas as pd
from copy import deepcopy

from ..generator.hull_generator import calc_one_hull


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

    def hide_labels_and_hulls(self, ax: Axes):
        raise NotImplementedError

    def show_labels_and_hulls(self, ax: Axes):
        raise NotImplementedError

    def show_labels(self, ax: Axes):
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

    def get_hull_line_size(self) -> float:
        return self.data['hulls_data']['line_size']

    def get_hull_hull_line(self, hull_name: str) -> dict:
        return self.data['hulls_data'][hull_name]['hull_line']

    def get_hulls_view_state(self) -> bool:
        return self.data['hulls_data']['view_state']

    def get_hull_line_points(self, hull_name: str, hull_line_id: int):
        return self.data['hulls_data'][hull_name]['hull_line'][hull_line_id]['x1'],\
               self.data['hulls_data'][hull_name]['hull_line'][hull_line_id]['y1'],\
               self.data['hulls_data'][hull_name]['hull_line'][hull_line_id]['x2'],\
               self.data['hulls_data'][hull_name]['hull_line'][hull_line_id]['y2']
    
    def get_hull_line_val(self, hull_name: str, hull_line_id: int) -> str:
        return self.data['hulls_data'][hull_name]['hull_line'][hull_line_id]['val']

    def get_hull_polygon_cords(self, hull_name: str) -> list[tuple[float, float]]:
        return self.data['hulls_data']['hulls'][hull_name]['cords']

    def get_hull_interpolated_cords(self, hull_name: str) -> list[tuple[float, float]]:
        return self.data['hulls_data']['hulls'][hull_name]['interpolate_points']

    def get_hull_lines_cords(self, hull_name: str) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        return self.data['hulls_data']['hulls'][hull_name]['line_cords']
    
    def update_hulls(self):
        if len(self.data['hulls_data']['undraw']) == 0 and len(self.data['hulls_data']['change']) == 0:
            return

        for hull_name in self.data['hulls_data']['undraw']:
            if hull_name in self.data['hulls_data']['hulls']:
                del self.data['hulls_data']['hulls'][hull_name]

        for hull_name in self.data['hulls_data']['change'].keys():
            new_hull = calc_one_hull(hull_name, self.data['hulls_data']['change'][hull_name], self.data)

            self.data['hulls_data']['hulls'][hull_name] = {
                'cords': new_hull[hull_name]['polygon_points'],
                'line_cords': new_hull[hull_name]['polygon_lines'],
                'cluster_points': new_hull[hull_name]['cluster_points'],
                "interpolate_points": new_hull[hull_name]["interpolate_points"],
                "hole_in_hulls": [],
                "artist": None
            }

        self.data['hulls_data']['undraw'] = set()
        self.data['hulls_data']['change'] = {}

    def redefine_hull(self, hull_name: str):

        hull_resources = self.data['hulls_data']['hulls'][hull_name]
        del self.data['hulls_data']['hulls'][hull_name]

        print(hull_resources['artist'])
        self.data['hulls_data']['hulls'][hull_name] = {
            "cords": hull_resources["polygon_points"],
            "line_cords": hull_resources["polygon_lines"],
            "cluster_points": hull_resources["cluster_points"],
            "interpolate_points": hull_resources["interpolate_points"],
            "hole_in_hulls": [],
            "artist": None
        }

    def add_hull(self, hull_name: str, cords = [], line_cords = [], cluster_points = [], interpolate_points = []):

        self.data['hulls_data']['hulls'][hull_name] = {
            "cords": cords,
            "line_cords": line_cords,
            "cluster_points": cluster_points,
            "interpolate_points": interpolate_points,
            "hole_in_hulls": [],
            "artist": None
        }

    def get_hull_view_state(self) -> bool:
        return self.data["hulls_data"]["view_state"]
    
    def get_all_hulls_name(self) -> bool:
        return self.data["hulls_data"]['hulls'].keys()

    def get_hulls_artist(self, hull_name: str):
        return self.data['hulls_data']['hulls'][hull_name]['artist']

    def get_hole_in_hulls(self, hull_name: str) -> None:
        return self.data['hulls_data']['hulls'][hull_name]['hole_in_hulls']

    def get_hulls_render_name(self) -> None:
        return self.data['hulls_data']['render_name'] + 1

    # ------------------------------- HULLS SETTERS ------------------------------ #

    def set_hull_to_undraw(self, hull_name):
        self.data['hulls_data']['undraw'].add(hull_name)

    def set_hull_to_change(self, hull_name, points):
        self.data['hulls_data']['change'][hull_name] = points

    def set_hulls_view_state(self, in_view: bool) -> None:
        self.data['hulls_data']['view_state'] = in_view

    def set_hulls_render_name(self, name: int) -> None:
        self.data['hulls_data']['render_name'] = name

    def set_hull_line_pos(self, hull_name: int, hull_line_id: int, x1: float, y1: float, x2: float, y2: float) -> None:
        self.data['hulls_data'][hull_name]['hull_line'][hull_line_id]['x1'] = x1
        self.data['hulls_data'][hull_name]['hull_line'][hull_line_id]['y1'] = y1
        self.data['hulls_data'][hull_name]['hull_line'][hull_line_id]['x2'] = x2
        self.data['hulls_data'][hull_name]['hull_line'][hull_line_id]['y2'] = y2

    def set_hull_line_val(self, hull_name: int, hull_line_id: int, val: str) -> None:
        self.data['hulls_data'][hull_name]['hull_line'][hull_line_id]['val'] = val

    def set_hull_line_size(self, size: float) -> None:
        self.data['hulls_data']['hull_line_size'] = size

    def hull_set_point(self, point_id: int, x: float, y: float) -> None:
        df = self.data['clusters_data']['points']
        df.loc[point_id] = [x, y, 'mine']

        df_1 = self.data['clusters_data']['colors']
        df_1['mine'] = f"#{0x0000000}"

    def hull_remove_point(self, point_id) -> None:
        df = self.data['clusters_data']['points']
        self.data['clusters_data']['points'] = df.drop(point_id)

        df_1 = self.data['clusters_data']['colors']
        if 'mine' in df_1.keys():
            del df_1['mine']

    def set_hull_polygon_cords(self, hull_name: str, new_cords) -> None:
        self.data['hulls_data']['hulls'][hull_name]['cords'] = new_cords

    def set_hull_lines_cords(self, hull_name: str, new_line_cords) -> None:
        self.data['hulls_data']['hulls'][hull_name]['line_cords'] = new_line_cords

    def save_hulls_artist(self, hull_name: str, artist) -> None:
        self.data['hulls_data']['hulls'][hull_name]['artist'] = artist

    def set_hull_interpolated_cords(self, hull_name: str, new_cords) -> None:
        self.data['hulls_data']['hulls'][hull_name]['interpolate_points'] = new_cords

    def set_hole_in_hulls(self, hull_name: str, points: tuple[tuple[float, float], tuple[float, float]]) -> None:
        self.data['hulls_data']['hulls'][hull_name]['hole_in_hulls'].append(points)

    def remove_hole_in_hulls(self, hull_name: str, points: tuple[tuple[float, float], tuple[float, float]]) -> None:
        self.data['hulls_data']['hulls'][hull_name]['hole_in_hulls'].remove(points)



    # ------------------------------- CLUSTER GETTERS ------------------------------ #
    def get_cluster(self, cluster_name: str) -> pd.DataFrame:
        df = self.data['clusters_data']['points']
        return df[df['type'] == cluster_name]

    def get_all_clusters(self) -> dict:
        df = self.data['clusters_data']['points']
        return df.groupby('type').apply(lambda x: x[['x', 'y']].values.tolist()).to_dict()

    def get_all_points(self) -> pd.DataFrame:
        return self.data['clusters_data']['points']

    def get_point(self, point_id) -> dict:
        df = self.data['clusters_data']['points'].loc[point_id]
        return df.to_dict()

    def get_point_pos(self, point_id) -> tuple:
        df = self.data['clusters_data']['points']
        return df.loc[point_id, 'x'], df.loc[point_id, 'y']

    def get_point_color(self, point_id) -> str:
        df = self.data['clusters_data']['points']
        cluster_name = df.loc[point_id, 'type']
        return self.data['clusters_data']['colors'][cluster_name]

    def get_normalised_clusters(self) -> dict:
        df = self.data['clusters_data']['points']
        data = {type_name: {'x': np.array(type_data['x']), 'y': np.array(type_data['y'])} for type_name, type_data in df.groupby('type')}
        if "Removed" in data.keys():
            data.pop("Removed")
        return data

    # ---------------------------------- SETTERS --------------------------------- #

    def set_label_text(self, label_id: int, text: str) -> None:
        self.data['labels_data'][label_id]['text'] = text

    def set_label_pos(self, label_id: int, x: float, y: float) -> None:
        self.data['labels_data'][label_id]['x'] = x
        self.data['labels_data'][label_id]['y'] = y

    def set_arrow_att_pos(self, label_id: int, arrow_id: int, att_x: float, att_y: float) -> None:
        self.data['labels_data'][label_id]['arrows'][arrow_id]['att_x'] = att_x
        self.data['labels_data'][label_id]['arrows'][arrow_id]['att_y'] = att_y

    def set_arrow_ref_pos(self, label_id: int, arrow_id: int, ref_x: float, ref_y: float) -> None:
        self.data['labels_data'][label_id]['arrows'][arrow_id]['ref_x'] = ref_x
        self.data['labels_data'][label_id]['arrows'][arrow_id]['ref_y'] = ref_y

    def set_arrow_val(self, label_id: int, arrow_id: int, val: str) -> None:
        self.data['labels_data'][label_id]['arrows'][arrow_id]['val'] = val

    def set_label_size(self, size: float) -> None:
        self.data['labels_data']['size'] = size

    def set_arrow_size(self, size: float) -> None:
        self.data['labels_data']['arrow_size'] = size

    # ------------------------------- CLUSTER SETTERS ------------------------------ #
    def set_cluster(self, cluster_name: str, points: list) -> None:
        if cluster_name not in self.data['clusters_data']['colors'].keys():
            self.data['clusters_data']['colors'][cluster_name] = f"#{random.randrange(0x1000000):06x}"
        df = self.data['clusters_data']['points']
        df.loc[df.index.isin(points), 'type'] = cluster_name
        self.data['clusters_data']['points'] = df

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

    def delete_label(self, label_id: int) -> None:
        self.data['labels_data'].pop(label_id)

    def delete_arrow(self, label_id: int, arrow_id: int) -> None:
        self.data['labels_data'][label_id]['arrows'].pop(arrow_id)

    def delete_hull(self, hull_name: str) -> None:
        self.data['hulls_data']['hulls'].pop(hull_name)

    def delete_hull_line(self, hull_name: str, hull_line_id: int) -> None:
        self.data['hulls_data'][hull_name]['hull_line'].pop(hull_line_id)

    def delete_hulls(self) -> None:
        keys = list(self.data['hulls_data']['hulls'].keys())
        for key in keys:
            self.delete_hull(key)

    # ----------------------------------- MISC ----------------------------------- #

    @staticmethod
    def _retype_state(state: dict, old_data: dict) -> dict:
        state_cp = deepcopy(state)
        
        if "labels_data" in state.keys():
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
        else:
            state_cp["labels_data"] = old_data["label_data"]

        if "data" in state.keys():
            state_cp["data"] = state["data"]
        else:
            state_cp["data"] = old_data["data"]

        if "clusters_data" in state.keys():
            state_cp["clusters_data"] = state["clusters_data"]
        else:
            state_cp["clusters_data"] = old_data["clusters_data"]
            
        if "hulls_data" in state.keys():
            state_cp["hulls_data"] = state["hulls_data"]
        else:
            state_cp["hulls_data"] = old_data["hulls_data"]
        
        return state_cp

    def get_raw(self) -> dict:
        return self.data
    
    def set_raw(self, data) -> None:
        self.data = data

    def _prepare_serialazable_data(self) -> dict:
        serializable_data = {}
        for block_name in ("data", "clusters_data", "hulls_data", "labels_data"):
            try:
                json.dumps(self.data[block_name])
                serializable_data[block_name] = self.data[block_name]
            except Exception as e:
                logging.info(f"Block {block_name} is not serializable")
        return serializable_data

    def save_state_to_file(self, fpath: str) -> None:
        with open(fpath, 'w') as f:
            json.dump(self._prepare_serialazable_data(), f, skipkeys=True)

    def load_state_from_file(self, fpath: str) -> 'State':
        with open(fpath, 'r') as f:
            data = json.load(f)
        self.data = self._retype_state(data, self.data)


class StateLinker:
    """ State class linker
    
    Every class that needs access to the global editor state should derive
    from this class. Basically works as a shared global static variable.

    """

    state: State

    @classmethod
    def link_state(cls, lstate):
        cls.state = lstate
