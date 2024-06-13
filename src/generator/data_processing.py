import random
from typing import Union

import pandas as pd
import numpy as np
from mapel.core.objects.Experiment import Experiment


def parse_data(data: Union[str, Experiment], delim=';') -> dict:
    """ Parse the input data into universal dataframe

    Parameters
    ----------
    data: str or Experiment
        Either string with path to the csv file or an instance of mapel...Experiment.
    delim: str
        If data is a csv file the delim is used as input delimeter.

    Raises
    ------
    TODO: what exceptions are thrown by parsing methodes
    
    Returns
    -------
    dict:
        Resulting universal dict for generator.
    
    """

    if isinstance(data, str):
        return _parse_csv(data, delim=delim)
    else:
        return _parse_experiment(data)


def normalize(data: dict, lb: int = -100, ub: int = 100) -> dict:
    """ Normalizes the data into given square bounds
    
    Parameters
    ----------
    data: dict
        The data to normalize.
    lb: int
        Lower bound.
    ub: int
        Upper bound.

    Returns
    -------
    dict:
        Reference to data

    """

    spread = ub - lb
    x_points = None
    y_points = None
    for category_data in data.values():

        if x_points is None or y_points is None:
            x_points = category_data['x']
            y_points = category_data['y']
            continue

        x_points = np.concatenate((x_points, category_data['x']))
        y_points = np.concatenate((y_points, category_data['y']))


    ul_point = (np.min(x_points), np.max(y_points))
    dr_point = (np.max(x_points), np.min(y_points))

    # scaling - all values are prescaled base on the upper-left and lower-right points soo that they
    # fit a square of bounds (lb, up). The proportions should be kept
    dx       = dr_point[0] - ul_point[0]
    dy       = ul_point[1] - dr_point[1]
    d        = max(dx, dy)
    x_shift  = None # shift are counted from left/down
    y_shift  = None #
    
    # unit calculation based on which orientation is wider
    if dx >= dy:
        x_shift = 0
        y_shift = (dx - dy)/2
    else:
        x_shift = (dx - dy)/2
        y_shift = 0

    for category_name in data.keys():
        for i in range(len(data[category_name]['x'])):

            rx = data[category_name]['x'][i]
            ry = data[category_name]['y'][i]

            nx = lb + (rx - ul_point[0] + x_shift)*spread/d
            ny = lb + (ry - dr_point[1] + y_shift)*spread/d

            data[category_name]['x'][i] = nx
            data[category_name]['y'][i] = ny

    return data


def get_all_points(data: dict) -> np.ndarray:
    xs, ys = None, None
    for name in data.keys():
        if xs is None:
            xs = data[name]['x']
        if ys is None:
            ys = data[name]['y']
        xs = np.concatenate((xs, data[name]['x']))
        ys = np.concatenate((ys, data[name]['y']))
    return np.column_stack((xs, ys))


def editor_format(data: dict) -> dict:
    """ Formats the data to editor readable format
    
    Parameters
    ----------
    data: dict
        The data to format.

    Returns
    -------
    dict:
        Reference to data

    """
    data = {"data": data,
            "clusters_data": {},
            "hulls_data": {},
            "labels_data": {}}
    
    return data


def _parse_csv(path: str, delim=';') -> dict:
    """ Parse method for csv files

    Parameters
    ----------
    path: str
        Path to the csv.
    delim: str
        Delimeter for csv parsing.

    Raises
    ------
    IOError:
        When any file operations fail.

    Returns
    -------
    dict:
        Resulting universal dict.
        
    """

    df  = pd.read_csv(path, sep=delim)
    ids = df.columns[0]

    df[ids] = df[ids].apply(lambda x: x.split(sep="_")[0])

    df = combine_mallows_urn(df, 'Mallows-Urn')

    categorized_names = df[ids].unique()
    gruped_points = {}

    for category in categorized_names:
        selected_points = df.loc[df[ids] == category]
        gruped_points[category] = {"x": selected_points["x"].to_numpy(),
                                   "y": selected_points["y"].to_numpy()}

    return gruped_points


def combine_mallows_urn(df, culture_name):
    """
    Method that combines cultures starting with the same culture name
    """
    mallows = df[df[df.columns[0]].str.contains(culture_name)]
    df.loc[mallows.index, df.columns[0]] = culture_name
    return df


def get_df_from_data(data):
    """
    Method that returns a dataframe from normalised data
    """
    return pd.DataFrame([
        {
            "x": data[culture_name]['x'][i],
            "y": data[culture_name]['y'][i],
            "type": culture_name,
        }
        for culture_name in data.keys()
        for i in range(len(data[culture_name]['x']))
    ])


def initialize_colors(data):
    """
    Method that generates colors for cultures based on normalised data
    """
    colors = {culture_name: f"#{random.randrange(0x1000000):06x}" for culture_name in data.keys()}
    colors["Removed"] = "black"
    return colors


def _parse_experiment(exp: Experiment) -> dict:
    """ Parse method for Experiment obj

    Parameters
    ----------
    exp: Experiment
        The experiment object

    Raises
    ------
    TODO: what raises?
        
    Returns
    -------
    dict:
        Resulting universal dict.
        
    """

    raise NotImplementedError
