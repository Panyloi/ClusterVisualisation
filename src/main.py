from .generator import *

from typing import Optional, Union
from mapel.core.objects.Experiment import Experiment


def draw_maps(raw_data: Union[str, Experiment], out_path: str, delim=';') -> Optional[dict]:
    """ Automatic map creator
    
    Parameters
    ----------
    raw_data: str or Experiment
        Either string with path to the csv file or an instance of mapel...Experiment
    out_path: str or None
        Path for the .jpg image to be saved to. If None raw drawer data is returned
        for the editor drawer operations
    delim: str
        If raw_data is a csv file the delim is used as input delimeter.

    Returns
    -------
    dict:
        If out_path is None. Dict has raw data needed for interactive editor.
    None:
        Otherwise.

    Notes
    -----
    Output format for the editor:
    {
        data: 
        {
            "culture_name":
            {
                'x': np.array(), 
                'y': np.array()
            },
            "seccond_culture_name":
            ...
        }

        cluster_data: 
        {
            "cluster_name": 
            {
                'x': np.array(), 
                'y': np.array(), 
                // TODO: additional cluster data?
                'org': list("original_culture_name", ...)
            },
            "seccond_cluster_name":
            ...
        }

        convex_data: 
        {
            "cluster_name":
            {
                // TODO: data for convex?
            },
            "seccond_cluster_name":
            ...
        }

        labels_data:
        {
            "cluster_name":
            {
                // TODO: data for labels?
            },
            "seccond_cluster_name":
            ...
        }
    }

    """

    parsed_data     = parse_data(raw_data)
    normalized_data = normalize(parsed_data)
    data            = editor_format(normalized_data)

    # TODO: generate the map

    return data


def draw_maps_editor(raw_data: Union[str, Experiment], delim=';') -> None:
    """ Interactive matplotlib editor for creating maps
    
    Parameters
    ----------
    raw_data: str or Experiment
        Either string with path to the csv file or an instance of mapel...Experiment
    delim: str
        If raw_data is a csv file the delim is used as input delimeter.
    
    """
    pass

