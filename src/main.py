from .generator.data_processing import *
from .editor.view_manager import *
from .editor.views import *

from typing import Optional, Union
from mapel.core.objects.Experiment import Experiment
import sys
import logging


def draw_maps(raw_data: Union[str, Experiment], out_path: str, delim=';') -> Optional[State]:
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
            label_id: int:
            {
                'text': str
                'x': float
                'y': float
                'arrows':
                {
                    "arrow_id":
                    {
                        'ref_x': float
                        'ref_y': float
                        'att_x': float
                        'att_y': float
                        'val': str
                    }
                }
                'att_points': list[tuple[float]]
                'ref_point_vals': list[str]
            },
            seccond_label_id: int:
            ...
        }
    }

    """

    parsed_data     = parse_data(raw_data)
    normalized_data = normalize(parsed_data)
    state_dict      = editor_format(normalized_data)

    # TODO: generate the map
    # add tmp labels
    state_dict['labels_data'][0] =  {'text': "tlabel", 
                                      'x': 60, 
                                      'y': 60,
                                      'arrows': 
                                      {
                                        0:
                                        {
                                            'ref_x': 0,
                                            'ref_y': 0,
                                            'att_x': 55,
                                            'att_y': 58,
                                            'val': "0.2"
                                        },
                                        1:
                                        {
                                            'ref_x': -10,
                                            'ref_y': 10,
                                            'att_x': 55,
                                            'att_y': 58,
                                            'val': "0.3"
                                        }
                                      }
                                    }
    state_dict['labels_data'][1] =  {'text': "another", 
                                      'x': -70, 
                                      'y': -70,
                                      'arrows': 
                                      {
                                        0:
                                        {
                                            'ref_x': 30,
                                            'ref_y': -30,
                                            'att_x': -65,
                                            'att_y': -68,
                                            'val': "a"
                                        },
                                        1:
                                        {
                                            'ref_x': -4,
                                            'ref_y': -10,
                                            'att_x': -65,
                                            'att_y': -68,
                                            'val': "b"
                                        }
                                      }
                                    }

    return State(state_dict)


def draw_maps_editor(raw_data: Union[str, Experiment], delim=';') -> None:
    """ Interactive matplotlib editor for creating maps
    
    Parameters
    ----------
    raw_data: str or Experiment
        Either string with path to the csv file or an instance of mapel...Experiment
    delim: str
        If raw_data is a csv file the delim is used as input delimeter.
    
    """
    
    state = draw_maps(raw_data, None)

    FORMAT = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
    editor = Editor(state)
    editor.run()
