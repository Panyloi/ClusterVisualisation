"""
Module with main exported functions of the package
"""

import sys
import logging
from typing import Optional, Union
import os

import matplotlib.pyplot as plt
from mapel.core.objects.Experiment import Experiment

from .generator.data_processing import parse_data, normalize, get_all_points
from .generator.data_processing import editor_format, get_df_from_data, initialize_colors
from .editor.view_manager import State, StateLinker
from .editor.views import parse_solution_to_editor, Editor
from .generator.labels_generator import calc
from .generator.hull_generator import calc_hull, parse_solution_to_editor_hull, set_hull_parameters

# disabling excessive Pillow logging
logging.getLogger('PIL').setLevel(logging.WARNING)


def draw_maps(raw_data: Union[str, Experiment],
              out_path: str | None, delim=';',
              config_id: str = 'divide_and_conquare') -> Optional[State]:
    """ Automatic map creator
    
    Parameters
    ----------
    raw_data: str or Experiment
        Either string with path to the csv file or an instance of mapel...Experiment
    out_path: str or None
        Path for the .jpg image to be saved to. If None raw drawer data is returned
        for the editor drawer operations
    delim: str
        If raw_data is a csv file the delim is used as input delimiter.
    config_id: str
        String id specifying a group of parameters in the Configuration class which 
        affect auto generation

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
            "second_culture_name":
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
            "second_cluster_name":
            ...
        }

        hulls_data: 
        {
            hulls:
            {
                hull_name
                {
                    cords': list[tuple[float, float]] 
                        // coordinates of hull points
                    'line_cords': list[tuple[tuple[float, float], tuple[float, float]]] 
                        // hull's lines
                    'cluster_points':
                    {
                        'x': np.array()
                        'y': np.array()
                    }
                }
            },

            change:
            {
                'hull_name':
                pd.Series('x', 'y', 'type') 
            },
            
            undraw: set(hull_name)
        }

        'labels_data':
        {
            label_id: int:
            {
                'text': str
                'x': float
                'y': float
                'arrows':
                {
                    arrow_id: int:
                    {
                        'ref_x': float
                        'ref_y': float
                        'att_x': float
                        'att_y': float
                        'val': str
                    }
                }
            },
            second_label_id: int:
            ...
            'size': float 10.0
            'arrow_size': float 1.0
        }
    }

    """

    parsed_data = parse_data(raw_data, delim)
    normalized_data = normalize(parsed_data)
    all_points = get_all_points(normalized_data)
    state_dict = editor_format(normalized_data)

    # cluster generation
    state_dict['clusters_data']['points'] = get_df_from_data(normalized_data)
    state_dict['clusters_data']['colors'] = initialize_colors(normalized_data)

    # labels generation
    labels = calc(normalized_data, all_points, config_id)
    state_dict = parse_solution_to_editor(labels, state_dict)
    state_dict["labels_data"]['size'] = 10.0
    state_dict["labels_data"]['arrow_size'] = 1.0

    # hulls generator
    state_dict['hulls_data']['render_name'] = 1
    state_dict["hulls_data"]["view_state"] = True
    set_hull_parameters(state_dict, 0.1, 20, 20)
    hulls = calc_hull(normalized_data, 0.1, 20, 20)
    state_dict = parse_solution_to_editor_hull(hulls, state_dict)
    state_dict["hulls_data"]['line_size'] = 1.0


    # ------------------------- RETURN FOR EDITOR LAUNCH ------------------------- #
    if out_path is None:
        return State(state_dict)
    # -------------------------------- END RETURN -------------------------------- #

    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)

    st = State(state_dict)
    StateLinker.link_state(st)
    st.draw(ax)

    bbox = ax.get_tightbbox().transformed(fig.dpi_scale_trans.inverted())
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches=bbox, pad_inches=0)

    return None


def draw_maps_editor(raw_data: Union[str, Experiment], delim=';') -> None:
    """ Interactive matplotlib editor for creating maps
    
    Parameters
    ----------
    raw_data: str or Experiment
        Either string with path to the csv file or an instance of mapel...Experiment
    delim: str
        If raw_data is a csv file the delim is used as input delimiter.
    
    """

    state = draw_maps(raw_data, None, delim)
    assert state is not None # editor needs init state for now (might use default cache later)

    format_string = '%(asctime)s %(filename)s %(funcName)s %(lineno)d %(message)s'
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG, format=format_string)
    editor = Editor(state)
    editor.run()
