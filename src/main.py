"""
Module with main exported functions of the package
"""

import sys
import logging
from typing import Optional, Union
import os

import matplotlib.pyplot as plt
from mapel.core.objects.Experiment import Experiment

from .configuration import Configuration
from .generator.data_processing import parse_data, normalize, get_all_points
from .generator.data_processing import editor_format, get_df_from_data, initialize_colors
from .editor.view_manager import State, StateLinker
from .editor.view_editor import Editor
from .generator.labels_generator import calc, parse_solution_to_editor
from .generator.hull_generator import calc_hull, parse_solution_to_editor_hull, set_hull_parameters

# disabling excessive Pillow logging
logging.getLogger('PIL').setLevel(logging.WARNING)


def draw_maps(raw_data: str,
              out_path: str | None,
              delim=';',
              config_id: str = 'divide_and_conquare') -> Optional[State]:
    """ Automatic map creator
    
    Parameters
    ----------
    raw_data: str
        String with path to the csv file
    out_path: str or None
        Path for the .png image to be saved to. If None raw drawer data is returned
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
    """

    parsed_data = parse_data(raw_data, delim)
    normalized_data = normalize(parsed_data)
    all_points = get_all_points(normalized_data)
    state_dict = editor_format(normalized_data)

    # cluster generation
    state_dict['clusters_data']['points'] = get_df_from_data(normalized_data)
    colors = Configuration["editor"]["colors"]
    state_dict['clusters_data']['colors'] = colors

    # labels generation
    labels = calc(normalized_data, all_points, config_id)
    state_dict = parse_solution_to_editor(labels, state_dict)
    state_dict["labels_data"]['size'] = 10.0
    state_dict["labels_data"]['arrow_size'] = 1.0

    # hulls generator
    state_dict['hulls_data']['render_name'] = 1
    state_dict["hulls_data"]["view_state"] = True
    
    set_hull_parameters(state=state_dict, 
                        circle_radious=0.1,
                        points_in_circle=20,
                        segment_length=10,
                        domain_expansion=1.5,
                        closest_points_radius=1)
    
    if Configuration['global']['generate_hulls']:
        hulls = calc_hull(normalized_data, 0.1, 20, 20)
    else:
        hulls = {}
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
    st.draw(ax, True)

    bbox = ax.get_tightbbox().transformed(fig.dpi_scale_trans.inverted())
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches=bbox, pad_inches=0, dpi=Configuration['global']['dpi'])

    return None


def draw_maps_editor(raw_data: str, delim=';', config_id='iterative') -> None:
    """ Interactive matplotlib editor for creating maps
    
    Parameters
    ----------
    raw_data: str or Experiment
        Either string with path to the csv file or an instance of mapel...Experiment
    delim: str
        If raw_data is a csv file the delim is used as input delimiter.
    config_id: str
        String id specifying a group of parameters in the Configuration class which 
        affect auto generation
    """

    state = draw_maps(raw_data, None, delim, config_id=config_id)
    assert state is not None # editor needs init state

    format_string = '%(asctime)s %(filename)s %(funcName)s %(lineno)d %(message)s'
    logging.basicConfig(stream=sys.stderr, level=logging.CRITICAL, format=format_string) # change level for debugging information
    editor = Editor(state)
    editor.run()
