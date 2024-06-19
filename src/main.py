import random
from typing import Optional, Union
from mapel.core.objects.Experiment import Experiment
import sys
import logging

from .generator.data_processing import *
from .editor.view_manager import *
from .editor.views import *
from .generator.labels_generator import *
from .generator.hull_generator import calc_hull, parse_solution_to_editor_hull, set_hull_parameters


def draw_maps(raw_data: Union[str, Experiment], out_path: str | None, delim=';') -> Optional[State]:
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
                    cords': list[tuple[float, float]] // coordinates of hull points
                    'line_cords': list[tuple[tuple[float, float], tuple[float, float]]] // hull's lines
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

    parsed_data = parse_data(raw_data)
    normalized_data = normalize(parsed_data)
    all_points = get_all_points(normalized_data)
    state_dict = editor_format(normalized_data)

    # cluster generation
    state_dict['clusters_data']['points'] = get_df_from_data(normalized_data)
    state_dict['clusters_data']['colors'] = initialize_colors(normalized_data)

    # todo delete prints later
    print(state_dict['clusters_data']['points'])
    print(state_dict['clusters_data']['colors'])

    # labels generation
    labels = calc(normalized_data, all_points, 10, 2)
    state_dict = parse_solution_to_editor(labels, state_dict)
    state_dict["labels_data"]['size'] = 10.0
    state_dict["labels_data"]['arrow_size'] = 1.0

    # hulls generator
    set_hull_parameters(state_dict, 2, 10, 20)
    hulls = calc_hull(normalized_data, 2, 10, 20)
    state_dict = parse_solution_to_editor_hull(hulls, state_dict)

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
    fig.savefig(out_path, bbox_inches=bbox, pad_inches=0)


def draw_maps_editor(raw_data: Union[str, Experiment], out_path: str | None, delim=';') -> None:
    """ Interactive matplotlib editor for creating maps
    
    Parameters
    ----------
    raw_data: str or Experiment
        Either string with path to the csv file or an instance of mapel...Experiment
    delim: str
        If raw_data is a csv file the delim is used as input delimiter.
    
    """
    
    state = draw_maps(raw_data, None)
    assert state is not None # editor needs init state for now (might use default cache later)

    FORMAT = '%(asctime)s %(filename)s %(funcName)s %(lineno)d %(message)s'
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG, format=FORMAT)
    editor = Editor(state)
    editor.run()
