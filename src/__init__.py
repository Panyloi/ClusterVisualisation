import matplotlib
from .main import draw_maps, draw_maps_editor
from .configuration import Configuration

# init global config
Configuration()
Configuration.load()

matplotlib.use('TKAgg', force=True)

__all__ = ['draw_maps', 'draw_maps_editor']
__doc__ = """Subpackage for visualization of maps of elections"""
