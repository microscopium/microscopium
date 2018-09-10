import os
import numpy as np
import click
from math import ceil, sqrt
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.plotting import figure, ColumnDataSource
from bokeh.layouts import row
from bokeh.models.tools import (ResetTool, PanTool, WheelZoomTool, TapTool,
                                BoxSelectTool, PolySelectTool, UndoTool,
                                RedoTool, HoverTool, BoxZoomTool)
from skimage import io
import pandas as pd


def imread(path):
    """Read an image from disk while ensuring it has an alpha channel.

    Parameters
    ----------
    path : string
        Any valid path for skimage.io. This includes a local path as well as
        a URL.

    Returns
    -------
    image : array, shape (M, N, 4)
        The resulting RGBA image.
    """
    image0 = io.imread(path)
    if image0.shape[2] == 3:  # RGB image
        shape = image0.shape[:2]
        alpha = np.full((shape + (1,)), 255, dtype='uint8')
        image = np.concatenate((image0, alpha), axis=2)
    else:  # already RGBA
        image = image0
    return image


def update_image_canvas_single(index, data, source):
    """Update image canvas when a single image is selected on the scatter plot.

    The ``ColumnDataSource`` `source` will be updated in-place, which will
    cause the ``image_rgba`` plot to automatically update.

    Parameters
    ----------
    index : string
        The index value of the selected point. This must match the index on
        `data`.
    data : DataFrame
        The image properties dataset. It must include a 'path' pointing to the
        image file for each image.
    source : ColumnDataSource
        The ``image_rgba`` data source. It must include the columns 'image',
        'x', 'y', 'dx', 'dy'.
    """
    index, filename = (data[['info', 'path']]
                       .iloc[index])
    image = imread(filename)
    source.data = {'image': [image], 'x': [0], 'y': [0], 'dx': [1], 'dy': [1]}


def update_image_canvas_multi(indices, data, source, max_images=25):
    """Update image canvas when multiple images are selected on scatter plot.

    The ``ColumnDataSource`` `source` will be updated in-place, which will
    cause the ``image_rgba`` plot to automatically update.

    Parameters
    ----------
    indices : list of string
        The index values of the selected points. These must match the index on
        `data`.
    data : DataFrame
        The image properties dataset. It must include a 'path' pointing to the
        image file for each image.
    source : ColumnDataSource
        The ``image_rgba`` data source. It must include the columns 'image',
        'x', 'y', 'dx', 'dy'.

    Notes
    -----
    Currently, we implement our own simple grid layout algorithm for the input
    images. It may or may not be better to instead use a grid of ``image_rgba``
    plots. It's unclear how we would update those, though.
    """
    n_images = len(indices)
    filenames = data['path'].iloc[indices]
    if n_images > max_images:
        filenames = filenames[:max_images - 1]
    images = [imread(fn) for fn in filenames]
    if n_images > max_images:
        # from the My First Pixel Art (TM) School of Design
        dotdotdot = np.full((7, 7, 4), 255, dtype=np.uint8)
        dotdotdot[3, 1::2, :3] = 0
        images.append(dotdotdot)
    sidelen = ceil(sqrt(min(n_images, max_images)))
    step_size = 1 / sidelen
    grid_points = np.arange(0, 1 - step_size/2, step_size)
    start_xs, start_ys = np.meshgrid(grid_points, grid_points, indexing='ij')
    n_rows = len(images)
    step_sizes = np.full(n_rows, step_size)
    source.data = {'image': images, 'x': start_xs.ravel()[:n_rows],
                   'y': start_ys.ravel()[:n_rows],
                   'dx': step_sizes, 'dy': step_sizes}
    print(source.data)


def _column_range(series):
    minc = np.min(series)
    maxc = np.max(series)
    rangec = maxc - minc
    return (minc, maxc, rangec)


def make_makedoc(filename):
    """Make the makedoc function required by Bokeh Server.

    To run a Bokeh server, we need to create a function that takes in a Bokeh
    "document" as input, and adds our figure (together with all the interactive
    bells and whistles we may want to add to it) to that document. This then
    initialises a ``FunctionHandler`` and an ``Application`` gets started by
    the server. See the `run_server` code for details.

    Parameters
    ----------
    filename : string
        A CSV file containing the data for the app.

    Returns
    -------
    makedoc : function
        A makedoc function as expected by ``FunctionHandler``.
    """
    dataframe = pd.read_csv(filename, index_col=0).set_index('index')
    directory = os.path.dirname(filename)
    dataframe['path'] = dataframe['url'].apply(lambda x:
                                               os.path.join(directory, x))
    minx, maxx, rangex = _column_range(dataframe['x'])
    miny, maxy, rangey = _column_range(dataframe['y'])

    def makedoc(doc):
        source = ColumnDataSource(dataframe)
        image_holder = ColumnDataSource({'image': [], 'x': [], 'y': [],
                                         'dx': [], 'dy': []})
        tools_pca = [ResetTool(), PanTool(), WheelZoomTool(), TapTool(),
                     BoxSelectTool(), PolySelectTool(), UndoTool(), RedoTool()]
        pca = figure(title='PCA',
                     x_range=[minx - 0.05 * rangex, maxx + 0.05 * rangex],
                     y_range=[miny - 0.05 * rangey, maxy + 0.05 * rangey],
                     sizing_mode='scale_both', tools=tools_pca)
        glyphs = pca.circle(source=source, x='x', y='y')

        tools_sel = [ResetTool(), PanTool(), WheelZoomTool(),
                     BoxZoomTool(match_aspect=True),
                     UndoTool(), RedoTool()]
        sel = figure(title='Selected image', x_range=[0, 1], y_range=[0, 1],
                     sizing_mode='scale_both', tools=tools_sel)
        image_canvas = sel.image_rgba('image', 'x', 'y', 'dx', 'dy',
                                      source=image_holder)

        def load_selected(attr, old, new):
            print('new index: ', new.indices)
            if len(new.indices) == 1:  # could be empty selection
                update_image_canvas_single(new.indices[0], data=dataframe,
                                           source=image_holder)
            elif len(new.indices) > 1:
                update_image_canvas_multi(new.indices, data=dataframe,
                                          source=image_holder)

        glyphs.data_source.on_change('selected', load_selected)

        fig = row([pca, sel])
        doc.title = 'Bokeh microscopium app'
        doc.add_root(fig)

    print('ready!')
    return makedoc


@click.command()
@click.argument('filename')
@click.option('-p', '--path', default='/')
@click.option('-P', '--port', type=int, default=5000)
def run_server(filename, path='/', port=5000):
    apps = {path: Application(FunctionHandler(make_makedoc(filename)))}

    server = Server(apps, port=port, allow_websocket_origin=['*'])
    server.run_until_shutdown()

if __name__ == '__main__':
    run_server()
