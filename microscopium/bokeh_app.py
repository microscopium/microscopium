import os
import numpy as np
import click
from math import ceil, sqrt
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.plotting import figure, ColumnDataSource
from bokeh.layouts import row
from bokeh.models.tools import TapTool, PanTool, BoxSelectTool
from skimage import io
import pandas as pd


def imread(path):
    image0 = io.imread(path)
    if image0.shape[2] == 3:  # RGB image
        shape = image0.shape[:2]
        im1 = np.concatenate((image0,
                              np.full((shape + (1,)), 255, dtype='uint8')),
                             axis=2)
    else:  # already RGBA
        im1 = image0
    return im1


def update_image_canvas_single(index, data, source):
    index, filename = (data[['info', 'path']]
                       .iloc[index])
    image = imread(filename)
    source.data = {'image': [image],
                         'x': [0], 'y': [0], 'dx': [1], 'dy': [1]}


def update_image_canvas_multi(indices, data, source, max_images=25):
    n_images = len(indices)
    filenames = data['path'].iloc[indices]
    if n_images > max_images:
        filenames = filenames[:max_images - 1]
    images = [io.imread(fn) for fn in filenames]
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


def make_document(filename):
    dataframe = pd.read_csv(filename, index_col=0).set_index('index')
    directory = os.path.dirname(filename)
    dataframe['path'] = dataframe['url'].apply(lambda x:
                                               os.path.join(directory, x))

    def makedoc(doc):
        source = ColumnDataSource(dataframe)
        image_holder = ColumnDataSource({'image': [], 'x': [], 'y': [],
                                         'dx': [], 'dy': []})
        pca = figure(title='PCA', x_range=[-0.6, 2.7], y_range=[-1.3, 1.8],
                     sizing_mode='scale_both', tools=[TapTool(), PanTool(),
                                                      BoxSelectTool()])
        glyphs = pca.circle(source=source, x='x', y='y')

        sel = figure(title='Selected', x_range=[0, 1], y_range=[0, 1],
                     sizing_mode='scale_both')
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

        fig = row([pca, sel], sizing_mode='stretch_both')
        doc.title = 'Bokeh microscopium app'
        doc.add_root(fig)

    print('ready!')
    return makedoc


@click.command()
@click.argument('filename')
@click.option('-p', '--path', default='/')
@click.option('-P', '--port', type=int, default=5000)
def run_server(filename, path='/', port=5000):
    apps = {path: Application(FunctionHandler(make_document(filename)))}

    server = Server(apps, port=port, allow_websocket_origin=['*'])
    server.run_until_shutdown()

if __name__ == '__main__':
    run_server()
