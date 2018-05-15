import os
import numpy as np
import click
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.plotting import figure, ColumnDataSource
from bokeh.layouts import row
from bokeh.models.tools import TapTool, PanTool
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


def make_document(filename):
    dataframe = pd.read_csv(filename, index_col=0).set_index('index')
    directory = os.path.dirname(filename)
    dataframe['path'] = dataframe['url'].apply(lambda x:
                                               os.path.join(directory, x))

    def makedoc(doc):
        source = ColumnDataSource(dataframe)
        image_holder = ColumnDataSource({'image': []})
        pca = figure(title='PCA', x_range=[-0.6, 2.7], y_range=[-1.3, 1.8],
                     sizing_mode='scale_both', tools=[TapTool(), PanTool()])
        glyphs = pca.circle(source=source, x='x', y='y')

        sel = figure(title='Selected', x_range=[0, 1], y_range=[0, 1],
                     sizing_mode='scale_both')
        image_canvas = sel.image_rgba('image', 0, 0, 1, 1, source=image_holder)

        def load_image(attr, old, new):
            print('new index: ', new.indices)
            if len(new.indices) > 0:  # could be empty selection
                index, filename = (dataframe[['info', 'path']]
                                   .iloc[new.indices[0]])
                image = imread(filename)
                image_holder.data = {'image': [image]}

        glyphs.data_source.on_change('selected', load_image)

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

    server = Server(apps, port=port)
    server.run_until_shutdown()

if __name__ == '__main__':
    run_server()
