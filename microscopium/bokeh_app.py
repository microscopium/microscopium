import os
import numpy as np
import click
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.plotting import figure, ColumnDataSource
from bokeh.layouts import row
from bokeh.models.tools import TapTool
from skimage import io
import pandas as pd


def imread(path):
    image0 = io.imread(path)
    shape = image0.shape[:2]
    im1 = np.concatenate((image0, np.full((shape + (1,)), 255, dtype='uint8')),
                         axis=2)
    return im1


def make_document(filename):
    dataframe = pd.read_csv(filename, index_col=0).set_index('index')
    directory = os.path.dirname(filename)
    dataframe['path'] = dataframe['url'].apply(lambda x:
                                               os.path.join(directory, x))
    print(dataframe.head())

    def makedoc(doc):
        source = ColumnDataSource(dataframe)
        pca = figure(title='PCA', x_range=[-0.6, 2.7], y_range=[-1.3, 1.8],
                     sizing_mode='scale_both')
        glyphs = pca.circle(source=source, x='x', y='y')

        sel = figure(title='selected', x_range=[0, 1], y_range=[0, 1],
                     sizing_mode='scale_both')
        image0 = imread(dataframe['path'].iloc[0])
        sel.image_rgba([image0], 0, 0, 1, 1)

        def load_image(attr, old, new):
            print(attr)
            print(old)
            print(new)

        glyphs.data_source.on_change('selected', load_image)

        fig = row([pca], sizing_mode='stretch_both')
        doc.title = 'Bokeh microscopium app'
        doc.add_root(fig)

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
