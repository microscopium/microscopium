"""This module runs the bokeh server."""

import os
from os.path import dirname, join
from math import ceil, sqrt
from collections import namedtuple

import click
from skimage import io
import numpy as np
import pandas as pd

from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.plotting import figure
from bokeh.layouts import widgetbox, layout
from bokeh.models import (ColumnDataSource,
                          CustomJS,
                          CDSView,
                          GroupFilter,
                          Legend,
                          RadioButtonGroup)
from bokeh.models.widgets import Button, DataTable, TableColumn
import bokeh.palettes

from .config import load_config, get_tooltips


def dataframe_from_file(filename, image_column='url'):
    """Read in pandas dataframe from filename."""
    df = pd.read_csv(filename, index_col=0)
    df['path'] = df[image_column].apply(lambda x: join(dirname(filename), x))
    return df


def prepare_xy(source, settings):
    default_embedding = settings['embeddings']['default']
    embedding_x = settings['embeddings'][default_embedding]['x']
    embedding_y = settings['embeddings'][default_embedding]['y']
    source.add(source.data[embedding_x], name='x')
    source.add(source.data[embedding_y], name='y')


def source_from_dataframe(dataframe, settings, current_selection):
    """"""
    source = ColumnDataSource(dataframe)
    embeddings_names = list(settings['embeddings'].keys())
    selected_name = embeddings_names[current_selection]
    selected_column_x = settings['embeddings'][selected_name][0]
    selected_column_y = settings['embeddings'][selected_name][1]
    # Create empty columns to put selected coordinate data into
    source.add(dataframe[selected_column_x], name='x')
    source.add(dataframe[selected_column_y], name='y')
    return source


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
    filename = data['path'].iloc[index]
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
    margin = 0.05 * step_size / 2
    source.data = {'image': images,
                   'x': start_xs.ravel()[:n_rows] + margin,
                   'y': start_ys.ravel()[:n_rows] + margin,
                   'dx': step_sizes * 0.95, 'dy': step_sizes * 0.95}


def _dynamic_range(fig, range_padding=0.05, range_padding_units='percent'):
    """Automatically rescales figure axes range when source data changes."""
    fig.x_range.range_padding = range_padding
    fig.x_range.range_padding_units = range_padding_units
    fig.y_range.range_padding = range_padding
    fig.y_range.range_padding_units = range_padding_units
    return fig


def _palette(num, type='categorical'):
    """Return a suitable palette for the given number of categories."""
    if type == 'categorical':
        if num in range(0, 3):
            return bokeh.palettes.Colorblind[3][:num]
        if num in range(3, 9):
            return bokeh.palettes.Colorblind[num]
        if num in range(9, 13):
            return bokeh.palettes.Set3[num]
        else:
            return bokeh.palettes.viridis(num)
    else:  # numerical
        return bokeh.palettes.viridis(num)


def embedding(source, settings):
    """Display a 2-dimensional embedding of the images.

    Parameters
    ----------
    source : ColumnDataSource
    settings : dictionary

    Returns
    -------
    embed : bokeh figure
        Scatterplot of precomputed x/y coordinates result
    """
    glyph_size = settings['plots']['glyph_size']
    tools_scatter = ['pan, box_select, poly_select, wheel_zoom, reset, tap']
    embed = figure(title='Embedding',
                   sizing_mode='scale_both',
                   tools=tools_scatter,
                   active_drag="box_select",
                   active_scroll='wheel_zoom',
                   tooltips=get_tooltips(settings),
                   output_backend='webgl')
    embed = _dynamic_range(embed)
    color_column = settings['color-columns']['categorical'][0]
    if color_column in source.data:
        group_names = pd.Series(source.data[color_column]).unique()
        my_colors = _palette(len(group_names))
        for i, group in enumerate(group_names):
            group_filter = GroupFilter(column_name=color_column, group=group)
            view = CDSView(source=source, filters=[group_filter])
            glyphs = embed.circle(x="x", y="y",
                                  source=source, view=view, size=glyph_size,
                                  color=my_colors[i], legend=group)
        embed.legend.location = "top_right"
        embed.legend.click_policy = "hide"
        embed.legend.background_fill_alpha = 0.5
    else:
        embed.circle(source=source, x='x', y='y', size=glyph_size)
    return embed


def _remove_axes_spines(plot):
    plot.xaxis.major_tick_line_color = None
    plot.xaxis.minor_tick_line_color = None
    plot.yaxis.major_tick_line_color = None
    plot.yaxis.minor_tick_line_color = None
    plot.xaxis.major_label_text_color = None
    plot.yaxis.major_label_text_color = None


def selected_images():
    """Create image canvas to display images from selected data.

    Returns
    -------
    selected_images : bokeh figure
    image_holder : data source to populate image figure
    """
    image_holder = ColumnDataSource({'image': [],
                                     'x': [], 'y': [],
                                     'dx': [], 'dy': []})
    tools_sel = ['pan, box_zoom, wheel_zoom, reset']
    selected_images = figure(title='Selected images',
                             x_range=[0, 1],
                             y_range=[0, 1],
                             sizing_mode='scale_both',
                             tools=tools_sel,
                             active_drag='pan',
                             active_scroll='wheel_zoom')
    selected_images.image_rgba('image', 'x', 'y', 'dx', 'dy',
                               source=image_holder)
    _remove_axes_spines(selected_images)
    return selected_images, image_holder


def button_save_table(table):
    """Button to save selected data table as csv.

    Notes
    -----
    * Does not work for column values containing tuples (like 'neighbors')
    * Currently columns being saved are hard coded in the javascript callback
    * Available styles: 'default', 'primary', 'success', 'warning', 'danger'
    """
    button = Button(label="Download selected data", button_type="success")
    button.callback = CustomJS(args=dict(source=table.source),
                               code=open(join(dirname(__file__),
                                              "js/download_data.js")).read())
    return widgetbox(button)


def button_print_page():
    """Button to print currently displayed webpage to paper or pdf.

    Notes
    -----
    * Available styles: 'default', 'primary', 'success', 'warning', 'danger'
    """
    button = Button(label="Print this page", button_type="success")
    button.callback = CustomJS(code="""print()""")
    return widgetbox(button)


def empty_table(df):
    """Display an empty table with column headings."""
    table_source = ColumnDataSource(pd.DataFrame(columns=df.columns))
    columns = [TableColumn(field=col, title=col) for col in df.columns]
    table = DataTable(source=table_source, columns=columns, width=800)
    return table


def update_table(indices, df, table):
    """Update table values to show only the currently selected data."""
    filtered_df = df.iloc[indices]
    table.source.data = ColumnDataSource(filtered_df).data


def switch_embeddings_button_group(settings):
    """Create radio button group for switching between UMAP, tSNE, and PCA."""
    default_embedding = settings['embeddings']['default']
    del settings['embeddings']['default']
    button_labels = list(settings['embeddings'].keys())
    default_embedding_idx = button_labels.index(default_embedding)
    radio_button_group = RadioButtonGroup(labels=button_labels,
                                          active=default_embedding_idx)
    return radio_button_group


def update_embedding(source, embedding, settings):
    """Update source of image embedding scatterplot."""
    embeddings = settings['embeddings']
    x_source = embeddings[embedding]['x']
    y_source = embeddings[embedding]['y']
    source.data['x'] = source.data[x_source]
    source.data['y'] = source.data[y_source]
    source.trigger("data", 0, 0)


def reset_plot_axes(plot, x_start=0, x_end=1, y_start=0, y_end=1):
    plot.x_range.start = x_start
    plot.x_range.end = x_end
    plot.y_range.start = y_start
    plot.y_range.end = y_end


def make_makedoc(filename, settings_filename):
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
    settings_filename: string
        Path to a yaml file 

    Returns
    -------
    makedoc : function
        A makedoc function as expected by ``FunctionHandler``.
    """
    settings = load_config(settings_filename)
    dataframe = dataframe_from_file(filename,
                                    image_column=settings['image-column'])

    def makedoc(doc):
        source = ColumnDataSource(dataframe)
        prepare_xy(source, settings)  # get the default embedding columns
        embed = embedding(source, settings)
        image_plot, image_holder = selected_images()
        table = empty_table(dataframe)
        controls = [button_save_table(table), button_print_page()]
        radio_buttons = switch_embeddings_button_group(settings)

        def load_selected(attr, old, new):
            """Update images and table to display selected data."""
            print('new index: ', new)
            # Update images & table
            if len(new) == 1:  # could be empty selection
                update_image_canvas_single(new[0], data=dataframe,
                                           source=image_holder)
            elif len(new) > 1:
                update_image_canvas_multi(new, data=dataframe,
                                          source=image_holder)
            reset_plot_axes(image_plot)  # effectively resets zoom level
            update_table(new, dataframe, table)

        def new_embedding(attr, old, new):
            embedding = list(settings['embeddings'])[radio_buttons.active]
            update_embedding(source, embedding, settings)

        source.selected.on_change('indices', load_selected)
        radio_buttons.on_change('active', new_embedding)

        page_content = layout([
            radio_buttons,
            [embed, image_plot],
            controls,
            [table]
            ], sizing_mode="scale_width")
        doc.title = 'Bokeh microscopium app'
        doc.add_root(page_content)
    print('ready!')
    return makedoc


def default_config(filename):
    d = os.path.dirname(filename)
    return os.path.join(d, 'settings.yaml')


@click.command()
@click.argument('filename')
@click.option('-c', '--config', default=None)
@click.option('-p', '--path', default='/')
@click.option('-P', '--port', type=int, default=5000)
@click.option('-u', '--url', default='http://localhost')
def run_server_cmd(filename, config=None, path='/', port=5000,
                   url='http://localhost'):
    run_server(filename, config=config, path=path, port=port, url=url)


def run_server(filename, config=None, path='/', port=5000,
               url='http://localhost'):
    """Run the bokeh server."""
    if config is None:
        config = default_config(filename)
    makedoc = make_makedoc(filename, config)
    apps = {path: Application(FunctionHandler(makedoc))}
    server = Server(apps, port=port, allow_websocket_origin=['*'])
    print('Web app now available at {}:{}'.format(url, port))
    server.run_until_shutdown()


if __name__ == '__main__':
    run_server_cmd()
