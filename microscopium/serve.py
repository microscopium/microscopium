"""This module runs the bokeh server."""

from os.path import dirname, join
from math import ceil, sqrt
from collections import namedtuple

import click
from skimage import io
import numpy as np
import pandas as pd
import matplotlib.colors
import matplotlib.cm
import ipyvolume as ipv
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
                          TapTool)
from bokeh.models.widgets import Button, DataTable, TableColumn
import bokeh.palettes


def dataframe_from_file(filename):
    """Read in pandas dataframe from filename."""
    df = pd.read_csv(filename, index_col=0).set_index('index')
    df['path'] = df['url'].apply(lambda x: join(dirname(filename), x))
    valid_x = df.x.notna()
    valid_y = df.y.notna()
    df = df[valid_x & valid_y]
    return df


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
    if image0.ndim == 4:
        image0 = np.moveaxis(image0, 1, -1)  # expect (pln, row, col, ch)
    shape = image0.shape[:-1]
    while image0.shape[-1] < 3:
        blank_channel = np.zeros((shape + (1,)), dtype='uint8')
        image0 = np.concatenate((image0, blank_channel), axis=-1)
    if image0.shape[-1] == 3:  # RGB image
        shape = image0.shape[:-1]
        alpha = np.full((shape + (1,)), 255, dtype='uint8')
        image = np.concatenate((image0, alpha), axis=-1)
    else:  # already RGBA
        image = image0
    return image


def maximum_intensity_projection(data):
    """Maximum intensity projection
     Parameters
    -----------
    data : ndarray
        Image volume, 3 spatial dimensions, plus optional channel dimensions
     Returns
    -------
    max_projection : ndarray
        Maximum intensity projection of image volume, 2D image plus channels
    """
    max_projection = np.array([np.max(data[..., channel], axis=0)
                               for channel in range(data.shape[-1])])
    max_projection = np.moveaxis(max_projection, 0, -1)
    return max_projection


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
    if image.ndim >= 4:
        image = maximum_intensity_projection(image)
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
    if images[0].ndim > 3:
        images = [maximum_intensity_projection(image) for image in images]
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


def linear_transfer_function(color,
                             min_opacity=0,
                             max_opacity=0.05,
                             reverse_opacity=False,
                             n_elements = 256):
    """Return ipyvolume transfer function of a single color and linear opacity.
    Parameters
    ----------
    color : Listlike RGB, or string with hexidecimal or named color.
        RGB values should be within 0-1 range.
    min_opacity : Minimum opacity, default value is 0.0.
        Lowest possible value is 0.0, optional. Float or integer value.
    max_opacity : Maximum opacity, default value is 0.05.
        Highest possible value is 1.0, optional. Float or integer value.
    reverse_opacity : Linearly decrease opacity, optional. Boolean value.
    n_elements : Integer length of rgba array transfer function attribute.
    Returns
    -------
    transfer_function : ipyvolume TransferFunction
    Example
    -------
    >>> import ipyvolume as ipv
    >>> green_tf = linear_transfer_function('green')
    >>> ds = ipv.datasets.aquariusA2.fetch()
    >>> ipv.volshow(ds.data[::4,::4,::4], tf=green_tf)
    >>> ipv.show()
    """
    r, g, b = matplotlib.colors.to_rgb(color)
    opacity = np.linspace(min_opacity, max_opacity, num=n_elements)
    if reverse_opacity:
        opacity = np.flip(opacity, axis=0)
    rgba = np.transpose(np.stack([[r] * n_elements,
                                  [g] * n_elements,
                                  [b] * n_elements,
                                  opacity]))
    transfer_function = ipv.transferfunction.TransferFunction(rgba=rgba)
    return transfer_function


def transfer_functions(colors):
    """Create dict of linear ipyvolume transfer functions for specified colors.
    Parameters
    ----------
    colors : List of colors in form accepted by matplotlib
        i.e. List containing elements of any of the following:
        1. hexidecimal strings,
        2. named colors,
        3. listlike RGB values. RGB values should be within 0-1 range.
    Returns
    -------
    tf_dict : dictionary of ipyvolume transfer functions.
        key, val = str(colorname), ipyvolume transfer function.
    """
    transfer_functions = [linear_transfer_function(color) for color in colors]
    return transfer_functions


def volume_rendering(image_filename, image_info, url, transfer_functions):
    """3D volume rendering saved to html file with ipyvolume.
    Parameters
    ----------
    index :
    data :
    url :
    transfer_functions :
    Returns
    -------
    """
    print('Volume rendering...')
    ipv.pylab.clear()
    image_3d = io.imread(image_filename)
    image_3d = np.moveaxis(image_3d, 1, 0)
    print(image_3d.shape)
    #possible_colors = ['red', 'green', 'blue', 'grey', 'cyan', 'magenta', 'yellow']
    #colors = possible_colors[:image_3d.shape[0]]
    # should make transfer functions if none passed in
    fig = ipv.figure()
    fig.vol = None
    ipv.pylab.style.box_off()
    ipv.pylab.style.axes_off()
    for channel, transfer_function in zip(image_3d, transfer_functions):
        ipv.volshow(channel, tf=transfer_function)
    ipv.embed.embed_html('./tmp/'+url, ipv.gcc(), title=image_info)


def _column_range(series):
    minc = np.min(series)
    maxc = np.max(series)
    rangec = maxc - minc
    column_range = namedtuple("column_range", ["minc", "maxc", "rangec"])
    return column_range(minc, maxc, rangec)


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


def embedding(source, glyph_size=1, color_column='group'):
    """Display a 2-dimensional embedding of the images.

    Parameters
    ----------
    source : ColumnDataSource
    glyph_size : size of scatter points, optional
    color_column : str
        Name of column in `source` to represent with color

    Returns
    -------
    embed : bokeh figure
        Scatterplot of precomputed x/y coordinates result
    """
    minx, maxx, rangex = _column_range(source.data['x'])
    miny, maxy, rangey = _column_range(source.data['y'])
    tooltips_scatter = [
        ("index", "$index"),
        ("info", "@info"),
        ("url", "@url")
    ]
    tools_scatter = ['pan, box_select, poly_select, wheel_zoom, reset, tap']
    embed = figure(title='Embedding',
                   x_range=[minx - 0.05 * rangex, maxx + 0.05 * rangex],
                   y_range=[miny - 0.05 * rangey, maxy + 0.05 * rangey],
                   sizing_mode='scale_both',
                   tools=tools_scatter,
                   active_drag="box_select",
                   tooltips=tooltips_scatter)
    if color_column in source.data:
        group_names = pd.Series(source.data[color_column]).unique()
        my_colors = _palette(len(group_names))
        for i, group in enumerate(group_names):
            group_filter = GroupFilter(column_name=color_column, group=group)
            view = CDSView(source=source, filters=[group_filter])
            glyphs = embed.circle(x="x", y="y", source=source, view=view,
                                  size=10, color=my_colors[i], legend=group)
        embed.legend.location = "top_right"
        embed.legend.click_policy = "hide"
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
    image_holder = ColumnDataSource({'image': [], 'x': [], 'y': [],
                                     'dx': [], 'dy': []})
    tools_sel = ['pan, box_zoom, wheel_zoom, reset']
    selected_images = figure(title='Selected images',
                             x_range=[0, 1],
                             y_range=[0, 1],
                             sizing_mode='scale_both',
                             tools=tools_sel,
                             active_drag='pan',
                             active_scroll='wheel_zoom')
    selected_images.image_rgba('image', 'x', 'y', 'dx', 'dy', source=image_holder)
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


def make_makedoc(filename, color_column=None):
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
    color_column : string, optional
        Which column in the CSV to use to color points in the embedding.

    Returns
    -------
    makedoc : function
        A makedoc function as expected by ``FunctionHandler``.
    """
    dataframe = dataframe_from_file(filename)
    volume_rendering_transfer_functions = transfer_functions(['red', 'green'])

    def makedoc(doc):
        source = ColumnDataSource(dataframe)
        embed = embedding(source, glyph_size=10, color_column=color_column)
        taptool = embed.select(TapTool)[0]
        image_plot, image_holder = selected_images()
        table = empty_table(dataframe)
        controls = [button_save_table(table), button_print_page()]
        url_base = "volume_preview.html"
        url = "http://localhost:5007/" + url_base

        def load_selected(attr, old, new):
            """Update images and table to display selected data."""
            print('new index: ', new.indices)
            # Update images & table
            if len(new.indices) == 1:  # could be empty selection
                update_image_canvas_single(new.indices[0], data=dataframe,
                                           source=image_holder)
                image_filename = dataframe['path'].iloc[new.indices[0]]
                image_info = dataframe['info'].iloc[new.indices[0]]
                volume_rendering(image_filename, image_info, url_base,
                                 volume_rendering_transfer_functions)
            elif len(new.indices) > 1:
                update_image_canvas_multi(new.indices, data=dataframe,
                                          source=image_holder)
            update_table(new.indices, dataframe, table)
        source.on_change('selected', load_selected)

        tap_callback = CustomJS(args=dict(url=url), code="""
            function myFunction() {
                window.open(url);
            }
            console.log(url)
            setTimeout(myFunction, 2000);
            //var columns = Object.keys(source.data);
            //var nrows = source.data[columns[0]].length;
            //console.log(nrows);
            //if (nrows == 0) {
            //    setTimeout(myFunction, 2000);
            //}
            """)
        taptool.callback = tap_callback

        page_content = layout([
            [embed, image_plot],
            controls,
            [table]
            ], sizing_mode="scale_width")
        doc.title = 'Bokeh microscopium app'
        doc.add_root(page_content)
    print('ready!')
    return makedoc


@click.command()
@click.argument('filename')
@click.option('-p', '--path', default='/')
@click.option('-P', '--port', type=int, default=5000)
@click.option('-c', '--color-column', default='group')
def run_server(filename, path='/', port=5000, color_column='group'):
    """Run the bokeh server."""
    makedoc = make_makedoc(filename, color_column=color_column)
    apps = {path: Application(FunctionHandler(makedoc))}

    server = Server(apps, port=port, allow_websocket_origin=['*'])
    server.run_until_shutdown()


if __name__ == '__main__':
    run_server()
