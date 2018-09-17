from os.path import abspath, dirname, join
import click
from math import ceil, sqrt
from collections import namedtuple
from skimage import io
import numpy as np
import pandas as pd

from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.layouts import row, widgetbox, layout
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.models.widgets import Button, DataTable, TableColumn


def dataframe_from_file(filename):
    """
    """
    df = pd.read_csv(filename, index_col=0).set_index('index')
    df['path'] = df['url'].apply(lambda x: join(dirname(filename), x))
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


def _column_range(series):
    minc = np.min(series)
    maxc = np.max(series)
    rangec = maxc - minc
    column_range = namedtuple("column_range", ["minc", "maxc", "rangec"])
    return column_range(minc, maxc, rangec)


def pca_plot(source, glyph_size=1, alpha_value=0.8):
    """Display principal components analysis as bokeh scatterplotself.

    Parameters
    ----------
    source : ColumnDataSource
    glyph_size : size of scatter points, optional
    alpha_value : opacity of scatter points, optional

    Returns
    -------
    pca : bokeh figure, scatterplot of principal components analysis result
    """
    minx, maxx, rangex = _column_range(source.data['x'])
    miny, maxy, rangey = _column_range(source.data['y'])
    TOOLTIPS = [
        ("index", "$index"),
        ("info", "@info"),
        ("url", "@url")
    ]
    tools_pca = ['pan, box_select, poly_select, wheel_zoom, reset, save']
    pca = figure(title='Principal components analysis',
                 x_range=[minx - 0.05 * rangex, maxx + 0.05 * rangex],
                 y_range=[miny - 0.05 * rangey, maxy + 0.05 * rangey],
                 sizing_mode='scale_both',
                 tools=tools_pca,
                 active_drag="box_select",
                 tooltips=TOOLTIPS)
    pca.circle(source=source, x='x', y='y', size=glyph_size)
    return pca


def selected_images():
    """Create image canvas to display images from selected data.

    Returns
    -------
    selected_images : bokeh figure
    image_holder : data source to populate image figure
    """
    image_holder = ColumnDataSource({'image': [], 'x': [], 'y': [],
                                     'dx': [], 'dy': []})
    tools_sel = ['pan, box_zoom, wheel_zoom, reset, save']
    selected_images = figure(title='Selected images',
                           x_range=[0, 1],
                           y_range=[0, 1],
                           sizing_mode='scale_both',
                           tools=tools_sel,
                           active_drag='pan',
                           active_scroll='wheel_zoom')
    selected_images.image_rgba('image', 'x', 'y', 'dx', 'dy', source=image_holder)
    # Do not display axes
    selected_images.xaxis.major_tick_line_color = None
    selected_images.xaxis.minor_tick_line_color = None
    selected_images.yaxis.major_tick_line_color = None
    selected_images.yaxis.minor_tick_line_color = None
    selected_images.xaxis.major_label_text_color = None
    selected_images.yaxis.major_label_text_color = None
    return selected_images, image_holder


def button_save_table():
    """Button to save selected data table as csv.

    Notes
    -----
    * Does not work for column values containing tuples (like 'neighbors')
    * Currently columns being saved are hard coded in the javascript callback
    * Available styles: 'default', 'primary', 'success', 'warning', 'danger'
    """
    button = Button(label="Download selected data", button_type="success")
    button.callback = CustomJS(args=dict(source=table.source),
                               code=open(join(dirname(__file__), "download_data_test.js")).read())
    return widgetbox(button)


def button_print_page():
    """Button to print currently displayed webpage to paper or pdf.

    Notes
    -----
    * Available styles: 'default', 'primary', 'success', 'warning', 'danger'
    """
    button = Button(label="Print this page", button_type="primary")
    button.callback = CustomJS(args=dict(source=table.source),
                               code="""print()""")
    return widgetbox(button)


def full_table(df):
    """Display the entire dataset table (minus 'neighbors' and 'path' cols)"""
    columns = [TableColumn(field=col, title=col) for col in df.columns if col not in ['neighbors', 'path']]
    df_table = df.drop(columns=['neighbors', 'path'])
    table_source = ColumnDataSource(df_table)
    table = DataTable(source=table_source, columns=columns, width=1200)
    return table


def empty_table(df):
    """Display an empty table, with column headings (minus 'neighbors' & 'path')"""
    column_names = [col for col in df.columns if col not in ['neighbors', 'path']]
    table_source = ColumnDataSource(pd.DataFrame(columns=column_names))
    columns = [TableColumn(field=col, title=col) for col in column_names]
    table = DataTable(source=table_source, columns=columns, width=800)
    return table


def update_table(indices, df, table):
    """Update table values to show only the currently selected data."""
    # javascript csv download can't handle tuple objects in dataframe columns
    column_names = [col for col in df.columns if col not in ['neighbors', 'path']]
    filtered_df = df[column_names].iloc[indices]
    table.source.data = ColumnDataSource.from_df(filtered_df)


def load_selected(attr, old, new):
    """Update images and table to display selected data."""
    print('new index: ', new.indices)
    # Update images
    if len(new.indices) == 1:  # could be empty selection
        update_image_canvas_single(new.indices[0], data=df,
                                   source=image_holder)
    elif len(new.indices) > 1:
        update_image_canvas_multi(new.indices, data=df,
                                  source=image_holder)
    # Update table
    update_table(new.indices, df, table)


# User input - absolute filepath to csv data summary
filename = join(dirname(__file__), '../../tests/testdata/images/data.csv')
# Set up data sources, plots and tables
df = dataframe_from_file(filename)
source = ColumnDataSource(df)
pca = pca_plot(source, glyph_size=10)
image_plot, image_holder = selected_images()
table = empty_table(df)
controls = [button_save_table(), button_print_page()]
# Callbacks
source.on_change('selected', load_selected)
# Page layout
page_content = layout([
   [pca, image_plot],
   controls,
   [table]
   ], sizing_mode="scale_width")
curdoc().add_root(page_content)
curdoc().title = 'microscopium'
