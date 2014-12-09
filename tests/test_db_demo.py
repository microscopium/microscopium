'''Temporary file to demonstrate the sample dataset.
'''

from pymongo import MongoClient
import pandas as pd
from skimage import io
import subprocess as sp
import time
import os

abspath = os.path.dirname(__file__)

def string2tuple(string_tuple):
    # TODO add docstring
    string_values = string_tuple.split(', ')
    coords = (int(string_values[0][1:]), string_values[1][1:-2])
    return coords


# start mongodb daemon

client = MongoClient('localhost', 27017)
db = client['myofusion_test']
collection = db.wells_test

# import documents if collection is empty
# wait two seconds to allow collection to import
if db.wells_test.find({}).count() == 0:
    sp.Popen(['mongoimport', '-host', 'localhost:27017', '-d',
              'myofusion_test', '-c', 'wells_test', os.path.join(abspath, 'testdata/wells_test.json')])
    time.sleep(2)

#  parse image filename and id and gene name
cursor = collection.find({})
titles = []
images = []
for doc in cursor:
    print doc
    key = doc['_id']
    gene_name = doc['gene_name']
    image_fn = doc['filename']
    image_fn = os.path.join(abspath, 'testdata/images/') + image_fn.split('/')[5]
    image = io.imread(image_fn)
    images.append(image)
    titles.append(' '.join([key, gene_name]))

# read dataframe from CSV, show first 5 rows
test_data = pd.read_csv(os.path.join(abspath, 'testdata/data_test.csv'), index_col=0,
                         converters={0: string2tuple})
print test_data.head()
