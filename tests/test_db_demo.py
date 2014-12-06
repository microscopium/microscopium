"""Temporary file to demonstrate the sample dataset.
"""

from pymongo import MongoClient
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
import subprocess as sp

# start mongodb daemon
sp.Popen(["mongod", "--dbpath", "./testdata/mongodb", "--port", "27020",
          "--smallfiles"])

client = MongoClient('localhost', 27020)
db = client['myofusion_test']
collection = db.wells_test

# import documents if collection is empty
if db.wells_test.find({}).count() == 0:
    sp.Popen(["mongoimport", "-host", "localhost:27020", "-d",
              "myofusion_test", "-c", "wells_test", "wells_test.json"])

#  parse image filename and id and gene name
cursor = collection.find({})
titles = []
images = []
for doc in cursor:
    print doc
    key = doc['_id']
    gene_name = doc['gene_name']
    image_fn = doc['filename']
    image_fn = './testdata/images/' + image_fn.split('/')[5]
    image = io.imread(image_fn)
    images.append(image)
    titles.append(' '.join([key, gene_name]))

# unpickle dataframe, show first 5 rows
test_data = pd.read_pickle('./testdata/data_test.pickle')
print test_data.head()

# display all images
fig, axes = plt.subplots(2, 4)

for ax in axes.ravel():
    ax.imshow(images.pop())
    ax.set_title(titles.pop())
    ax.axis('off')

plt.show()
