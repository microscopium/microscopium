"""Temporary file to demonstrate the sample dataset.
"""

from pymongo import MongoClient
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
import subprocess as sp
import itertools as it

# start mongodb daemon
sp.Popen(["mongod", "--dbpath", "./mongodb", "--port", "27020",
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
imgs = []
for i in range(0, cursor.count()):
    doc = next(cursor, None)
    print doc
    key = doc['_id']
    gene_name = doc['gene_name']
    file_name = doc['filename']
    file_name = './images/' + file_name.split('/')[5]
    img = io.imread(file_name)
    imgs.append(img)
    titles.append(' '.join([key, gene_name]))

# unpickle dataframe, show first 5 rows
test_data = pd.read_pickle('data_test.p')
print test_data.head()

# display all images

fig, axes = plt.subplots(2, 4)

for i in it.product([0, 1], [0, 1, 2, 3]):
    axes[i].imshow(imgs.pop())
    axes[i].set_title(titles.pop())
    axes[i].axis('off')

plt.show()
