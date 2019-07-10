Microscopium
============

Unsupervised clustering and dataset exploration for high content screens.

[![Build Status](https://travis-ci.org/microscopium/microscopium.svg?branch=master)](https://travis-ci.org/microscopium/microscopium)
[![Coverage Status](https://img.shields.io/coveralls/microscopium/microscopium.svg)](https://coveralls.io/r/microscopium/microscopium?branch=master)
[![Gitter](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/microscopium/microscopium?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

## See microscopium in action
![microscopium_bbbc021](https://user-images.githubusercontent.com/30920819/47262600-c2ed0c00-d538-11e8-8bd0-224ade21f8eb.gif)

Public dataset BBBC021 from the [Broad Bioimage Benchmark Collection](https://data.broadinstitute.org/bbbc/BBBC021/) with t-SNE image embedding.


## For developers
We encourage pull requests - please get in touch if you think you might like to contribute.


### License

This project uses the 3-clause BSD license. See `LICENSE.txt`.

### Development installation

First, clone this repository, and change into its directory:
```
git clone https://github.com/microscopium/microscopium.git
cd microscopium
```

Then, install the dependencies via one of the below methods

#### conda, new environment (recommended)

```
conda env create -f environment.yml
conda activate mic
```

#### conda, existing environment

```
# conda activate <env-name>
conda install -f environment.yml
```

#### pip

```
pip install -r requirements.txt
```

Finally, install microscopium, optionally as an editable package:

```
pip install [-e] .
```

### Serving the web app

Supported browsers are Chrome and Firefox. However we have observed that performance is much better on Chrome.
(Unfortunately, we do not currently support Safari or Internet Explorer.)

Your data needs to have the following format:

- a collection of image files (can be in a directory, or in a nested directory
  structure)
- a `.csv` file containing, at a minimum, the x/y coordinates of each image,
  and the path to the image in the directory. The path should be relative to
  the location of the `.csv` file.
- a `.yaml` file containing settings. At a minimum, it should contain an
  `embeddings` field with maps from `<embedding name>` to column names for `x`
  and `y`, as well as `image-column` containing the name of the column
  containing the path to each image. If you don't want to specify the settings
  file path, place `settings.yaml` next to the `.csv` file. Microscopium will
  look here by default.

For example data, see:

- `tests/testdata/images/*.png`
- `tests/testdata/images/data.csv`
- `tests/testdata/images/settings.yaml`

To run the web app locally, try:

`python -m microscopium.serve tests/testdata/images/data.csv -c tests/testdata/images/settings.yaml`

You should then be able to see the app in your web browser at:
http://localhost:5000

You can specify a port number with `-P`.

`python -m microscopium.serve tests/testdata/images/data.csv -P 5001`

This specifies the port number as 5001, and the app will run locally at: http://localhost:5001/

For more information, run `python -m microscopium.serve --help`
