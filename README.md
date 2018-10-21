Microscopium
============

Unsupervised clustering and dataset exploration for high content screens.

[![Build Status](https://travis-ci.org/microscopium/microscopium.svg?branch=master)](https://travis-ci.org/microscopium/microscopium)
[![Coverage Status](https://img.shields.io/coveralls/microscopium/microscopium.svg)](https://coveralls.io/r/microscopium/microscopium?branch=master)
[![Gitter](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/microscopium/microscopium?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

## See microscopium in action
![microscopium_bbbc021](https://user-images.githubusercontent.com/30920819/47262600-c2ed0c00-d538-11e8-8bd0-224ade21f8eb.gif)

Public dataset BBBC021 from the [Broad Bioimage Benchmark Collection](https://data.broadinstitute.org/bbbc/BBBC021/) with t-SNE image embedding.

## License

This project uses the 3-clause BSD license. See `LICENSE.txt`.

## For developers
We encourage pull requests - please get in touch if you think you might like to contribute.

### Serving the web app locally

To run the web app locally in your browser:
`python microscopium/serve.py tests/testdata/images/data.csv`
You should then be able to see the app in your web browser at:
http://localhost:5000

Additionally, you can specify the port number using -P
`python microscopium/serve.oy tests/testdata/images/data.csv -P 5001`

Which will run the web app locally at http://localhost:5001/

For more information, run `python microscopium/bokeh_app.py --help`

