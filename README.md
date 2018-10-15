Microscopium
============

Unsupervised clustering and dataset exploration for high content screens.

[![Build Status](https://travis-ci.org/microscopium/microscopium.svg?branch=master)](https://travis-ci.org/microscopium/microscopium)
[![Coverage Status](https://img.shields.io/coveralls/microscopium/microscopium.svg)](https://coveralls.io/r/microscopium/microscopium?branch=master)
[![Gitter](https://badges.gitter.im/Join Chat.svg)](https://gitter.im/microscopium/microscopium?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

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
`python microscopium/bokeh_app.oy tests/testdata/images/data.csv -P 5001`

Which will run the web app locally at http://localhost:5001/

For more information, run `python microscopium/bokeh_app.py --help`

