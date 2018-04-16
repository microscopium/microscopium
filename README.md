Microscopium
============

Unsupervised clustering and dataset exploration for high-content screens.

## Introduction
`Microscopium` is an analysis platform for studying high-content screens. The data are a set microscopy images of MCF-7 human breast cancer cells, furnished by the Broad Institute, available [here](https://data.broadinstitute.org/bbbc/BBBC021/). Each image (or sample) has been treated by one of 113 compound treatments, where each of the compounds have up to eight levels of concentrations. Each sample is then annotated with a *mechanism-of-action* (MOA).  Our goal, then, is to characterize the MOA in terms of 246 characteristics (or *features*) available in `Microscopium`; examples of features are "number of nuclei", "cell density", colour distribution statistics, and so on. 

## Clustering & Data Visualization
After profiling each sample in terms of `Microscopium`'s features, we use an unsupervised clustering algorithm, such as K-means, to group samples by similarity. After this, we use a dimensionality-reduction technique such as PCA or t-SNE to map these groupings on to two dimensions, so that humans can make a visual assessment of these groupings. The visualizations are available [here](http://play.microscopium.io/).

[![Build Status](https://travis-ci.org/microscopium/microscopium.svg?branch=master)](https://travis-ci.org/microscopium/microscopium)
[![Coverage Status](https://img.shields.io/coveralls/microscopium/microscopium.svg)](https://coveralls.io/r/microscopium/microscopium?branch=master)
[![Gitter](https://badges.gitter.im/Join Chat.svg)](https://gitter.im/microscopium/microscopium?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

# License

This project uses the 3-clause BSD license. See `LICENSE.txt`.
