# DarkShards

[//]: [![DOI](https://zenodo.org/badge/131850900.svg)](https://zenodo.org/badge/latestdoi/131850900) 
[//]: [![arXiv](https://img.shields.io/badge/arXiv-1805.09034-B31B1B.svg)](https://arxiv.org/abs/1805.09034)
[![MIT Licence](https://badges.frapsoft.com/os/mit/mit.svg?v=103)](https://opensource.org/licenses/mit-license.php)

This repository contains the code, data and notebooks for (one hopes) reproducing the entirety of our recent paper "*Impact of the ex-situ halo on searches for dark matter*". There are also many results which did not make it in to final version but I've left in for potential future interest.

Please contact me at ciaran.aj.ohare@gmail.com if you want to winge about why the code doesn't work.

## Contents

The code, plots, datas, etc. are sorted as follows:

* `data/` - Contains the *Gaia* sample of halo stars as well as various cleaned samples of the same data and the extracted substructures and fits.
* `code/` - 
* `plots/` - all plots get put here in pdf and png formats.

## Requirements

The code is all written in python3 and makes substantial use of the standard numpy, matplotlib, scipy etc. There are several additonal libraries that you may need to investigate depending on your installation:

* [`astropy`](https://www.astropy.org/), used for various things and also required by ...
* [`galpy`](https://galpy.readthedocs.io/en/v1.4.0/), used for computing stellar orbits
* [`scikit-learn`](https://scikit-learn.org/stable/), used in fitting the shards and doing kernel density estimates for action space variables
* [`cmocean`](https://matplotlib.org/cmocean/), nice aesthetic colormaps that don't just look like a unicorn's vomit
* [`cartopy`](https://scitools.org.uk/cartopy/docs/latest/), used to make Mollweide skymaps
* [`healpy`](https://healpy.readthedocs.io/en/latest/), can't remember what this was used for but it's there


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
