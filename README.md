# DarkShards

[//]: [![DOI](https://zenodo.org/badge/????????.svg)](https://zenodo.org/badge/latestdoi/????????) 
[//]: [![arXiv](https://img.shields.io/badge/arXiv-19??.????-.svg)](https://arxiv.org/abs/????.????)
[![MIT Licence](https://badges.frapsoft.com/os/mit/mit.svg?v=103)](https://opensource.org/licenses/mit-license.php)

This repository contains the code, data and notebooks for (one hopes) reproducing the entirety of our recent paper "*Impact of the ex-situ halo on searches for dark matter*". There are also many results which did not make it in to final version but I've left in for potential future interest. In case you're interested in a particular plot see below for a list of various plots and the links to the specific notebook that makes it.

Please contact me at ciaran.aj.ohare@gmail.com if you want to winge about why something doesn't work.


<img src="movies/SDSS-Gaia-Halo.gif" width="600" height="400">


## Contents

The code, plots, datas, etc. are sorted as follows:

* `data/` - Contains the *Gaia* sample of halo stars as well as various cleaned samples of the same data and the extracted substructures and fits.
* `code/` - contains various functions which are used by the notebooks to generate the results and plots 
* `notebooks/` - Notebooks which spit out various plots and other results that can be found in the paper
* `plots/` - all plots get put here in pdf and png formats.
* `movies/` - A few movies just for visualising the halo sample etc.

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

---


# The Results
Here is each figure in the paper in order


## Fig. X
<img src="plots/plots_png/vrvphi-zfehcut_dist.png" width="600" height="600">

[Click here for the notebook](https://github.com/cajohare/DarkShards/blob/master/code/plot_Skymaps.ipynb)

---


## Fig. X
<img src="plots/plots_png/Actions_decomposed_all.png" width="1000" height="300">

<img src="plots/plots_png/Actions_decomposed_sausage.png" width="1000" height="300">

<img src="plots/plots_png/Actions_decomposed_halo.png" width="1000" height="300">

[Click here for the notebook](https://github.com/cajohare/DarkShards/blob/master/code/plot_Skymaps.ipynb)

---


## Fig. X
<img src="plots/plots_png/ShardsStars_partitioned.png" width="400" height="400">
<img src="plots/plots_png/Shards_Efehdist_all.png" width="400" height="400">

[Click here for the notebook](https://github.com/cajohare/DarkShards/blob/master/code/plot_Skymaps.ipynb)

---

## Fig. X
<img src="plots/plots_png/fv3_halo_highE.png" width="400" height="400">

[Click here for the notebook](https://github.com/cajohare/DarkShards/blob/master/code/plot_Skymaps.ipynb)

---


## Fig. X
<img src="plots/plots_png/Vtriangle_S1.png" width="400" height="400">
<img src="plots/plots_png/Vtriangle_S2.png" width="400" height="400">

<img src="plots/plots_png/Vtriangle_Rg5.png" width="400" height="400">
<img src="plots/plots_png/Vtriangle_Cand9.png" width="400" height="400">

[Click here for the notebook](https://github.com/cajohare/DarkShards/blob/master/code/plot_Skymaps.ipynb)

---

## Fig. X
<img src="plots/plots_png/XYZ_S1.png" width="700" height="400">

<img src="plots/plots_png/XYZ_S2.png" width="700" height="400">

<img src="plots/plots_png/XYZ_Cand14.png" width="700" height="400">

[Click here for the notebook](https://github.com/cajohare/DarkShards/blob/master/code/plot_Skymaps.ipynb)

---

## Fig. X
<img src="plots/plots_png/Shards_fv_wVL2.png" width="400" height="400">

[Click here for the notebook](https://github.com/cajohare/DarkShards/blob/master/code/plot_Skymaps.ipynb)

---

## Fig. X
<img src="plots/plots_png/fv3_Shards.png" width="1000" height="400">

[Click here for the notebook](https://github.com/cajohare/DarkShards/blob/master/code/plot_Skymaps.ipynb)

---

## Fig. X
<img src="plots/plots_png/fv_Shards.png" width="400" height="400">

[Click here for the notebook](https://github.com/cajohare/DarkShards/blob/master/code/plot_Skymaps.ipynb)

---

## Fig. X
<img src="plots/plots_png/Shards_axionspectrum.png" width="400" height="400">

[Click here for the notebook](https://github.com/cajohare/DarkShards/blob/master/code/plot_Skymaps.ipynb)

---

## Fig. X
<img src="plots/plots_png/Shards_dRdE.png" width="400" height="400">

[Click here for the notebook](https://github.com/cajohare/DarkShards/blob/master/code/plot_Skymaps.ipynb)

---

## Fig. X
<img src="plots/plots_png/Shards_annualmod_gravfocus.png" width="400" height="400">

[Click here for the notebook](https://github.com/cajohare/DarkShards/blob/master/code/plot_Skymaps.ipynb)

---

## Fig. X
<img src="plots/plots_png/Shards_annualmod_gravfocus_Energies.png" width="800" height="800">

[Click here for the notebook](https://github.com/cajohare/DarkShards/blob/master/code/plot_Skymaps.ipynb)

---

## Fig. X
<img src="plots/plots_png/Shards_Directional.png" width="1000" height="1000">

[Click here for the notebook](https://github.com/cajohare/DarkShards/blob/master/code/plot_Skymaps.ipynb)

---



