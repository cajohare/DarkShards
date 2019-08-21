# DarkShards

[![DOI](https://zenodo.org/badge/156694427.svg)](https://zenodo.org/badge/latestdoi/156694427)
[//]: [![arXiv](https://img.shields.io/badge/arXiv-19??.????-.svg)](https://arxiv.org/abs/????.????)
[![MIT Licence](https://badges.frapsoft.com/os/mit/mit.svg?v=103)](https://opensource.org/licenses/mit-license.php)

This repository contains the code, data and notebooks for (one hopes) reproducing the entirety of our recent paper "*Dark Shards*". There are also a few results and bits and pieces which did not make it in to final version but I've left in for potential future interest. In case you're interested in a particular plot see below for a list of various plots and the links to the specific notebook that makes it.

Please contact me at ciaran.aj.ohare@gmail.com if you want to winge about why something doesn't work.


<img src="movies/SDSS-Gaia-Halo.gif" width="600" height="400">


## Contents

The code, plots, datas, etc. are sorted as follows:

* `data/` - Contains the *Gaia* sample of halo stars as well as various cleaned samples of the same data and the extracted substructures and fits.
* `code/` - Notebooks and python files which spit out various plots and other results that can be found in the paper
* `plots/` - plots get put here in pdf and png formats.
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


## Fig. 1
<img src="plots/plots_png/vrvphi-zfehcut_dist.png" width="600" height="600">

[Click here for the notebook](https://github.com/cajohare/DarkShards/blob/master/code/Plot_ThreeComponentHalo.ipynb)

---


## Fig. 2
<img src="plots/plots_png/Actions_decomposed_all.png" width="1000" height="300">

<img src="plots/plots_png/Actions_decomposed_sausage.png" width="1000" height="300">

<img src="plots/plots_png/Actions_decomposed_halo.png" width="1000" height="300">

[Click here for the notebook](https://github.com/cajohare/DarkShards/blob/master/code/Plot_Actions_decomp.ipynb)

---

## Fig. 4
<img src="plots/plots_png/fv3_halo_highE.png" width="1000" height="300">

[Click here for the notebook](https://github.com/cajohare/DarkShards/blob/master/code/Plot_comparespeeddists_halo.ipynb)

---


## Fig. 5
<p float="left">
  <img src="plots/plots_png/Vtriangle_S1.png" width="330" height="300">
  <img src="plots/plots_png/Vtriangle_S2.png" width="330" height="300">
</p>

<p float="left">
  <img src="plots/plots_png/Vtriangle_Rg5.png" width="330" height="300">
  <img src="plots/plots_png/Vtriangle_Cand14.png" width="330" height="300">
</p>

[Click here for the notebook](https://github.com/cajohare/DarkShards/blob/master/code/Stars_GenerateAllPlots.ipynb)

---

## Fig. 6
<img src="plots/plots_png/XYZ_S1.png" width="900" height="400">
<img src="plots/plots_png/XYZ_S2.png" width="900" height="400">
<img src="plots/plots_png/XYZ_Cand14.png" width="900" height="400">

[Click here for the notebook](https://github.com/cajohare/DarkShards/blob/master/code/Stars_GenerateAllPlots.ipynb)


---

## Fig. 8
<img src="plots/plots_png/fv3_Shards.png" width="1000" height="400">

[Click here for the notebook](https://github.com/cajohare/DarkShards/blob/master/code/Plot_SpeedDistributions.ipynb)

---

## Fig. 9
<img src="plots/plots_png/fv_Shards.png" width="450" height="400">

[Click here for the notebook](https://github.com/cajohare/DarkShards/blob/master/code/plot_Skymaps.ipynb)

---

## Fig. 10
<img src="plots/plots_png/ShardsFlux.png" width="1000" height="400">

[Click here for the notebook](https://github.com/cajohare/DarkShards/blob/master/code/Plot_Skymaps.ipynb)

---

## Fig. 11
<img src="plots/plots_png/Shards_axionspectrum.png" width="450" height="400">

[Click here for the notebook](https://github.com/cajohare/DarkShards/blob/master/code/Plot_axionspectrum.ipynb)

---

## Fig. 12
<img src="plots/plots_png/Shards_dRdE.png" width="450" height="400">

[Click here for the notebook](https://github.com/cajohare/DarkShards/blob/master/code/Plot_NRdists.ipynb)

---

## Fig. 13
<img src="plots/plots_png/Shards_annualmod_gravfocus_50GeV.png" width="450" height="400">

[Click here for the notebook](https://github.com/cajohare/DarkShards/blob/master/code/Plot_DAMAShards.ipynb)

---

## Fig. 14
<img src="plots/plots_png/Shards_annualmod_binned_50GeV.png" width="800" height="600">

[Click here for the notebook](https://github.com/cajohare/DarkShards/blob/master/code/Plot_DAMAShards.ipynb)

---

## Fig. 15
<img src="plots/plots_png/Shards_Directional.png" width="1000" height="600">

[Click here for the notebook](https://github.com/cajohare/DarkShards/blob/master/code/Plot_DirectionalNR.ipynb)

---

## Fig. 16
<img src="plots/plots_png/Shards_indiv_Directional.png" width="900" height="1000">

[Click here for the notebook](https://github.com/cajohare/DarkShards/blob/master/code/Plot_DirectionalNR.ipynb)

---

# Bonus plots

## Comparisons of stellar halo and Shards
<p float="left">
  <img src="plots/plots_png/ShardsStars_partitioned.png" width="330" height="300">
  <img src="plots/plots_png/Shards_Efehdist_all.png" width="330" height="300">
</p>

[Click here for the notebook](https://github.com/cajohare/DarkShards/blob/master/code/Plot_EnergyMetallicity.ipynb)

---


## Annual modulation harmonic analysis
<p float="left">
  <img src="plots/plots_png/Shards_FourierA.png" width="330" height="300">
  <img src="plots/plots_png/Shards_FourierA_gmin.png" width="330" height="300">
</p>

[Click here for the notebook (left)](https://github.com/cajohare/DarkShards/blob/master/code/Plot_StreamFourierSeries.ipynb)

[Click here for the notebook (right)](https://github.com/cajohare/DarkShards/blob/master/code/Plot_StreamFourierSeries2.ipynb)

---
