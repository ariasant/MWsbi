# MW-sbi

[![Paper](https://img.shields.io/badge/preprint-arXiv2603.12317-red)](https://doi.org/10.48550/arXiv.2603.12317)

## Overview

This repository contains codes to develop a simulation-based inference analysis predicting the properties of the Milky Way disrupted satellite galaxies at accretion from the current properties of its stellar debris. Important implementation details:

- The stellar properties and debris definition are taken from the **GAIAEDR3** and **APOGEE** surveys following the selections in [Horta D., et al., 2023](http://dx.doi.org/10.1093/mnras/stac3179). 

- The analysis leverages the merger-debris information encoded in the **Auriga simulations** ([Grand R. J. J., et al., 2017](https://ui.adsabs.harvard.edu/abs/2017MNRAS.467..179G)).

- Before training the neural density estimator, the stellar debris data is optimally compressed using the **Fishnets** model ([Makinen T. L., et al., 2023](http://dx.doi.org/10.48550/arXiv.2310.03812)).

- The simulation-based inference model and validation is based on the **LTU-ILI** framework ([Ho M., et al., 2024](http://dx.doi.org/10.33232/001c.120559)).

## File content

- `codes`: folder with the functions used in the analysis.
- `create_data.py`: calculates stellar properties of Milky Way stars.
- `main.py`: implements and validated the SBI model.
- `paper_plots.ipynb`: reproduces the plots in [Sante A., et al., 2026](https://doi.org/10.48550/arXiv.2603.12317).



## Example Usage

```
from main import pipeline

stellar_features = ['E', 'L', 'FeH', 'MgFe']
merger_parameters = ['infall_time','log_Mprog_stellar', 'log_Mprog', 'log_Mprog2host']
substructures = [
    'GES', 
    'Sagittarius', 
    'Helmi',
    'Sequoia_K19',
    'Sequoia_M19',
    'Sequoia_N20',
    'Iitoi', 
    'Thamnos',
    'LMS', 
    'Heracles', 
]

data_dir = "/your/path/to/MW/data"
output_dir = "/your/path/to/output/folder"

plot_labels=[
    '$\\tau \, [\mathrm{Gyr}]$',
    'log($M_{*}/\\rm{M}_{\odot}$)',
    'log($M/\\rm{M}_{\odot}$)', 
    'MMR (log)'
]

plot_ranges=[
    [0.1,13.9], # infall time
    [5.9,10.9], # log stellar mass
    [8.5,11.9], # log halo mass
    [-3.2,-0.1] # log MMR
]

mwsbi = pipeline(
    features=features,
    parameters=parameters,
    substructures=substructures,
    data_dir=data_dir,
    output_dir=output_dir,
    plot_labels=plot_labels,
    plot_ranges=plot_ranges
)

mwsbi.run()

```

## Cite this work

```
@ARTICLE{2026arXiv260312317S,
       author = {{Sante}, Andrea and {Font}, Andreea S. and {Kawata}, Daisuke and {Makinen}, T. Lucas and {Grand}, Robert J.~J.},
        title = "{A simulation-based inference of the Milky Way merger history}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics of Galaxies},
         year = 2026,
        month = mar,
          eid = {arXiv:2603.12317},
        pages = {arXiv:2603.12317},
          doi = {10.48550/arXiv.2603.12317},
archivePrefix = {arXiv},
       eprint = {2603.12317},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2026arXiv260312317S},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```


