[![DOI](https://zenodo.org/badge/1026316112.svg)](https://doi.org/10.5281/zenodo.16440998)

# SeaLevelBayes

**Bayesian projections of future relative sea level rise tied to historical observations**

## Overview

This repository contains the code accompanying the following scientific article:

> **Relative sea level projections constrained by historical trends at tide gauge sites**
> *Mahé Perrette and Matthias Mengel*
> Potsdam Institute for Climate Impact Research (PIK), Germany
> **Science Advances* (submitted), 2025
> *DOI will be provided upon acceptance*

**Teaser**: *Local sea level rise projections consistent with historical data improve coastal impact assessment and adaptation planning.*


## Quickstart

### Install the Package

We used `python=3.11` with `pymc=5.9` for this project.

We recommend using a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
```

Clone the repository and install it in **editable** mode:

```bash
pip install -e .
```

In case issues are encountered, you may try the exact package versions used by the authors [requirements-manifest.txt](requirements-manifest.txt), as output from `pip freeze`.

> Without installation, command-line scripts such as `sealevelbayes-run` will not be available. Use `python -m sealevelbayes.runslr` or similar instead.

## Configuration

Configuration (e.g. paths) is stored in a `*.toml` file, located in one of the following:

- `./sealevelbayes.toml` (current directory)
- `$HOME/.config/sealevelbayes.toml`
- `$HOME/.sealevelbayes.toml`
- `{REPODIR}/config.toml`
- via CLI `--config-file`

Create and customize one with:

```bash
sealevelbayes-config > config.toml
```

Example entries:

```toml
datadir = "{REPODIR}/sealeveldata"
rundir = "{REPODIR}/runs"
```

## Running the Model

Once installed, the model can be run with:

```bash
sealevelbayes-run [...]
```

and the associated code can be found in the submodule [sealevelbayes.runslr](sealevelbayes/runslr.py).

The global-only version should work out of the box:
```bash
sealevelbayes-run --global-slr-only
```
(work is in progress to provide all data for local SLR as well)

The model parameters are defined in [runparams.py](/sealevelbayes/runparams.py) via the python argparse module.
The default is to run the default experiment in Perrette and Mengel (2025).
See the [pm2025](/pm2025) folder to run the various sensitivity experiments described in the manuscript.

An overview of all available parameters can be printed via

    sealevelbayes-run --help

A complete documentation is still work in progress.

### Model parameters input and run ID

Indicating different parameters via
the command line will usually result in an automatically generated `run ID`,
which is used as name for the output folder.

It is possible to print the runID via

    sealevelbayes-runid [...]

And the full config via

    sealevelbayes-runid [...] --print-config

Where `[...]` refers to any parameter you would pass to `sealevelbayes-run`.
This allows a workflow like:

    sealevelbayes-runid [...any parameter you want or nothing] --print-config > options.json
    # maybe edit the options.json
    sealevelbayes-run --param-file options.json

Also note that if an experiment is interrupted during sampling, it is possible to resume it via

    sealevelbayes-run --cirun <run ID> --resume

### Postprocessing

The `pymc` model produces a "trace.nc" file (arviz' Inference data format) with all samples inside. However
the trace only contains the random variables, observations, and the two SSP scenarios needed to apply the 2100
constraints (otherwise it would grow too large). Running additional scenarios, or calculating specific diagnostic requires a re-run with
`pymc.sample_posterior_predictive`, whereby the model is extended with new diagnostics, and the global-mean-temperature
driver is augmented with new scenarios. Typically that step is much faster than the actual sampling (a few minutes vs up to 24 hours).
To streamling the process of reloading, extending or redefining models and traces,
see [sealevelbayes.postproc.run.ExperimentTrace](/sealevelbayes/postproc/run.py).

#### Figures

The figures included in the associated manuscript were produced from the [jupyter notebooks](/notebooks) included in this repository.

#### Web app

The [web app](https://sealevel.netlify.app) and related [zenodo data](https://doi.org/10.5281/zenodo.15230503) were produced
by the [sealevelbayes.postproc.web](sealevelbayes/postproc/web.py) module, which can serve as an entry point for the curious user.

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.
See the [LICENSE](./LICENSE) file for details.

> Any use, modification, or redistribution — including via network services — must comply with AGPL-3.0 terms.

## Citation

If you use this code, please cite:

```bibtex
@article{perrette2025sealevelbayes,
  author  = {Perrette, Mahé and Mengel, Matthias},
  title   = {Relative sea level projections constrained by historical trends at tide gauge sites},
  journal = {Science Advances},
  year    = {2025},
  doi     = {DOI TO BE ADDED}
}
```

## Contact

For questions or feedback, please open a GitHub issue or contact the authors.

## Funding

This work has received funding from:

- The German Federal Ministry of Education and Research (BMBF) under project 01LP1907A
- The European Union’s Horizon Europe research and innovation programme under:
  - Grant agreement No. 101081369 (SPARCCLE)
  - Grant agreement No. 101135481 (COMPASS)

## Data and Materials Availability

Third-party data are generally not included in the repository.
They must be downloaded directly from openly available sources as documented in the Methods section of the associated article.
To make that task easier, all datasets are listed in the machine-readable [catalogue.json](/sealevelbayes/datasets/catalogue.json), which can be used with our [helper tool](#data-download-tool).
The global datasets are downloaded "on-demand", so that `sealevelbayes-run --global-slr-only` works out of the box. The following datasets were obtained via personal communication:

- Glacier fingerprints shared by Thomas Frederikse, used in his work in: [Frederikse et al. (2020), *Nature*](https://www.nature.com/articles/s41586-020-2591-3)
- The GIA ensemble described in: [Caron et al. (2018), *Geophysical Research Letters*](https://doi.org/10.1002/2017GL076644)

These datasets are essential components required to run our local model.
If you wish to obtain a copy of these data, please contact the corresponding author, or the respective authors directly.

The data output from the main analysis are available at:
  https://doi.org/10.5281/zenodo.15230503

An interactive visualization of sea-level estimates at tide gauge stations, based on the output data included in the zenodo repository, is available at:
  https://sealevel.netlify.app


## Data download tool

An helper tool is provided to download many of the openly available datasets.
They can be listed and downloaded via the `sealevelbayes-download` script,
which reads a [catalogue.json](/sealevelbayes/datasets/catalogue.json) file.

Examples:

```bash
sealevelbayes-download church_white_gmsl_2011_up naturalearth/ne_110m_coastline
sealevelbayes-download --ls --global
sealevelbayes-download --ls
sealevelbayes-download --global
sealevelbayes-download --all
sealevelbayes-download *garner* --print
```

The datasets required for the global SLR model are downloaded on-demand in the code, via the [require_dataset](sealevelbayes/datasets/manager.py) function.

WARNINGS: some datasets are very large (e.g. the Garner et al. 2021 regional data is 37 GB). So make sure you have enough available space before using the `--all` option.

The listing does not include the CMIP6 archive for piControl runs. They are available from the usual ESGF portals.
