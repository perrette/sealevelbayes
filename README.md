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

We used `python=3.11` with `pymc=5.9` for this project.

We recommend using a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
```

### Install the Package

Clone the repository and install it in **editable** mode:

```bash
pip install -e .
```

This allows development and editing without reinstallation. Alternatively, add the repository to your `PYTHONPATH` and install dependencies via:

```bash
pip install -r requirements.txt
```

The exact package versions used can be found in [requirements-manifest.txt](requirements-manifest.txt), as output from `pip freeze`.

> Without installation, command-line scripts such as `sealevelbayes-run` will not be available. Use `python -m sealevelbayes.runslr` or similar instead.

## Fetch data dependencies

Data that can be downloaded from external, official sources are not included in the repository.
Instead they are registered internally and can be listed and downloaded `sealevelbayes-download` script,
which is an alias for `python -m sealevelbayes.datasets.manager`, e.g.

```bash
sealevelbayes-download --ls
```

Examples:

```bash

    sealevelbayes-download --name church_white_gmsl_2011_up naturalearth/ne_110m_coastline
    sealevelbayes-download --name psmsl*
    sealevelbayes-download --json  # a custom selection of datasets for basic use of the package without recalibration
    sealevelbayes-download --all
```

The specific datasets can be found in [datasets.json](pm2025/datasets.json).

To reproduce results from the submitted manuscript by Perrette & Mengel (2025), ensure you use the correct historical dataset versions. Contact the authors or consult the Zenodo archive (forthcoming) if needed.

WARNINGS: some datasets are very large (e.g. the Garner et al. 2021 regional data is 37 GB). You may want to edit [datasets.json](pm2025/datasets.json) to remove it, if not used.

The listing does not include the CMIP6 archive for piControl runs. They are available from the usual ESGF portals.

Additional data from Frederikse et al. (2020) and Caron et al. (2017) were obtained via personal communication with the authors and are available from the corresponding author upon reasonable request.

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
download_facts_data = false
datadir = "{REPODIR}/sealeveldata"
rundir = "{REPODIR}/runs"
isimipdir = "/p/projects/isimip/isimip"
```

The last line points to the ISIMIP mirror on the PIK cluster. (not used in Perrette and Mengel (2025), Sci. Adv., submitted)

## Running the Model

Once installed, explore available options with:

```bash
sealevelbayes-run --help
```

A default configuration and example runs will be made available in future updates.
See the [pm2025](/pm2025) folder to run the experiments in Perrette and Mengel (2025)


## Model parameters

The model parameters are defined in [runparams.py](/sealevelbayes/runparams.py) via the python argparse module.
The default is to run the default experiment in Perrette and Mengel (2025).

See for [this example](/pm2025/run_all_experiments.sh) to run the various sensitivity experiments.

It is possible to quickly print the runID via

    sealevelbayes-runid [...]

Or the full config via

    sealevelbayes-runid --print-config

This allow a workflow like:

    sealevelbayes-runid [...any parameter you want or nothing] --print-config > options.json
    # maybe edit the options.json
    sealevelbayes-run --param-file options.json

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

The data required to reproduce the main analysis are available at:
  https://doi.org/10.5281/zenodo.15230503

An interactive visualization of sea-level estimates at tide gauge stations is available at:
  https://sealevel.netlify.app

**Note on third-party data**:

Some datasets included in the repository (e.g. under sealeveldata/frederikse2020-personal-comm/) were obtained via personal communication and do not have a clearly defined public license. These include:

- Glacier fingerprints and glacial isostatic adjustment (GIA) statistics shared by Thomas Frederikse, partly based on his work in: [Frederikse et al. (2020), *Nature*](https://www.nature.com/articles/s41586-020-2591-3)

- GIA statistics derived from: [Caron et al. (2018), *Geophysical Research Letters*](https://doi.org/10.1002/2017GL076644)

These datasets are required to fully reproduce the published results but may not be redistributed without the consent of the original authors.
If you wish to use these data beyond reproducibility, please contact the respective authors directly. We thank them for sharing their data and recommend proper acknowledgment in any derivative work.