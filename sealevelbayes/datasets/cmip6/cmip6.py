"""
CMIP6 archive on climate_data_central repository, as downlaoded by Matthias Büchner
"""

from pathlib import Path
import glob, os
import logging
import subprocess as sp

from sealevelbayes.datasets.manager import get_datapath

logging.basicConfig()
logger = logging.getLogger('CMIP6')

DATA = "/p/projects/isipedia/perrette/sealevel/data"
CMIP6_ROOT = "/p/projects/climate_data_central/CMIP/CMIP6"

all_experiments = ['historical', 'ssp585', 'ssp126', 'ssp245', 'ssp370', 'ssp119']


def fmtmodel(model):
    return model.lower().replace('-','_')


def list_cmip6_files(model_group="*", model="*", experiment="*", ensemble="*", domain="*", variable="*", grid="*", version="*", timestamp="*"):
    return list(sorted(glob.glob(f"{CMIP6_ROOT}/*/{model_group}/{model}/{experiment}/{ensemble}/{domain}/{variable}/{grid}/{version}/{variable}_{domain}_{model}_{experiment}_{ensemble}_{grid}_{timestamp}.nc")))


def parse_filename(file):
    file = file.replace(CMIP6_ROOT+"/", "")
    keys = "_", "model_group", "model", "experiment", "ensemble", "domain", "variable", "grid", "version", "name"
    o = {k:v for k,v in zip(keys, file.split("/"))}
    timestamp = os.path.splitext(o["name"])[0].split("_")[-1]
    o["timestamp"] = timestamp
    return o

# all available models
all_models = list(sorted(parse_filename(f)['model'] for f in list_cmip6_files()))

MODELS_MAP = {fmtmodel(model):model for model in all_models}
