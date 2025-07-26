import os
import json
import subprocess as sp
import sealevelbayes
from .manager import register_dataset, get_datapath

datasets_json = os.path.join(sealevelbayes.__path__[0], "datasets", "catalogue.json")
for record in json.load(open(datasets_json))["records"]:
    register_dataset(**record)

picontrol_url = "foote.pik-potsdam.de:/p/projects/isipedia/perrette/sealevel/slr-tidegauges-future/sealeveldata/cmip6/zos/regridded/piControl"

def sync_cmip6_zos_picontrol():
    target = get_datapath("cmip6/zos/regridded/piControl")
    target.mkdir(parents=True, exist_ok=True)
    sp.check_call(f"rsync -aLrvz --progress {picontrol_url}/ {target}/",
        shell=True)

register_dataset("cmip6/zos/regridded/piControl", url=picontrol_url, caller=sync_cmip6_zos_picontrol, info="Synced from foote.pik-potsdam.de (private)")


climexp_folder = get_datapath("climexp.knmi.nl")

def download_climexp():
    climexp_folder.mkdir(exist_ok=True)
    sp.check_call("wget -r -e robots=off -nH --no-parent https://climexp.knmi.nl/CMIP5/Tglobal", shell=True, cwd=climexp_folder)

register_dataset("climexp.knmi.nl/CMIP5/Tglobal", url="https://climexp.knmi.nl/CMIP5/Tglobal", caller=download_climexp)