import subprocess as sp
from sealevelbayes.config import CONFIG
from .manager import register_dataset, get_datapath

require_giss = register_dataset("gistemp/tabledata_v4/GLB.Ts+dSST.csv", "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv", info="https://data.giss.nasa.gov/gistemp/")
require_hermans2021 = register_dataset("hermans2021", "https://data.4tu.nl/file/47d62866-50ea-448c-a20a-f26913acd72f/fcc93446-8e63-4f15-8f31-b1cc94d1bf53", doi="10.4121/12958079.v1", ext=".zip")

def register_coast(res):
    return register_dataset(f"naturalearth/ne_{res}_coastline",
        f"https://naturalearth.s3.amazonaws.com/{res}_physical/ne_{res}_coastline.zip", ext=".zip")
        # f"https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/{res}/physical/ne_{res}_coastline.zip", ext=".zip")

def register_land(res):
    return register_dataset(f"naturalearth/ne_{res}_land",
        f"https://naturalearth.s3.amazonaws.com/{res}_physical/ne_{res}_land.zip", ext=".zip")
        # f"https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/{res}/physical/ne_{res}_land.zip", ext=".zip")

require_coast_110m = register_coast("110m")
require_land_50m = register_land("50m")

register_dataset("zenodo-6419954-garner_kopp_2022/modules-data",
    "https://zenodo.org/records/6419954/files/modules-data.zip?download=1",
    doi="10.5281/zenodo.6419953", ext=".zip", skip_download=not CONFIG.get("download_facts_data"))

register_dataset("zenodo-6419954-garner_kopp_2022/modules-data-zos",
    "https://zenodo.org/records/6419954/files/modules-data-zos.zip?download=1",
    doi="10.5281/zenodo.6419953", ext=".zip", skip_download=not CONFIG.get("download_facts_data"))


require_ar6_wg3 = register_dataset("zenodo-6496232-AR6-WG3-plots/spm-box1-fig1-warming-data.csv",
    "https://zenodo.org/records/6496232/files/data/raw/spm-box-1-fig-1/spm-box1-fig1-warming-data.csv?download=1")


require_frederikse = register_dataset("zenodo-3862995-frederikse2020", "https://zenodo.org/api/records/3862995/files-archive", doi="10.5281/zenodo.3862994", ext=".zip")
require_bamber = register_dataset("pangaea-10.1594/PANGAEA.890030/Bamber-etal_2018.tab", "https://doi.pangaea.de/10.1594/PANGAEA.890030?format=textfile", doi="10.1594/PANGAEA.890030")

register_dataset("zenodo-6382554-garner2021/ar6-regional-confidence",
    "https://zenodo.org/records/6382554/files/ar6-regional-confidence.zip?download=1",
    ext=".zip", extract_name="zenodo-6382554-garner2021")

register_dataset("zenodo-6382554-garner2021/ar6",
    "https://zenodo.org/records/6382554/files/ar6.zip?download=1",
    ext=".zip", extract_name="zenodo-6382554-garner2021")

register_dataset("zenodo-6382554-garner2021/location_list.lst",
    "https://zenodo.org/records/6382554/files/location_list.lst?download=1")

require_church2011 = register_dataset("church_white_gmsl_2011_up", "http://www.cmar.csiro.au/sealevel/downloads/church_white_gmsl_2011_up.zip",
        ext=".zip", info="http://www.cmar.csiro.au/sealevel/sl_data_cmar.html", extract_name=".")


require_prandi2021 = register_dataset("prandi2021/quality_controlled_data_77853.nc",
                                      "https://www.seanoe.org/data/00637/74862/data/77853.nc",
                                      info="https://www.seanoe.org/data/00637/74862/")


require_rignot2019 = register_dataset("10.1594/PANGAEA.896940", "https://store.pangaea.de/Publications/VandenBroekeM_2019/RACMO2.3p1_ANT27_SMB_yearly_1979_2014.zip", ext=".zip", doi="10.1594/PANGAEA.896940")



# Also register older datasets
register_dataset("psmsl/2023/rlr_annual",
    "https://psmsl.org/data/obtaining/year_end/2023/rlr_annual.zip",
    extract_name="psmsl",
    )

register_dataset("psmsl/2022/rlr_annual",
    "https://psmsl.org/data/obtaining/year_end/2022/rlr_annual.zip",
    extract_name="psmsl",
    )

register_dataset("psmsl/2021/rlr_annual",
    "https://psmsl.org/data/obtaining/year_end/2021/rlr_annual.zip",
    extract_name="psmsl",
    )

download_rlr_annual = register_dataset("psmsl/rlr_annual",
    "https://psmsl.org/data/obtaining/rlr.annual.data/rlr_annual.zip",
    extract_name="psmsl",
    )

download_rlr_monthly = register_dataset("psmsl/rlr_monthly",
    "https://psmsl.org/data/obtaining/rlr.monthly.data/rlr_monthly.zip",
    extract_name="psmsl",
    )

download_met_monthly = register_dataset("psmsl/met_monthly",
    "https://psmsl.org/data/obtaining/met.monthly.data/met_monthly.zip",
    extract_name="psmsl",
    )


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

register_dataset("zenodo-3557199-zemp2019",
        "https://zenodo.org/records/3557199/files/Zemp_etal_results_regions_global_v11.zip?download=1",
        recursive=True, info="https://zenodo.org/records/3557199")



doc_nsidc = """
NASA NSIDC Datasets:
See instruction here to setup a .netrc file with your EarthData login
echo "machine urs.earthdata.nasa.gov login $LOGIN password $PASSWORD" >> ~/.netrc
chmod 0600 ~/.netrc
""".replace("\n", " ").strip()

NSIDC_WGET_ARGS = """--load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies --no-check-certificate --auth-no-challenge=on --reject "index.html*" -np -e robots=off"""
def get_nsidc_url(datapath):
    return f"https://daacdata.apps.nsidc.org/pub/DATASETS/{datapath}"

require_rgi7_global = register_dataset("nsidc0770_rgi_v7/global_files/RGI2000-v7.0-G-global",
                              url=get_nsidc_url("nsidc0770_rgi_v7/global_files/RGI2000-v7.0-G-global.zip"),
                              wget_args=NSIDC_WGET_ARGS, recursive=True, info=doc_nsidc)

require_rgi6_attribs = register_dataset("nsidc0770_rgi_v6/nsidc0770_00.rgi60.attribs",
                              url=get_nsidc_url("nsidc0770_rgi_v6/nsidc0770_00.rgi60.attribs.zip"),
                              wget_args=NSIDC_WGET_ARGS, recursive=True, info=doc_nsidc)

require_rgi5_attribs = register_dataset("nsidc0770_rgi_v5/nsidc0770_00.rgi50.attribs",
                              url=get_nsidc_url("nsidc0770_rgi_v5/nsidc0770_00.rgi50.attribs.zip"),
                              wget_args=NSIDC_WGET_ARGS, recursive=True, info=doc_nsidc)


register_dataset(**{"doi": "10.1594/PANGAEA.931657",
            "name": "marzeionmalles2021/suppl_reconstruction_data_region.nc",
            "url": "https://download.pangaea.de/dataset/931657/files/suppl_reconstruction_data_region.nc"})

register_dataset(        **{
            "name": "zenodo-7492152-hock2023",
            "url": "https://zenodo.org/records/7492152/files/fmaussion/global_glacier_volume_paper-v1.1.zip?download=1"
        })

register_dataset(        **{
            "name": "pangaea-MarzeionB-etal_2020/suppl_GlacierMIP_results.nc",
            "url": "https://store.pangaea.de/Publications/MarzeionB-etal_2020/suppl_GlacierMIP_results.nc"
        })