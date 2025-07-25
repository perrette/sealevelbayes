import os
import tqdm
import requests
import zipfile
from sealevelbayes.config import CONFIG, get_webpath, get_runpath
from sealevelbayes.logs import logger

# DEPOSITION_ID = "15230504"
DEPOSITION_ID = "16375278"

def get_deposition(deposit_id, token):

    HEADERS = {"Authorization": f"Bearer {token}"}

    # Fetch the draft deposition
    r = requests.get(f"https://zenodo.org/api/deposit/depositions/{deposit_id}", headers=HEADERS)
    r.raise_for_status()

    return r.json()

def create_new_version(deposit_id, token):
    headers = {"Authorization": f"Bearer {token}"}

    r = requests.post(f"https://zenodo.org/api/deposit/depositions/{deposit_id}/actions/newversion",
                  headers=headers)

    r.raise_for_status()
    return r.json()

def upload_zenodo(filename, deposit_id, target=None, token=None, bucket_url=None):

    if token is None:
        token = os.getenv("ZENODO_TOKEN")

    if bucket_url is None:
        deposition = get_deposition(deposit_id, token)
        bucket_url = deposition["links"]["bucket"]
        print(f"Bucket URL: {bucket_url}")

    if target is None:
        target = os.path.basename(filename)

    with open(filename, "rb") as fp:
        r = requests.put(
            f"{bucket_url}/{target}",
            data=fp,
            headers={"Authorization": f"Bearer {token}"},
        )
        r.raise_for_status()
        print(f"Uploaded: {target}")


def delete_zenodo(deposit_id, names=[], keep_only=None, token=None):

    if token is None:
        token = os.getenv("ZENODO_TOKEN")

    HEADERS = {"Authorization": f"Bearer {token}"}

    # Step 1: List current files
    r = requests.get(f"https://zenodo.org/api/deposit/depositions/{deposit_id}", headers=HEADERS)
    files = r.json()["files"]

    # Step 2: Delete unwanted file(s)
    for f in files:
        if f["filename"] in names or (keep_only is not None and f["filename"] not in keep_only):
            file_id = f["id"]
            del_url = f"https://zenodo.org/api/deposit/depositions/{deposit_id}/files/{file_id}"
            del_r = requests.delete(del_url, headers=HEADERS)
            del_r.raise_for_status()
            print(f"Deleted: {f['filename']}")



def merge_zip_archives(main_zip_filename, zip_filenames, zip_subfolders=None):
    with zipfile.ZipFile(main_zip_filename, 'w', zipfile.ZIP_DEFLATED) as main_zip:
        for i, zip_filename in enumerate(tqdm.tqdm(zip_filenames)):
            # Open each zip file and extract its content
            with zipfile.ZipFile(zip_filename, 'r') as sub_zip:
                # Create a subfolder for each subarchive
                if zip_subfolders is not None:
                    subfolder_name = zip_subfolders[i]
                else:
                    # Use the name of the zip file without extension as the subfolder name
                    # e.g. 'subarchive.zip' -> 'subarchive'
                    subfolder_name = os.path.splitext(os.path.basename(zip_filename))[0]  # e.g. 'subarchive'

                # Create a folder and add all files from this subarchive inside
                for file_name in sub_zip.namelist():
                    file_data = sub_zip.read(file_name)
                    # Writing file content into the main zip, within the subfolder
                    main_zip.writestr(f"{subfolder_name}/{file_name}", file_data)


def update_all(deposit_id, experiments, default_experiment, version, token=None):
    staging = get_webpath(f"zenodo_{version}_{deposit_id}")
    staging.mkdir(parents=True, exist_ok=True)
    # upload all files
    appfiles = [get_runpath(cirun)/"postproc"/ "quantiles_per_station.zip" for cirun in experiments]
    webappfile = staging/"webapp_quantiles_per_station.zip"
    logger.info(f"Experiments: {experiments}")
    logger.info(f"Archive: {appfiles}")
    logger.info(f"Merge experiment files into a unique web app archive {webappfile}")
    merge_zip_archives(webappfile, appfiles, zip_subfolders=experiments)

    logger.info(f"Upload webapp_quantiles_per_station.zip for {default_experiment}")
    upload_zenodo(str(webappfile), deposit_id=deposit_id, target=webappfile.name, token=token)

    # upload_zenodo(str(filename_csv), deposit_id=o.zenodo_deposit_id, target=f"webapp/{o.cirun}/quantiles_per_station.zip".replace("/", "__"))

    # if o.upload_zenodo:
    logger.info(f"Upload quantiles_all.nc for {default_experiment}")
    for cirun in [default_experiment]:
        ncfile = get_runpath(cirun)/"postproc"/ "quantiles_all.nc"
        upload_zenodo(str(ncfile), deposit_id=deposit_id, target=f"netcdf/{cirun}/quantiles_all.nc".replace("/", "__"), token=token)


def remove_tar(deposit_id, token=None):
    if token is None:
        token = os.getenv("ZENODO_TOKEN")

    HEADERS = {"Authorization": f"Bearer {token}"}

    ZENODO_URL = f"https://zenodo.org/api/deposit/depositions/{deposit_id}"

    # List all files in the deposit
    response = requests.get(ZENODO_URL, headers=HEADERS)
    response.raise_for_status()
    files = response.json()["files"]

    # Loop through and delete all .tar files
    for f in files:
        if f["filename"].endswith(".tar"):
            file_id = f["id"]
            del_url = f"{ZENODO_URL}/files/{file_id}"
            print(f"Deleting {f['filename']} ...")
            r = requests.delete(del_url, headers=HEADERS)
            r.raise_for_status()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--deposit_id", default=DEPOSITION_ID)
    parser.add_argument("--target")
    parser.add_argument("--delete", nargs="+", help="Delete files from the deposition")
    parser.add_argument("--keep-only", nargs="+")
    parser.add_argument("--token", default=os.getenv("ZENODO_TOKEN"))
    parser.add_argument("--update-all", action="store_true", help="Upload all files to the deposits")
    parser.add_argument("--remove-tar", action="store_true")
    parser.add_argument("--experiments", nargs="*", default=CONFIG.get("experiments"))
    parser.add_argument("--default_experiment", default=CONFIG.get("default_experiment"))
    parser.add_argument("--version", default=CONFIG.get("version",""))
    parser.add_argument("--create-new-version", action="store_true",)
    parser.add_argument("--upload")

    args = parser.parse_args()

    if args.token is None:
        args.token = os.getenv("ZENODO_TOKEN", None)

    if args.create_new_version:
        # Create a new version of the deposition
        new_version = create_new_version(args.deposit_id, args.token)
        print("New version created:", new_version)
        return

    if args.update_all:
        # return update_all(args.deposit_id, args.token, experiments=args.experiments, default_experiment=args.default_experiment, version=args.version)
        return update_all(args.deposit_id, args.experiments, args.default_experiment, args.version, token=args.token)

    if args.remove_tar:
        remove_tar(args.deposit_id, token=args.token)
        return

    if args.delete:
        delete_zenodo(args.deposit_id, names=args.delete, keep_only=args.keep_only, token=args.token)

    if args.upload:
        upload_zenodo(args.upload, deposit_id=args.deposit_id, target=args.target, token=args.token)

    print("Done")

if __name__ == "__main__":
    main()