"""Handle remote datasets to be downloaded
"""
import os
from pathlib import Path
import fnmatch
import json
import datetime
import tqdm
import shutil
import functools
import subprocess as sp

import sealeveldata
from sealevelbayes.logs import logger, log_parser, setup_logger
from sealevelbayes.config import CONFIG, config_parser, CACHE_FOLDER, get_sharedpath

DOWNLOAD_FOLDER = Path(CONFIG['datadir'])
DATASET_JSON = Path(CONFIG["datadir"]) / "datasets.json"

_DEFAULTDATADIR = sealeveldata.__path__[0]

def get_downloadpath(relpath=''):
    return Path(CONFIG.get('downloaddir', CACHE_FOLDER / "download")) / relpath


def get_datapath(relpath=''):
    return Path(CONFIG.get('datadir', _DEFAULTDATADIR)) / relpath

def search_datapath(relpath='', raise_error=True, quiet=False):
    """look in more places than get_datapath, and raise error is file is not found
    """
    search_folders = [
        get_datapath(),
        get_sharedpath(),  # this is the "work" directory writable by the user
        _DATADIRLFS,  # original data directory installed by pip
        Path(_DATADIRLFS) / "savedwork",
        ]

    for folder in search_folders:
        file = Path(folder) / relpath
        if file.exists():
            logger.debug(f"{relpath} found in {folder}")
            return file
    if raise_error:
        raise FileNotFoundError(relpath)
    elif not quiet:
        logger.warning(f"File not found: {relpath}")

    # return Path(CONFIG['datadir']) / relpath

    return get_datapath(relpath)


MEGABYTES = 1024*1024

def download(url, destination, chunk_size=MEGABYTES, wget_args=None):
    partial = Path(str(destination) + ".download")

    if wget_args:
        if partial.exists():
            wget_args = "--continue " + wget_args
        logger.info(f"Download {url} to {partial}")
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        cmd = f"wget {url} -O {partial} {wget_args}"
        logger.debug(cmd)
        sp.check_call(cmd, shell=True)
        shutil.move(partial, destination)
        return

    if partial.exists():
        response = _resume_download(url, partial, chunk_size)

    else:
        response = _download(url, partial, chunk_size)

    shutil.move(partial, destination)
    return response


def _download(url, destination, chunk_size=MEGABYTES):
    """
    ref: https://realpython.com/python-download-file-from-url/#using-the-third-party-requests-library
    """
    logger.info(f"Download {url} to {destination}")

    import requests
    with requests.get(url, stream=True, allow_redirects=True) as response:
        Path(destination).parent.mkdir(parents=True, exist_ok=True) # create folder if it does not exist
        total = int(response.headers.get('content-length', 0))
        with open(destination, mode="wb") as file, tqdm.tqdm(total=round(total/MEGABYTES,2), unit='MB') as bar :
            for chunk in response.iter_content(chunk_size=chunk_size):
                size = file.write(chunk)
                bar.update(round(size/MEGABYTES,2))

    assert response.ok
    return response


def _resume_download(url, destination, chunk_size=MEGABYTES):
    """
    ref: https://stackoverflow.com/a/22894873/2192272
    """
    import requests

    logger.info(f"Resume download {url} to {destination}")

    with open(destination, mode="ab") as file:
        resume_byte_pos = file.tell()
        resume_header = {'Range': 'bytes=%d-' % resume_byte_pos}

        with requests.get(url, stream=True, headers=resume_header, allow_redirects=True) as response:
            if response.headers.get('content-range'):
                total = int(response.headers.get('content-range', "/0").split("/")[-1])
            else:
                total = int(response.headers.get('content-length', 0)) + int(resume_byte_pos)

            with tqdm.tqdm(
                initial=round(int(resume_byte_pos)/MEGABYTES, 2),
                total=round(total/MEGABYTES, 2),
                unit='MB',
                ) as bar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    size = file.write(chunk)
                    bar.update(round(size/MEGABYTES, 2))

    assert response.ok
    return response


def _get_extension(archive):
    stripped = str(archive).strip().split("?")[0]
    if stripped.endswith(".tar.gz"):
        ext = ".tar.gz"
    else:
        basename, ext = os.path.splitext(stripped)
    return ext


def _is_archive(path, ext=None):
    if ext is None:
        ext = _get_extension(path)
    return ext in KNOWN_ARCHIVE_EXTENSIONS


KNOWN_ARCHIVE_EXTENSIONS = [".zip", ".tar", ".gz"]


def extract_archive(downloaded, path, ext=None, members=None, recursive=False, delete_archive=False):
    # Extract (ref: https://ioflood.com/blog/python-unzip)
    archive = str(downloaded)

    if not ext:
        ext = _get_extension(archive)

    if not ext:
        if archive != path:
            logger.info(f"mv {archive} {path}")
            shutil.move(archive, path)
        return

    logger.info(f"Extract {archive} to {path}")

    if ext == ".zip":
        import zipfile
        with zipfile.ZipFile(archive, 'r') as zip_ref:
            zip_ref.extractall(path, members=members)
            if members is None: members = zip_ref.namelist()

    elif ext in (".tar", ".tar.gz"):
        import tarfile
        with tarfile.open(archive, 'r:gz' if ext == ".tar.gz" else 'r') as tar_ref:
            tar_ref.extractall(path, members=members)
            if members is None: members = tar_ref.getmembers()

    elif ext in (".gz"):
        import gzip
        with gzip.open(archive, 'rb') as f_in:
            with open(path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            if members is None: members = [] # unknown method

    elif ext:
        raise NotImplementedError(f"Unknown extension {ext}")

    if recursive:
        for member in members:
            extracted_path = str(Path(path) / member)
            if os.path.isfile(extracted_path) and extracted_path.endswith(('.zip', '.tar', '.gz')):
                ext = _get_extension(extracted_path)
                extract_archive(extracted_path, extracted_path[:-len(ext)], ext=ext, recursive=True, delete_archive=True)

    # delete if everything above went fine
    if delete_archive:
        os.remove(archive)

def get_filename_from_url(url):
    return os.path.basename(url)

def _require_dataset(name, url=None, extract=None, force_download=None, extract_name=None, members=None, recursive=False, skip_download=False, ext=None, caller=None, ignore_cache=False, wget_args=None, **metadata):

    filepath = get_datapath(name)

    if type(url) is list:
        # if url is a list, register each url with the same name
        for u in url:
            # parser url u and derive a sub-name from it, to append to name
            # use urlparse to remove query parameters
            if u.endswith("/"):
                u = u[:-1]
            for char in ["?", "#"]:
                if char in u:
                    u = u.split(char)[0]
            sub_name = u.split("/")[-1]
            _require_dataset(name + "/" + sub_name, u, extract=extract, force_download=force_download, extract_name=extract_name, members=members, recursive=recursive, skip_download=skip_download, ext=ext, caller=caller, ignore_cache=ignore_cache, wget_args=wget_args, **metadata)
        return filepath


    dataset_json = get_datapath("datasets.json")

    download_folder = get_downloadpath()

    if not skip_download and (not filepath.exists() or force_download):

        if caller is not None:
            return caller()

        download_name = get_filename_from_url(url)
        downloaded = download_folder / name / download_name
        download_folder.mkdir(exist_ok=True, parents=True)

        # if not (downloaded).exists() or force_download:
        if not (downloaded).exists() or ignore_cache:
            download(url, downloaded, wget_args=wget_args)
        elif (downloaded).exists() and force_download:
            logger.warning(f"{downloaded} found on disk and will be reused. Please manually delete or pass --ignore-cache to force new download.")

        if extract is None:
            if _is_archive(downloaded, ext=ext):
                extract = True
            else:
                extract = False

        target = get_datapath(extract_name or name)

        if extract:
            extract_archive(downloaded, target, ext=ext, members=members, recursive=recursive)

        else:
            logger.info(f"mv {downloaded} {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(downloaded, target)

        # also keep a centralized .json that can be git-tracked
        metadata.update({"url": url, "date": str(datetime.datetime.now()), "extract_name": str(extract_name) if extract_name else None, "name": str(name), "ext": ext, "members": members, "recursive": recursive})

        if dataset_json.exists():
            logger.info(f"Update {dataset_json}")
            all_data_info = json.load(open(dataset_json))
        else:
            logger.info(f"Create {dataset_json}")
            all_data_info = { "records": [] }

        # re-arrange as dict to ease update
        by_key = {r['name']:r for r in all_data_info["records"]}
        by_key[metadata['name']] = metadata

        # but save as list of records to make editing less redundant
        all_data_info["records"]  = sorted(by_key.values(), key=lambda r: r['name'])

        # remove redundant fields
        for r in all_data_info["records"]:
            if 'recursive' in r and not r['recursive']: r.pop('recursive')
            if 'members' in r and not r['members']: r.pop('members')
            if 'extract_name' in r and (not r['extract_name'] or not r.get('ext') or r['extract_name'] == r['name']): r.pop('extract_name')
            if 'ext' in r and not r['ext']: r.pop('ext')

        with open(dataset_json, "w") as f:
            json.dump(all_data_info, f, indent=4, sort_keys=True)

    return filepath


DATASET_REGISTER = { "records": [] }

def register_dataset(name, url=None, **kwargs):
    """Add dataset to the DATASET_REGISTER (useful for download scripts) and return require function
    """
    record = {"name": name, "url": url, **kwargs }
    DATASET_REGISTER['records'].append(record)
    return functools.partial(_require_dataset, **record)

def require_dataset(name):
    for record in DATASET_REGISTER['records']:
        if record['name'] == name:
            return _require_dataset(**record)

    raise ValueError(f"Dataset {name} not found in the register. Available datasets are {', '.join([r['name'] for r in DATASET_REGISTER['records']])}")


def download_by_records(records, **kwargs):

    for r in records:
        _require_dataset(**r, **kwargs)

def expand_names(names):
    """The input list may contain wild cards
    """
    all_datasets = [r['name'] for r in DATASET_REGISTER['records']]
    expanded_names = []
    for name in names:
        filtered = fnmatch.filter(all_datasets, name)
        if not filtered:
            from rapidfuzz import process, fuzz
            candidates = [c[0] for c in process.extract(name, all_datasets, scorer=fuzz.WRatio)]
            logger.warning(f"{name} does not match any dataset. Did you mean {' '.join(candidates)} ?")
        expanded_names.extend(filtered)
    return expanded_names

def download_by_names(names, **kwargs):
    return download_by_records([r for r in DATASET_REGISTER['records'] if r['name'] in names], **kwargs)

NL = '\n'

def print_all_datasets():
    all_datasets = get_all_datasets()
    print(f"Available datasets are:{NL}{NL}{NL.join(sorted(all_datasets))}{NL}")

def print_local_datasets():
    local_datasets = get_local_datasets()
    print(f"Datasets found locally are:{NL}{NL}{NL.join(sorted(local_datasets))}{NL}")

def print_missing_datasets():
    missing_datasets = get_missing_datasets()
    print(f"Missing datasets are:{NL}{NL}{NL.join(sorted(missing_datasets))}{NL}")

def get_all_datasets():
    return [r['name'] for r in DATASET_REGISTER['records']]

def get_local_datasets():
    return [name for name in get_all_datasets() if get_datapath(name).exists()]

def get_missing_datasets():
    return [name for name in get_all_datasets() if not get_datapath(name).exists()]



# Needs to be packed in
def main():
    import argparse

    all_datasets = [r['name'] for r in DATASET_REGISTER['records']]

    parser = argparse.ArgumentParser(__name__, parents=[config_parser, log_parser])
    e = parser.add_mutually_exclusive_group()
    e.add_argument("--name", nargs='+', default=[], help="List of dataset names to be downloaded. Wildcard are allowed.")
    e.add_argument("--json", action='store_true', help=f"Download datasets from json file (custom selection of datasets).")
    parser.add_argument("--json-files", nargs='+', default=[get_datapath('datasets.json')], help='specify alternative json file(s)')
    parser.add_argument("--ls", action="store_true", help='show all available datasets')
    parser.add_argument("--ls-local", action="store_true", help='list locally available datasets (datasets that have already been downloaded)')
    parser.add_argument("--ls-missing", action="store_true", help='list locally unavailable datasets (datasets that have not been downloaded)')
    parser.add_argument("--all", action='store_true', help='download all available datasets')
    parser.add_argument("--force", action='store_true', help='Extract downloaded files anew, but re-use download cache')
    parser.add_argument("--ignore-cache", action='store_true', help='Ignore download cache to effectively download anew (to be used together with --force)')
    parser.add_argument("--print", action='store_true', help='Show dataset specification (name, url, etc.)')

    o = parser.parse_args()
    setup_logger(o)

    if o.ls:
        print_all_datasets()
        return

    if o.ls_local:
        print_local_datasets()
        return

    if o.ls_missing:
        print_missing_datasets()
        return

    if o.all:
        o.name = all_datasets

    if o.print:
        dataset_by_name = {r['name']: r for r in DATASET_REGISTER['records']}
        datasets_to_print = { "records" : [dataset_by_name[name] for name in o.name if name in dataset_by_name] }
        bad_names = [name for name in o.name if name not in dataset_by_name]
        if bad_names:
            logger.warning(f"Dataset names {' '.join(bad_names)} not found in the register. Available datasets are {' '.join(all_datasets)}")
        def fallback_serializer(obj):
            return str(obj)  # or return None, or raise, depending on your needs
        print(json.dumps(datasets_to_print, indent=2, default=fallback_serializer))
        return

    # download from json file
    if o.json:
        records = []
        for jsfile in o.json_files:
            js = json.load(open(jsfile))
            records.extend(js["records"])
        download_by_records(records, force_download=o.force, ignore_cache=o.ignore_cache)
        return


    # download select only one out of several
    if not o.name:
        print_all_datasets()
        print("Use the --name NAME or --all flag to specify datasets to download.")
        parser.exit(1)

    expanded_names = expand_names(o.name)
    download_by_names(expanded_names, force_download=o.force)


if __name__ == "__main__":
    main()
