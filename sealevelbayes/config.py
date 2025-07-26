import os
import sys
import argparse
import subprocess as sp
from pathlib import Path
from flatdict import FlatDict
import toml

import sealevelbayes
from sealevelbayes.logs import logger, setup_logger

def get_version():
    try:
        return sp.check_output("git rev-parse --short HEAD", shell=True).decode().strip()
    except sp.CalledProcessError:
        from sealevelbayes import __version__  ## !!! --> this will show last time an editable version was used...
        return __version__

def get_runpath_ci(experiment="", ref=None):
    return get_runpath(ref or get_version()) / experiment

def load_config(file):
    with open(file) as f:
        cfg = toml.load(f)
    return FlatDict(cfg, delimiter=".")

def get_runpath(experiment=""):
    return Path(CONFIG['rundir']) / experiment

def get_webpath(experiment=""):
    return Path(CONFIG.get('webdir', get_runpath("web"))) / experiment

def get_url(experiment=""):
    from urllib.parse import urljoin
    HOME = os.getenv("HOME")
    try:
        relpath = str(get_webpath().relative_to(f"{HOME}/www/"))
        rooturldef = urljoin("https://www.pik-potsdam.de/~perrette/slr-tidegauges/", relpath)
    except:
        rooturldef = "file://"+str(get_webpath().resolve())
    return urljoin(CONFIG.get('url', rooturldef), experiment)

def get_sharedpath(relpath=""):
    return Path(CONFIG.get('shared', get_runpath("shared"))) / relpath

def get_logpath(relpath=""):
    return Path(CONFIG.get('logdir', get_runpath("logs"))) / relpath

def get_experiment_info(runid):
    if not runid: # runid is None
        return {}
    runpath = get_runpath(runid)
    figdir = get_webpath(runid) / "figures"
    url = get_url(runid) + "/figures"
    return {
        "ID": runid,
        "RUN": str(runpath),
        "FIG": str(figdir),
        "WWW": url,
    }

def format_experiment_info(meta, backend=None):

    nkey = max(len(k) for k in meta.keys())
    nval = max(len(v) for v in meta.values())
    n = nkey+4 + nval+ 3
    fmtkey = f"{{0:<{nkey}}}"
    fmtval = f"{{1:<{nval}}}"
    if backend == "markdown":
        sepheader = f"| {fmtkey} | {fmtval} |".format("---", "---")
        sep = None
        header = f"| {fmtkey} | {fmtval} |".format("<!-- -->", "<!-- -->")
        footer = None
    else:
        sep = "-" * n
        sepheader = sep
        header = ""
        footer = sep
    lines = []
    if header:
        lines.append(header)
    if sepheader: lines.append(sepheader)
    for k, v in meta.items():
        lines.append(f"| {fmtkey} | {fmtval} |".format(k, v))
        if sep: lines.append(sep)
    if footer:
        lines[-1] = footer
    else:
        lines = lines[:-1]
    return "\n".join(lines)


def print_experiment_info(runid, backend=None, **kwargs):
    """print experiment info
    """
    meta = get_experiment_info(runid)
    meta.update(kwargs)
    info = format_experiment_info(meta, backend=backend)
    print(info)


_REPODIR = Path(sealevelbayes.__file__).parent.parent  # this makes sense for editable pip install, otherwise less so

def detect_app_config():
    """define default paths (this will be updated by the config file, if any is found / provided)
    """
    appconfig = {
        "compiledir": str(Path(os.getenv("XDG_CACHE_HOME", os.path.join(os.getenv("HOME"), ".cache"))) / sealevelbayes.__name__ / "pytensor"),
        "virtualenv": sys.prefix,
        "downloaddir": str(CACHE_FOLDER / "download"),
        "datadir": str(Path(os.getenv("XDG_CACHE_HOME", os.path.join(os.getenv("HOME"), ".cache"))) / sealevelbayes.__name__ / "data"),
        "rundir": str(_REPODIR / "runs"),
        "isimipdir": "/p/projects/isimip/isimip",
    }

    return FlatDict(appconfig, delimiter=".")

# Source: https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html
_HOME = os.getenv("HOME")
_CACHE_FOLDER_SYSTEM = os.getenv("XDG_CACHE_HOME", os.path.join(_HOME, ".cache"))
CACHE_FOLDER = Path(_CACHE_FOLDER_SYSTEM) / sealevelbayes.__name__

# _CONFIG_FOLDER_SYSTEM = os.getenv("XDG_CONFIG_HOME", os.path.join(_HOME, ".config"))
# GLOBAL_CONFIG_FILE = Path(_CONFIG_FOLDER_SYSTEM) / (sealevelbayes.__name__ + ".toml")

DEFAULT_CONFIG = detect_app_config()
DEFAULT_CONFIG.update({
        "download_facts_data": False,
        })

CONFIG = DEFAULT_CONFIG.copy()

def search_default_config():
    # seach current working directory
    candidates = ['sealevelbayes.toml']
    candidates.append(Path(os.environ.get("HOME","")) / '.config/sealevelbayes.toml')
    candidates.append(Path(os.environ.get("HOME","")) / '.sealevelbayes.toml')
    candidates.append(_REPODIR / 'config.toml')
    candidates.append('./config.toml')
    candidates.append('./../config.toml') # for notebooks

    for candidate in candidates:
        if Path(candidate).exists():
            logger.debug(f"Found config file: {candidate}")
            return str(candidate)

    # not found, so return default
    logger.debug(f"No config file found, use defaults.")
    # return DEFAULT_CONFIG_FILE

config_parser = argparse.ArgumentParser(add_help=False)
g = config_parser.add_argument_group("config")
g.add_argument("--version", action='store_true')
g.add_argument("--config-file", default=search_default_config())

o, _ = config_parser.parse_known_args()

if o.version:
    from sealevelbayes._version import __version__
    print(__version__)
    config_parser.exit(0)

def set_config(file_path):
    global CONFIG

    CONFIG = load_config(file_path)

    # update undefined fields with defaults
    for field, default_value in DEFAULT_CONFIG.items():
        CONFIG.setdefault(field, default_value)

if o.config_file:
    set_config(o.config_file)

def print_config(cfg=None):
    print(toml.dumps((cfg or CONFIG).as_dict()))

def main():
    """show configuration file"""
    parser = argparse.ArgumentParser(parents=[config_parser])
    parser.add_argument("name", nargs="?", default=None, help="Name of the configuration to print")
    o = parser.parse_args()
    if o.name:
        value = CONFIG[o.name]
        if type(value) is list:
            print("\n".join(value))
        else:
            print(value)
        return 0
    print_config()
    parser.exit(0)


if __name__ == "__main__":
    main()
