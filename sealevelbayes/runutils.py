import re
from pathlib import Path
import os
import subprocess as sp

import sealevelbayes
from sealevelbayes.config import logger, get_runpath, get_logpath

import io
import sys
import contextlib

class CaptureOutput:
    def __init__(self, display_output=False, merge_output=False):
        self.stdout = io.StringIO()
        if merge_output:
            self.stderr = self.stdout
        else:
            self.stderr = io.StringIO()
        self.merge_output = merge_output
        self.display_output = display_output

    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, *args):
        sys.stdout = self._stdout
        sys.stderr = self._stderr

    def write(self, data):
        if self.display_output:
            self._stdout.write(data)
            self._stdout.flush()
        self.stdout.write(data)

    def flush(self):
        if self.display_output:
            self._stdout.flush()
            self._stderr.flush()
        self.stdout.flush()

    def get_output(self):
        return self.stdout.getvalue(), self.stderr.getvalue()





ALTCIDIRS = [str(get_runpath()), "/p/projects/ou/rd3/dmcci/perrette/slr-tidegauges-ci-runs", "/p/tmp/perrette/runs"]

# RECIRUN = re.compile(rf'Simulation will be saved in ({"|".join(ALTCIDIRS)})/(\w.*)')
RECIRUN = re.compile(r"\| ID\s*\|\s*([^|]+)\s*\|")
REDURATION = re.compile(rf'took (\d+) seconds.')

def parse_runid(txt):
    """Parse the run ID from the given text."""
    m = RECIRUN.search(txt)
    if m:
        return m.group(1).strip()
    else:
        raise ValueError(f"Could not parse run ID from text: {txt}")

def _get_slurm_files(jobid=None, logdir=None):
    if logdir is None:
        logdir = get_logpath()

    assert jobid is not None
    return Path(logdir)/f"slurm-{jobid}.out"

def parse_slurm_log(slurm_file=None, **kwargs):
    if slurm_file is None:
        slurm_file = _get_slurm_files(**kwargs)

    f = open(slurm_file)
    for l in f.readlines()[:10]:
        m = RECIRUN.search(l)
        if m:
            return m.groups()[1]

def parse_sampling_duration(slurm_file=None, **kwargs):
    if slurm_file is None:
        slurm_file = _get_slurm_files(**kwargs)

    with open(slurm_file) as f:
        for l in f.readlines()[::-1]:
            m = REDURATION.search(l)
            if m:
                return int(m.groups()[0])

    raise ValueError(f"Could not find duration in {slurm_file}")

def find_slurm_files(cirun, logdir=None):
    if logdir is None:
        logdir = get_logpath()
    res = sp.check_output(f"grep '{cirun}' '{logdir}' -rl", shell=True)
    return sorted(res.decode().strip().splitlines(), key=os.path.getmtime)[::-1]

def find_slurm_file(cirun, logdir=None):
    return find_slurm_files(cirun, logdir)[0]


def cdo(cmd):
    logger.info(f"cdo {cmd}")
    return sp.check_call(f"module load cdo/1.9.6/gnu-threadsafe && cdo {cmd}", shell=True, env=os.environ.copy())


def _parse_elapsed_time(time_str):
    """Parse time in mm:ss or hh:mm:ss format and return total hours."""
    if len(time_str.split(':')) == 1:
        return 0

    if len(time_str.split(':')) == 2:
        # mm:ss format
        minutes, seconds = map(int, time_str.split(':'))
        return (minutes * 60 + seconds) / 3600
    elif len(time_str.split(':')) == 3:
        # hh:mm:ss format
        hours, minutes, seconds = map(int, time_str.split(':'))
        return hours + (minutes * 60 + seconds) / 3600
    else:
        raise ValueError(f"Unrecognized time format: {time_str}")

def _parse_sampling_progress(filepath):
    # pattern = re.compile(r"(\d+)%\|[^\|]+\| (\d+)/(\d+) \[(\d+:\d+)<(\d+:\d+), [\d.]+it/s\]")
    # pattern = re.compile(r"\s*(\d+)%\|\s*\|\s*(\d+)/(\d+)\s*\[(\d+:\d+)<(\d+:\d+),\s*[\d.]+it/s\]")
    pattern = re.compile(r"\|.*\|\s(\w.+)%\s\[(\d+)/(\d+)\s(\w.+)<(\w.+)\sSampling\s\d+\schains,\s(\d+)\sdivergences\]")
    results = []
    start_parsing = False
    with open(filepath, 'r') as file:
        for line in file.readlines():
            if not line.strip(): continue
            if "Initializing NUTS" in line:
                start_parsing = True
                continue
            if start_parsing:
                match = pattern.search(line)
                if match:
                    percentage = float(match.group(1))
                    iteration = int(match.group(2))
                    total_iterations = int(match.group(3))
                    time_elapsed = _parse_elapsed_time(match.group(4))
                    time_left = _parse_elapsed_time(match.group(5))
                    divergences = int(match.group(6))
                    results.append({
                        'percentage': percentage,
                        'iteration': iteration,
                        'total_iterations': total_iterations,
                        'time_elapsed': time_elapsed,
                        'time_left': time_left,
                        'divergences': divergences,
                    })
                elif len(results) > 4:
                    # print(line, match)
                    # 1/0
                    # break
                    pass

    return results

def parse_sampling_progress(jobid, filepath=None, logdir=None):
    if filepath is None:
        filepath = find_slurm_file(jobid, logdir)
    return _parse_sampling_progress(filepath)


import ast
import importlib.util
import os
import inspect

# def get_default_kwargs_from_module(module_name, function_name):
#     # Find the module file
#     spec = importlib.util.find_spec(module_name)
#     if spec is None or spec.origin is None or not os.path.isfile(spec.origin):
#         raise ImportError(f"Could not locate source file for module '{module_name}'")

#     # Parse the file
#     with open(spec.origin, "r", encoding="utf-8") as f:
#         tree = ast.parse(f.read(), filename=spec.origin)

#     # Find the target function
#     for node in ast.walk(tree):
#         if isinstance(node, ast.FunctionDef) and node.name == function_name:
#             args = node.args.args
#             defaults = node.args.defaults

#             kwonlyargs = getattr(node.args, "kwonlyargs", [])
#             kw_defaults = getattr(node.args, "kw_defaults", [])

#             # Align positional defaults
#             pos_default_offset = len(args) - len(defaults)
#             default_kwargs = {}

#             for i, arg in enumerate(args):
#                 if i >= pos_default_offset:
#                     default = defaults[i - pos_default_offset]
#                     default_kwargs[arg.arg] = ast.unparse(default)

#             # Handle keyword-only args
#             for kwarg, default in zip(kwonlyargs, kw_defaults):
#                 if default is not None:
#                     default_kwargs[kwarg.arg] = ast.unparse(default)

#             return default_kwargs

#     raise ValueError(f"Function '{function_name}' not found in module '{module_name}'")

import ast
import importlib.util
import os

def get_default_kwargs_from_module(module_name, qualname):
    # Split qualified name: e.g. MyClass.my_method → ['MyClass', 'my_method']
    path = qualname.split(".")

    # Find module source file
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None or not os.path.isfile(spec.origin):
        raise ImportError(f"Could not locate source file for module '{module_name}'")

    with open(spec.origin, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=spec.origin)

    # Recursively descend into class/function structure
    def find_node(path, node):
        if not path:
            return node
        name = path[0]
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.ClassDef)) and child.name == name:
                return find_node(path[1:], child)
        return None

    target = find_node(path, tree)
    if not isinstance(target, ast.FunctionDef):
        raise ValueError(f"Function '{qualname}' not found in module '{module_name}'")

    # Extract defaults
    args = target.args.args
    defaults = target.args.defaults
    kwonlyargs = getattr(target.args, "kwonlyargs", [])
    kw_defaults = getattr(target.args, "kw_defaults", [])

    # Skip 'self' or 'cls' if it's a method
    start = 1 if args and args[0].arg in ('self', 'cls') else 0
    pos_default_offset = len(args) - len(defaults)

    default_kwargs = {}
    for i, arg in enumerate(args[start:], start=start):
        if i >= pos_default_offset:
            default = defaults[i - pos_default_offset]
            # default_kwargs[arg.arg] = ast.unparse(default)
            try:
                value = ast.literal_eval(default)
            except Exception:
                value = ast.unparse(default)
            default_kwargs[arg.arg] = value


    for kwarg, default in zip(kwonlyargs, kw_defaults):
        if default is not None:
            # default_kwargs[kwarg.arg] = ast.unparse(default)
            try:
                value = ast.literal_eval(default)
            except Exception:
                value = ast.unparse(default)
            default_kwargs[kwarg.arg] = value


    # return {k:v for k,v in default_kwargs.items() if v is not None}
    return default_kwargs


def get_default_kwargs_dyn(func):
    sig = inspect.signature(func)
    return {k: sig.parameters[k].default for k in sig.parameters if sig.parameters[k].default is not inspect.Parameter.empty}

def get_default_kwargs(module, funcname, static=False):
    """Get default kwargs from a function in a module, without importing the module"""
    if static:
        return get_default_kwargs_from_module(module, funcname)

    mod = importlib.import_module(module)

    if "." in funcname:
        classname, methodname = funcname.split('.')
        cls = getattr(mod, classname)
        func = getattr(cls, methodname)
    else:
        func = getattr(mod, funcname)

    return get_default_kwargs_dyn(func)

