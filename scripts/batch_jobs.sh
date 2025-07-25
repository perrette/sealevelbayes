#!/bin/bash
#
# -----------------------------------------------------------------------------
# This script is used to submit jobs to the HPC cluster
#
# Relies on `./scripts/definitions.sh` : defines various helper functions.
VENV="/p/projects/isipedia/perrette/sealevel/slr-tidegauges-future/.venv-pymc59-py311"
REF="v0.3.4"
source ./scripts/definitions.sh
print_header
module load nco  # useful for web
# The environment variables are not overwritten (a default is set inside if not already defined).
#
# Example of commands defined in scripts/definitions.sh:
#
# `runcmd "<command>" <slurm args...> ...`: run a command with slurm options. It uses `sbatch --wrap` and write logs and define pytensor compile dir in the right place.
#
#     runcmd "sealevelbayes-run --figs --json --cirun v0.3.4/run_v0.3.4_leave-out-gps-sat-tg-526523 --resume" --cpu-per-task=16 --qos medium --time 30:00:00
#
# `checkstatus jobid1 jobid2 ...`
#
# `runit ...` : wrapper around sealevelbayes-run
#
# `runnotebook <nobebook name> <runid>` : wrapper around jupyter nbconvert
#
#     runnotebook figures-global v0.3.4/run_v0.3.4
#     runnotebook figures-validation v0.3.4/sensitivity
#
# `run_all` or `run_all --figs`: run all sensitivity experiments
#
# `runbundle`: run all of the above, with slurm workflow management. By default does nothing, unless you set the environment variables RUN, FIG, SENSTRACE, SENSFIG, WEB to 1.
#
# Examples
# --------
#
# Run all experiments and make all figures and web data, with appropriate workflow management:
#
#     RUN=1 FIG=1 SENSTRACE=1 SENSFIG=1 WEB=1 runbundle
#
# Crunch all figures when the jobs 992332 and 990001 are done, for the corresponding runs:
#
#     JOBS="992332 990001" FIG=1 runbundle
#
# Do all the postprocessing for these two runs
#
#     RUNDIDS="v0.3.4/run_v0.3.4_leave-out-gps-sat-tg-526523 v0.3.4/run_v0.3.4_leave-out-gps-sat-tg-526524" FIG=1 SENSTRACE=1 SENSFIG=1 WEB=1 runbundle
#
# -----------------------------------------------------------------------------

# set -euo pipefail
RUNQOS="--cpus-per-task=16 --qos short"
NBQOS="--mem=64G --qos short"
WEBQOS="--mem=64G --qos short"
ZENQOS="--qos priority"

RUN=1 FIG=1 SENSTRACE=1 SENSFIG=1 WEB=1 runbundle