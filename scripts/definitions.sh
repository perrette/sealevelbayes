#!/bin/bash
# This script is used to submit jobs to the HPC cluster

# The  CPU of the new HPC system is the AMD EPYC 9554 64 core at 3.1GHz (code name Genoa),
# which will be installed in pairs on each compute node, giving us 128 CPU cores per
# node. Total RAM per node will be 768 GB (6 GB per core). If you use multithreaded
# applications which assume a full node has 16 or 32 CPU cores, you may want to test
# the effect (if any) of scaling to more cores. Be aware that many software
# applications assume use of all visible CPU cores so will try to automatically scale
# to 128 threads. See [1] for the spec sheet.
# [1] https://www.amd.com/en/products/cpu/amd-epyc-9554

# set -euo pipefail

: "${RUNQOS:="--cpus-per-task=16 --qos short"}"
: "${NBQOS:="--mem=64G --qos short"}"
: "${WEBQOS:="--mem=64G --qos short"}"
: "${ZENQOS:="--qos priority"}"

: "${VENV:=$($VENV/bin/sealevelbayes-config virtualenv)}"
: "${REF:=$($VENV/bin/sealevelbayes-config version)}"
: "${RUNDIR:=$($VENV/bin/sealevelbayes-config rundir)}"
: "${WWW:=$($VENV/bin/sealevelbayes-config webdir)}"
: "${COMPILEDIR:=$($VENV/bin/sealevelbayes-config compiledir)}"

# the flags below should be set individually for the various run commands
# the default is for runbundle
INTERACTIVE=0
SENSFIG=0
WEB=0
ZENODO=0
RUN=0
CHECK=0
FIG=0
LOCALFIG=0
GLOBALFIG=0
NUM=0
SENSTRACE=0
VERBOSE=0
ARGS=""
JOBS=""


function print_header() {
    echo "-------------------------------------------"
    echo " Batch job definitions                     "
    echo "-------------------------------------------"
    echo "Python environment                 : $VENV"
    echo "Default runs directory             : $RUNDIR"
    echo "Default fig directory              : $WWW"
    echo "Default experiment group (version) : $REF"
    echo "-------------------------------------------"
}

function take_a_break() {
    echo "Sleeping for 10 seconds"
    sleep 10
}

counter=0

function runcmd() {
    local cmd="$1"              # First argument is the command to run
    shift                       # Remaining arguments are SLURM options
    local slurm_args="$@"       # Optional SLURM args go here

    # Auto-incrementing compiledir
    counter=$((counter + 1))

    # Log file structure: jobs/2025w13.log
    local jobfile="jobs/$(date +%-Yw%-V).log"
    [ -d jobs ] || mkdir -p jobs
    [ -d logs ] || mkdir -p logs

    # Echo command for history logging
    echo "$cmd  # $(date)" >> "$jobfile"

    if [[ $INTERACTIVE == 1 ]] ; then
        echo "Running interactively: $cmd"
        PYTENSOR_FLAGS="base_compiledir=$COMPILEDIR/$counter" \
        eval "$cmd"
        return
    fi

    if [[ $VERBOSE == 1 ]] ; then
        echo "Submitting command: $cmd" > /dev/tty
        echo "SLURM args: $slurm_args" > /dev/tty
    fi

    # Wrap and submit the command
    PYTENSOR_FLAGS="base_compiledir=$COMPILEDIR" \
    sbatch --parsable -o logs/slurm-%j.out $slurm_args --wrap="$cmd" \
        2>&1 | tee -a "$jobfile"
}

function getlogfile() {
    # get the latest log file
    JOBID=$1
    LOGFILE=$(ls -t logs/slurm-$JOBID*.out | head -1)
    if [ -z $LOGFILE ] ; then
        echo "No log file found"
        return 1
    fi
    echo $LOGFILE
}

function parselogforrunid() {
    # parse the log file for the runid
    LOGFILE=$1
    RUNID=$(grep -m1 "| ID  |" $LOGFILE | cut -d'|' -f3 | sed 's/ //g')
    if [[ $RUNID == "" ]] ; then
        # fom /p/tmp/perrette/runs/v0.3.3-v2/run_v0.3.3-v2_glacier-v2-n0.76 remove /p/tmp/perrette/runs/
        RUNID=$(grep "Simulation will be saved in" "$LOGFILE" | awk -F 'in ' '{print $2}'  | sed 's/\/p\/tmp\/perrette\/runs\///g')
    fi
    echo $RUNID
}

function getrunid() {
    set -e
    jobid=$1
    if [[ -z $jobid ]] ; then
        echo "Usage: getrunid <jobid>"
        return 1
    fi
    jobfile=$(getlogfile $jobid)
    runid=$(parselogforrunid $jobfile)
    if [[ -z $runid ]] ; then
        echo "No runid found in log file $jobfile"
        return 1
    fi
    echo $runid
}

function getjobstate() {
    jobid=$1
    if [[ -z $jobid ]] ; then
        echo "Usage: getjobstate <jobid>"
        return 1
    fi
    state=$(sacct -j "$jobid" --format=State --noheader | head -n 1 | awk '{print $1}')
    if [[ -z $state ]] ; then
        echo "No state found for job $jobid"
        return 1
    fi
    echo $state
}

function checkstatus() {
    local jobs="$@"
    for j in $jobs ; do
        state=$(getjobstate $j)
        if [ $state == "PENDING" ] ; then
            runid="?"
        else
            runid=$(getrunid $j)
        fi
        echo "$j $state $runid"
    done
}

function runit() {
    if [[ -z $REF ]] ; then
        echo "Please set REF environment variable to the desired reference version."
        return 1
    fi
    local args="$ARGS $*"
    if [[ "$FIG" == 1 ]] ; then
        args+=" --figs"
    fi
    CMD="$VENV/bin/sealevelbayes-run $args --ref $REF $*"
    if [[ "$CHECK" == 1 ]] ; then
        eval "$CMD"
    else
        runcmd "$CMD" $RUNQOS --cpus-per-task=16
    fi
}


function runglb() {
    runit --global-slr-only $@
}

function runtwo() {
    runit --global-slr-only $@
    runit $@
}

function default() {
    if [[ -n $VENVDEFAULT ]] ; then
        VENV=$VENVDEFAULT
    fi
    if [[ -n $DEFAULTREF ]] ; then
        REF=$DEFAULTREF
    fi
    VENV=$VENV REF=$REF $@;
}

function showenv {
    echo "VENV: $VENV"
    echo "REF: $REF"
    $VENV/bin/sealevelbayes-run --version
}

function rundefault() {
    default runit $@
}

function run_all() {
    runit $@ --global-slr-only
    runit $@
    runit $@ --global-slr-only --greenland-future-constraint ar6-low-confidence --antarctica-future-constraint ar6-low-confidence
    runit $@ --greenland-future-constraint ar6-low-confidence --antarctica-future-constraint ar6-low-confidence

    runit $@ --skip-constraints tidegauge
    runit $@ --skip-constraints satellite
    runit $@ --skip-constraints gps
    runit $@ --skip-constraints tidegauge satellite gps
    runit $@ --skip-constraints satellite gps
    # runit $@ --skip-constraints satellite tidegauge
    # runit $@ --skip-constraints tidegauge gps

    # for scale in 1 10 20 50 200 500 1000 2000; do
    for scale in 1 10 1000; do
        runit $@ --vlm-res-spatial-scale $scale
    done

# }
# function sensitivity_constraints_obserror() {
    runit $@ --satellite-measurement-error-method constant
    runit $@ --no-estimate-tidegauge-measurement-error
    runit $@ --satellite-measurement-error-method constant --no-estimate-tidegauge-measurement-error
    runit $@ --wind-correction

    run_leave_out_50p $@  # new default is to leave out all obs types

    # run_leave_out_basins $@  # for future analysis
# }
}

function run_leave_out_50p() {
    for i in {1..10}; do
        if [[ -z $TIDEGAUGE_SEED ]]; then
            runit $@ --leave-out-tidegauge-fraction 0.5
        else
            runit $@ --leave-out-tidegauge-fraction 0.5 --random-seed $(($TIDEGAUGE_SEED + $i))
        fi
    done
}

function run_leave_out_basins() {
    runit $@ --leave-out-tidegauge-basin-id 1
    runit $@ --leave-out-tidegauge-basin-id 2
    runit $@ --leave-out-tidegauge-basin-id 3
    runit $@ --leave-out-tidegauge-basin-id 4
    runit $@ --leave-out-tidegauge-basin-id 5
    runit $@ --leave-out-tidegauge-basin-id 6
    runit $@ --leave-out-tidegauge-basin-id 8
}

function runnotebook() {
    NAME="$1"
    RUNID="$2"
    shift 2  # Remove the first two arguments
    if [[ -z $NAME ]] || [[ -z $RUNID ]] ; then
        echo "Usage: runnotebook <notebook_name> <runid>"
        return 1
    fi
    IPYNB=$(find ./notebooks -name "${NAME}.ipynb" | head -n 1)
    if [[ -z $IPYNB ]] ; then
        echo "Notebook $NAME not found in ./notebooks"
        return 1
    fi
    HTML="$WWW/$RUNID/notebooks/${NAME}.html"
    mkdir -p $(dirname $HTML)
    CMD="$VENV/bin/jupyter nbconvert --to html --execute --allow-errors $IPYNB --output $HTML"
    echo "Executing notebook: $IPYNB (output: $HTML)" > /dev/tty
    CIRUN="$RUNID" runcmd "$CMD" "$NBQOS $*"
}

function is_valid_jobid() {
    if [[ -z "$1" ]] || [[ ! "$1" =~ ^[0-9]+$ ]]; then
        return 1  # Invalid job ID
    fi
    # Check if the job ID exists in the SLURM system
    scontrol show jobid "$1" &>/dev/null
}

function getparam() {
    runid="$1"
    name="$2"
    if [[ -z $runid ]] || [[ -z $name ]] ; then
        echo "Usage: getparam <runid> <parameter_name>"
        return 1
    fi
    paramfile="$RUNDIR/$runid/options.json"
    if [[ ! -f $paramfile ]] ; then
        echo "Parameter file $paramfile not found"
        return 1
    fi
    result=$($VENV/bin/sealevelbayes-runid --query $name --param-file $paramfile)
    echo $result
}

function getconfigparam() {
    name="$1"
    if [[ -z $name ]] ; then
        echo "Usage: getconfigparam <parameter_name>"
        return 1
    fi
    result=$($VENV/bin/sealevelbayes-config $name)
    echo $result
}

function is_local_run() {
    runid="$1"
    #TODO: Implement logic to determine if the run is local
    global_slr_only=$(getparam "$runid" global_slr_only)
    if [[ $global_slr_only == "True" ]] ; then
        return 1  # Not a local run
    fi
    return 0  # 0 is true
}

function getdep() {
    # Run a command with a dependency on the previous job
    # Usage: getdep <jobid>
    # returns -d afterany:<jobid> if jobid is valid
    # If more jobs
    # Usage: getdep <jobid> <jobid2> ...
    # returns -d afterany:<jobid>:<jobid2>:... for any valid jobid

    if [[ -z "$1" ]]; then
        echo ""  # No job ID provided, no dependency
        return 0
    fi

    # Multiple job IDs provided, join them with ":"
    local jobids=("$@")
    local valid_jobids=()
    for jobid in "${jobids[@]}"; do
        if is_valid_jobid "$jobid"; then
            valid_jobids+=("$jobid")
        fi
    done
    if [[ ${#valid_jobids[@]} -eq 0 ]]; then
        echo ""  # No valid job IDs, no dependency
        return 0
    fi
    dep="-d afterany:$(IFS=:; echo "${valid_jobids[*]}")"
    echo $dep
}


function runbundle () {

    if [[ -z $REF ]] ; then
        echo "Please set REF environment variable to the desired reference version."
        return 1
    fi

    if [[ $RUN == 1 ]] ; then
        if [[ ! -z $RUNIDS ]] ; then
            echo "Cannot run with RUNIDS set. Please unset RUNIDS or set RUN to 0."
            return 1
        fi
        JOBS=$(run_all | tr '\n' ' ' | xargs)
        echo "Submitted runid jobs: $JOBS"
    fi

    if [[ -z "$JOBS" ]] ; then
        : "${RUNIDS:=$(getconfigparam experiments | tr "\n" " " | xargs)}"
        list_runids=($RUNIDS)
        # create a list of empty jobid
        list_jobids=()
        for runid in "${list_runids[@]}"; do
            list_jobids+=("")
        done
    else
        list_runids=()
        for job in $JOBS; do
            runid=$(getrunid $job)
            list_runids+=("$runid")
        done
        RUNIDS=$(printf "%s " "${list_runids[@]}")
    fi

    # hooks for individual jobs
    list_webid=()
    list_localfigid=()
    list_globalfigid=()
    list_numid=()
    list_sensid=()
    for i in ${!list_runids[@]}; do
        runid=${list_runids[$i]}
        job=${list_jobids[$i]}
        dep=$(getdep "$job")
        if [[ $WEB == 1 ]]  ; then
            webid=$(runcmd "$VENV/bin/sealevelbayes-web $runid" $dep $WEBQOS)
            list_webid+=("$webid")
        fi
        if [[ $FIG == 1 || $GLOBALFIG == 1 ]] ; then
            list_globalfigid+=("$(runnotebook figures-global $runid $dep)")
        fi
        if [[ $FIG == 1 || $LOCALFIG == 1 ]] ; then
            if is_local_run $runid ; then
                list_localfigid+=("$(runnotebook figures-local $runid $dep)")
            fi
        fi
        if [[ $NUM == 1 ]] ; then
            list_numid+=("$(runnotebook numbers $runid $dep)")
        fi
        if [[ $SENSTRACE == 1 ]] ; then
            pycode="from sealevelbayes.postproc.sensitivityfigs import crunch_one ; crunch_one($runid)";
            cmd="$VENV/bin/python -c \"$pycode\""
            sensid=$(runcmd "$cmd" $dep --mem=64G --qos short)
            list_sensid+=("$sensid")
        fi
    done

    # Print the output
    if [[ ${#list_webid[@]} > 0 ]] ; then echo "Submitted web jobs: ${list_webid[@]}"; fi
    if [[ ${#list_localfigid[@]} > 0 ]] ; then echo "Submitted local fig jobs: ${list_localfigid[@]}"; fi
    if [[ ${#list_globalfigid[@]} > 0 ]] ; then echo "Submitted global fig jobs: ${list_globalfigid[@]}"; fi
    if [[ ${#list_numid[@]} > 0 ]] ; then echo "Submitted number jobs: ${list_numid[@]}"; fi
    if [[ ${#list_sensid[@]} > 0 ]] ; then echo "Submitted sensitivity trace jobs: ${list_sensid[@]}"; fi

    # hooks for all jobs

    # Join job IDs with ":" separator
    DEP=$(getdep "${JOBS[@]}")

    if [[ $SENSFIG == 1 ]] ; then
        sensjobs=$(runnotebook figures-sensitivity $REF/sensitivity $DEP | tr '\n' ' ' | xargs)
        valjobs=$(runnotebook figures-validation $REF/sensitivity $DEP | tr '\n' ' ' | xargs)
        echo "Submitted sensitivity fig jobs: $sensjobs"
        echo "Submitted validation fig jobs: $valjobs"
    fi

    if [[ $ZENODO == 1 ]] ; then
        ZENDEP=$(getdep "${list_webid[@]}")
        ZENID=$(runcmd "$VENV/bin/sealevelbayes-zenodo --update-all" $ZENDEP --mem=64G)
        echo "Submitted Zenodo update job: $ZENID"
    fi

}

