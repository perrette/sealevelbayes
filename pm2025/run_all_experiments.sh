#!/usr/bin/bash

alias runcommand="sealevelbayes-run"

function writeconfig() {
    runid=$(sealevelbayes-runid $@)
    config="configs/$runid.json"
    mkdir -p $(dirname "$config")
    sealevelbayes-runid $@ --print > "$config"
}

function runcommand() {
    writeconfig $@ --ref v0.3.4
    # sealevelbayes-run "$@"
}

runcommand
runcommand --greenland-future-constraint ar6-low-confidence --antarctica-future-constraint ar6-low-confidence
runcommand --skip-constraints tidegauge
runcommand --skip-constraints satellite
runcommand --skip-constraints gps
runcommand --skip-constraints tidegauge satellite gps
runcommand --skip-constraints satellite gps

for scale in 1 10 1000; do
    runcommand --vlm-res-spatial-scale $scale
done

runcommand --satellite-measurement-error-method constant
runcommand --no-estimate-tidegauge-measurement-error
runcommand --satellite-measurement-error-method constant --no-estimate-tidegauge-measurement-error
runcommand --wind-correction

TIDEGAUGE_SEED=42
for i in {1..10}; do
    if [[ -z $TIDEGAUGE_SEED ]]; then
        runcommand --leave-out-tidegauge-fraction 0.5
    else
        runcommand --leave-out-tidegauge-fraction 0.5 --random-seed $(($TIDEGAUGE_SEED + $i))
    fi
done