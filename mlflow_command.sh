#!/bin/bash
set -ux

dvc remote add --local local /data/mctm/
dvc pull -r local

set -eu

dvc exp run $@
git add dvc.lock
gc -m "updates dvc.lock from cml runner"

dvc push -r local
