#!/bin/bash
set -ux

dvc remote add --force --local local /data/mctm/
dvc pull -r local

set -eu

dvc exp run $@

dvc push -r local
