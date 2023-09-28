#!/bin/bash
set -ux

dvc remote add --force --local local /data/mctm/
dvc pull -r local

# set -eu

dvc repro malnutrition

dvc push -r local
