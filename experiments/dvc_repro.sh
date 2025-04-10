#!/usr/bin/env bash
dvc status | grep @ | tr -d : | parallel --delay 25 dvc exp run --queue {}
