#!/bin/bash
set -ux

dvc remote add --local local /data/mctm/
dvc pull -r local

set -eu

dvc exp run $@
git add dvc.lock
gc -m "updates dvc.lock from cml runner"

dvc push -r local

git config --global user.email "${GITLAB_USER_EMAIL}"
git config --global user.name "${GITLAB_USER_NAME}"

git remote set-url --push origin git@gitlab:${CI_PROJECT_PATH}
git push --follow-tags origin HEAD:${CI_COMMIT_REF_NAME}
