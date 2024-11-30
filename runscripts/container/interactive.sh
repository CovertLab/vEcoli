#!/bin/sh
# Start an interactive Docker or Apptainer container from image built using
# build-wcm.sh

set -eu

WCM_IMAGE="${USER}-wcm-code"
BUILD_APPTAINER=0

usage_str="Usage: interactive.sh [-w WCM_IMAGE] [-b BIND_PATH] [-a]\n\
Options:\n\
    -w: Path of Apptainer wcm-code image to load if -a, otherwise Docker \
tag; defaults to "$USER-wcm-code".\n\
    -a: Load Apptainer image.\n"

print_usage() {
  printf "$usage_str"
}

while getopts 'w:a' flag; do
  case "${flag}" in
    w) WCM_IMAGE="${OPTARG}" ;;
    a) BUILD_APPTAINER=1 ;;
    *) print_usage
       exit 1 ;;
  esac
done

if (( $BUILD_APPTAINER )); then
    echo "=== Launching Apptainer container from ${WCM_IMAGE} ==="
    apptainer shell -e ${WCM_IMAGE}
else
    echo "=== Launching Docker container from ${WCM_IMAGE} ==="
    docker container run -it ${WCM_IMAGE} bash
fi
