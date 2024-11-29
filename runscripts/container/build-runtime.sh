#!/bin/sh
# Use Google Cloud Build or local Docker install to build a personalized
# image with requirements.txt installed. If using Cloud Build, store the
# built image in the "vecoli" folder in the Google Artifact Registry.
#
# ASSUMES: The current working dir is the vEcoli/ project root.

set -eu

RUNTIME_IMAGE="${USER}-wcm-runtime"
RUN_LOCAL='false'

usage_str="Usage: build-runtime.sh [-r RUNTIME_IMAGE] [-l]\n\
    -r: Docker tag for the wcm-runtime image to build; defaults to \
${USER}-wcm-runtime\n\
    -l: Build image locally.\n"

print_usage() {
  printf "$usage_str"
}

while getopts 'r:l' flag; do
  case "${flag}" in
    r) RUNTIME_IMAGE="${OPTARG}" ;;
    l) RUN_LOCAL="${OPTARG}" ;;
    *) print_usage
       exit 1 ;;
  esac
done

# This needs only one payload file so copy it in rather than using a config at
# the project root which would upload the entire project.
cp requirements.txt runscripts/container/runtime/

if [ "$RUN_LOCAL" = true ]; then
    echo "=== Locally building WCM runtime Docker Image: ${RUNTIME_IMAGE} ==="
    docker build -f runscripts/container/runtime/Dockerfile -t "${WCM_RUNTIME}" .
else
    echo "=== Cloud-building WCM runtime Docker Image: ${RUNTIME_IMAGE} ==="
    # For this script to work on a Compute Engine VM, you must
    # - Set default Compute Engine region and zone for your project
    # - Set access scope to "Allow full access to all Cloud APIs" when
    #   creating VM
    # - Run gcloud init in VM
    REGION=$(gcloud config get compute/region)
    # This needs a config file to identify the project files to upload and the
    # Dockerfile to run.
    gcloud builds submit --timeout=3h --region=$REGION --tag \
      '${LOCATION}-docker.pkg.dev/${PROJECT_ID}/vecoli/'${RUNTIME_IMAGE} \
      runscripts/container/runtime/
fi

rm runscripts/container/runtime/requirements.txt
