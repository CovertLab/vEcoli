#!/bin/sh
# Start an interactive Docker or Apptainer container from an image.
# Supports optional bind mounts and Cloud Storage bucket mounting

set -eu  # Exit on any error or unset variable

# Default configuration variables
WCM_IMAGE="${USER}-wcm-code"  # Default image name for Docker/Apptainer
USE_APPTAINER=0              # Flag: Use Apptainer if set to 1
BIND_MOUNTS=()               # Array for bind mount paths
BIND_CWD=""                  # Formatted bind mount string for runtime
BUCKET=""                    # Cloud Storage bucket name

# Help message string
usage_str="Usage: interactive.sh [-w WCM_IMAGE] [-a] [-b] [-p]...\n\
Options:\n\
    -w: Path of Apptainer image if -a, otherwise name of Docker \
image inside vecoli Artifact Repository; defaults to "$USER-wcm-code".\n\
    -a: Load Apptainer image.\n\
    -b: Name of Cloud Storage bucket to mount inside container; first mounts
bucket to VM at $HOME/bucket_mnt using gcsfuse (does not work with -a).\n\
    -p: Path(s) to mount inside container; can specify multiple with \
\"-p path1 -p path2\"\n"

# Function to print usage instructions
print_usage() {
  printf "$usage_str"
}

# Parse command-line options
while getopts 'w:ab:p:' flag; do
  case "${flag}" in
    w) WCM_IMAGE="${OPTARG}" ;;                              # Set custom image name
    a) USE_APPTAINER=1 ;;                                    # Enable Apptainer mode
    b) BUCKET="${OPTARG}" ;;                                 # Set the Cloud Storage bucket
    p) BIND_MOUNTS+=($(realpath "${OPTARG}")) ;;             # Convert path to absolute and add to array
    *) print_usage                                           # Print usage for unknown flags
       exit 1 ;;
  esac
done

# Apptainer-specific logic
if (( $USE_APPTAINER )); then
    # If there are bind mounts, format them for Apptainer
    if [ ${#BIND_MOUNTS[@]} -ne 0 ]; then
      BIND_CWD=$(printf " -B %s" "${BIND_MOUNTS[@]}")
    fi
    echo "=== Launching Apptainer container from ${WCM_IMAGE} ==="
    # Start Apptainer container with bind mounts
    apptainer exec -e --writable-tmpfs ${BIND_CWD} ${WCM_IMAGE} bash -c "uv sync --frozen && bash"
else
    # Docker-specific logic
    # Get GCP project name and region to construct image path
    PROJECT=$(gcloud config get project)                     
    REGION=$(gcloud config get compute/region)
    WCM_IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/vecoli/${WCM_IMAGE}"

    # If there are bind mounts, format them for Docker
    if [ ${#BIND_MOUNTS[@]} -ne 0 ]; then
      BIND_CWD=$(printf " -v %s:%s" "${BIND_MOUNTS[@]}" "${BIND_MOUNTS[@]}")
    fi

    # Mount the cloud storage bucket using gcsfuse if provided
    if [ -n "$BUCKET" ]; then
      echo "=== Mounting Cloud Storage bucket ${BUCKET} ==="
      # Create mount point and mount bucket with gcsfuse
      mkdir -p $HOME/bucket_mnt
      gcsfuse --implicit-dirs $BUCKET $HOME/bucket_mnt
      # Nextflow mounts bucket to /mnt/disks so we need to copy that for
      # symlinks to work properly
      BIND_CWD="${BIND_CWD} -v ${HOME}/bucket_mnt:/mnt/disks/${BUCKET}"
    fi

    # Launch the Docker container
    echo "=== Launching Docker container from ${WCM_IMAGE} ==="
    docker container run -it ${BIND_CWD} ${WCM_IMAGE} bash   # Start Docker container with bind mounts
fi
