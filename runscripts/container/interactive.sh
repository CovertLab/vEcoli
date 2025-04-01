#!/bin/bash
# Start an interactive Docker or Apptainer container from an image.
# Supports optional bind mounts and Cloud Storage bucket mounting

set -eu  # Exit on any error or unset variable

unmount() {
  fusermount -u $HOME/bucket_mnt &>/dev/null || true
}

# Ensure bucket is unmounted on script exit
trap unmount EXIT

# Default configuration variables
IMAGE_NAME="${USER}-code-image"  # Default image name for Docker/Apptainer
USE_APPTAINER=0              # Flag: Use Apptainer if set to 1
IS_RUNTIME_IMAGE=0           # Flag: Is this a runtime image (vs code image)
BIND_MOUNTS=()               # Array for bind mount paths
BIND_CWD=""                  # Formatted bind mount string for runtime
BUCKET=""                    # Cloud Storage bucket name

# Help message string
usage_str="Usage: interactive.sh [-w IMAGE_NAME] [-a] [-r] [-b] [-p]...\n\
Options:\n\
    -w: Path of Apptainer image if -a, otherwise name of Docker \
image inside vecoli Artifact Repository; defaults to \"$IMAGE_NAME\".\n\
    -a: Load Apptainer image.\n\
    -r: Treat as runtime image (automatically run 'uv sync --frozen' before shell).\n\
    -b: Name of Cloud Storage bucket to mount inside container; first mounts
bucket to VM at $HOME/bucket_mnt using gcsfuse (does not work with -a).\n\
    -p: Path(s) to mount inside container; can specify multiple with \
\"-p path1 -p path2\"\n"

# Function to print usage instructions
print_usage() {
  printf "$usage_str"
}

# Parse command-line options
while getopts 'w:arb:p:' flag; do
  case "${flag}" in
    w) IMAGE_NAME="${OPTARG}" ;;                              # Set custom image name
    a) USE_APPTAINER=1 ;;                                    # Enable Apptainer mode
    r) IS_RUNTIME_IMAGE=1 ;;                                 # Mark as runtime image
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
    
    echo "=== Launching Apptainer container from ${IMAGE_NAME} ==="
    
    # Different handling based on image type
    if (( $IS_RUNTIME_IMAGE )); then
        echo "Runtime image detected. Will run 'uv sync --frozen' before launching shell."
        # For runtime images, run uv sync before the shell
        apptainer exec -e --writable-tmpfs ${BIND_CWD} ${IMAGE_NAME} bash -c "uv sync --frozen && exec bash"
    else
        echo "Code image detected. Launching shell directly."
        # For code images, just start a shell
        apptainer exec -e --writable-tmpfs ${BIND_CWD} ${IMAGE_NAME} bash
    fi
else
    # Docker-specific logic
    # Get GCP project name and region to construct image path
    PROJECT=$(gcloud config get project)                     
    REGION=$(gcloud config get compute/region)
    IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT}/vecoli/${IMAGE_NAME}"

    # If there are bind mounts, format them for Docker
    if [ ${#BIND_MOUNTS[@]} -ne 0 ]; then
      BIND_CWD=$(printf " -v %s:%s" "${BIND_MOUNTS[@]}" "${BIND_MOUNTS[@]}")
    fi

    # Mount the cloud storage bucket using gcsfuse if provided
    if [ -n "$BUCKET" ]; then
      echo "=== Mounting Cloud Storage bucket ${BUCKET} ==="
      # Create mount point and mount bucket with gcsfuse
      mkdir -p $HOME/bucket_mnt
      gcsfuse -o allow_other --implicit-dirs $BUCKET $HOME/bucket_mnt
      # Nextflow mounts bucket to /mnt/disks so we need to copy that for
      # symlinks to work properly
      BIND_CWD="${BIND_CWD} -v ${HOME}/bucket_mnt:/mnt/disks/${BUCKET}"
    fi

    echo "=== Launching Docker container from ${IMAGE_NAME} ==="
    # Different handling based on image type
    if (( $IS_RUNTIME_IMAGE )); then
      echo "Runtime image detected. Will run 'uv sync --frozen' before launching shell."
      docker container run -it ${BIND_CWD} ${IMAGE_NAME} bash -c "uv sync --frozen && exec bash"
    else
      echo "Code image detected. Launching shell directly."
      docker container run -it ${BIND_CWD} ${IMAGE_NAME} bash
    fi
fi
