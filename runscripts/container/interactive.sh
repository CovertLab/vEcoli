#!/bin/bash
# Start an interactive Docker or Apptainer container from an image
# built with runscripts/container/build-code-image.sh.
# Supports optional bind mounts and Cloud Storage bucket mounting

set -eu # Exit on any error or unset variable

# Keep track of resources to clean up
TMP_OVERLAY_DIR=""

cleanup() {
  # Unmount bucket if mounted
  if [ -d "$(pwd)/bucket_mnt" ]; then
    fusermount -u $(pwd)/bucket_mnt &>/dev/null || true
    rm -rf "$(pwd)/bucket_mnt" &>/dev/null || true
  fi

  # Remove the temporary overlay directory if it exists
  if [ -n "$TMP_OVERLAY_DIR" ] && [ -d "$TMP_OVERLAY_DIR" ]; then
    rm -rf "$TMP_OVERLAY_DIR" &>/dev/null || true
    echo "Cleaned up temporary overlay directory"
  fi

  # Remove the .venv copied from the container
  if [ -d "./.venv" ]; then
    # Use find with -delete which is faster for many files
    find ./.venv -type f -print0 | xargs -0 rm -f &>/dev/null || true
    find ./.venv -type d -print0 | xargs -0 rmdir &>/dev/null || true
    # Final cleanup in case some directories weren't empty
    rm -rf ./.venv &>/dev/null || true
  fi
}

# Ensure resources are cleaned up on exit
trap cleanup EXIT INT TERM

# Default configuration variables
IMAGE_NAME="${USER}-code-image" # Default image name for Docker/Apptainer
USE_APPTAINER=0                 # Flag: Use Apptainer if set to 1
RUN_LOCAL=0                     # Flag: Use local Docker image if set to 1
DEV_MODE=0                      # Flag: Mount current dir on host to /vEcoli in container if set to 1
BIND_MOUNTS=()                  # Array for bind mount paths
BIND_CWD=""                     # Formatted bind mount string for runtime
BUCKET=""                       # Cloud Storage bucket name
OVERLAY_SIZE=1024               # Size of sparse temporary Apptainer overlay image in MB

# Help message string
usage_str="Usage: interactive.sh [-w IMAGE_NAME] [-a] [-l] [-d] [-b] [-s OVERLAY_SIZE] [-p]...\n\
Options:\n\
    -w: Path to code image if -a or -l are passed, otherwise name of Docker \
image inside vecoli Artifact Repository; defaults to \"$IMAGE_NAME\".\n\
    -d: Mount current working directory to /vEcoli in container; \
useful for making and testing code changes that persist and can be \
checked into git history.\n\
    -s: Size of sparse temporary Apptainer overlay image in MB; \
defaults to \"$OVERLAY_SIZE\".\n
    -a: Load Apptainer image.\n\
    -l: Load local Docker image.\n\
    -b: Name of Cloud Storage bucket to mount inside container; first mounts
bucket at $(pwd)/bucket_mnt using gcsfuse (does not work with -a).\n\
    -p: Path(s) to mount inside container; can specify multiple with \
\"-p path1 -p path2\"\n"

# Function to print usage instructions
print_usage() {
  printf "$usage_str"
}

# Parse command-line options
while getopts 'w:adls:b:p:' flag; do
  case "${flag}" in
  w) IMAGE_NAME="${OPTARG}" ;; # Set custom image name
  a)
    # Make sure -a and -l are not both specified
    if [ "$RUN_LOCAL" -eq 1 ]; then
      echo "ERROR: Options -a (Apptainer) and -l (local Docker) cannot be used together."
      print_usage
      exit 1
    fi
    USE_APPTAINER=1
    ;;             # Enable Apptainer mode
  d) DEV_MODE=1 ;; # Enable development mode
  l)
    # Make sure -l and -a are not both specified
    if [ "$USE_APPTAINER" -eq 1 ]; then
      echo "ERROR: Options -l (local Docker) and -a (Apptainer) cannot be used together."
      print_usage
      exit 1
    fi
    RUN_LOCAL=1
    ;;                                         # Enable local Docker mode
  s) OVERLAY_SIZE="${OPTARG}" ;;               # Set the size of the sparse overlay
  b) BUCKET="${OPTARG}" ;;                     # Set the Cloud Storage bucket
  p) BIND_MOUNTS+=($(realpath "${OPTARG}")) ;; # Convert path to absolute and add to array
  *)
    print_usage # Print usage for unknown flags
    exit 1
    ;;
  esac
done

# Validate that bucket mounting is not used with Apptainer
if [ -n "$BUCKET" ] && [ "$USE_APPTAINER" -eq 1 ]; then
  echo "ERROR: Bucket mounting (-b) is not supported with Apptainer mode (-a)."
  print_usage
  exit 1
fi

# Apptainer-specific logic
if (($USE_APPTAINER)); then
  # If there are bind mounts, format them for Apptainer
  if [ ${#BIND_MOUNTS[@]} -ne 0 ]; then
    BIND_CWD=$(printf " -B %s" "${BIND_MOUNTS[@]}")
  fi

  echo "=== Launching Apptainer container from ${IMAGE_NAME} ==="

  # Create a temporary overlay directory
  TMP_OVERLAY_DIR=$(mktemp -d)
  echo "Creating ${OVERLAY_SIZE}MB sparse temporary overlay at ${TMP_OVERLAY_DIR}/overlay.img"
  # Create a sparse file (only allocates blocks as needed)
  dd if=/dev/zero of=${TMP_OVERLAY_DIR}/overlay.img bs=1M count=0 seek=${OVERLAY_SIZE}
  # Format the file as ext3 filesystem
  mkfs.ext3 -F ${TMP_OVERLAY_DIR}/overlay.img

  if (($DEV_MODE)); then
    # Check if .venv already exists in the current directory
    if [ -d "./.venv" ]; then
      echo "ERROR: A .venv directory already exists in the current directory."
      echo "This will conflict with the container's virtual environment."
      echo "Please back up or remove your .venv directory before proceeding:"
      echo "  mv .venv .venv.backup  # To back up"
      echo "  rm -rf .venv           # To remove"
      echo ""
      echo "Then run this script again."
      exit 1
    fi

    # Copy .venv from container to current directory
    echo "Copying .venv from container to current directory..."
    apptainer exec -B .:/host_repo ${IMAGE_NAME} \
      bash -c "cd /vEcoli && tar cf - .venv | tar xf - -C /host_repo/"

    # Now add bind mount for current directory
    BIND_CWD="${BIND_CWD} -B .:/vEcoli"
  fi
  # Fakeroot necessary for overlay to work
  apptainer exec -e --overlay ${TMP_OVERLAY_DIR}/overlay.img \
    --fakeroot ${BIND_CWD} ${IMAGE_NAME} bash
else
  # Docker-specific logic
  if ((!$RUN_LOCAL)); then
    # We're using an image from Artifact Registry, so build the full path
    PROJECT=$(gcloud config get project)
    REGION=$(gcloud config get compute/region)
    IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT}/vecoli/${IMAGE_NAME}"
  fi
  echo "=== Launching Docker container from ${IMAGE_NAME} ==="

  # If there are bind mounts, format them for Docker
  if [ ${#BIND_MOUNTS[@]} -ne 0 ]; then
    BIND_CWD=$(printf " -v %s:%s" "${BIND_MOUNTS[@]}" "${BIND_MOUNTS[@]}")
  fi

  # Mount the cloud storage bucket using gcsfuse if provided
  if [ -n "$BUCKET" ]; then
    echo "Mounting Cloud Storage bucket ${BUCKET} at $(pwd)/bucket_mnt"
    # Create mount point and mount bucket with gcsfuse
    mkdir -p $(pwd)/bucket_mnt
    gcsfuse -o allow_other --implicit-dirs $BUCKET $(pwd)/bucket_mnt
    # Nextflow mounts bucket to /mnt/disks so we need to copy that for
    # symlinks to work properly
    BIND_CWD="${BIND_CWD} -v $(pwd)/bucket_mnt:/mnt/disks/${BUCKET}"
  fi

  if (($DEV_MODE)); then
    # Bind current directory but use anonymous volume to retain
    # /vEcoli/.venv inside the container
    BIND_CWD="${BIND_CWD} --rm -v .:/vEcoli -v /vEcoli/.venv"
  fi

  docker container run -it ${BIND_CWD} ${IMAGE_NAME} bash
fi
