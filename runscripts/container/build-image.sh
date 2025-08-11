#!/bin/bash
# Use Google Cloud Build or local Docker to build a personalized image with
# current state of the vEcoli repo. If using Cloud Build, store the built
# image in the "vecoli" repository in Artifact Registry.
#
# ASSUMES: The current working dir is the vEcoli/ project root.

set -eu

# Keep track of all temporary files
TEMP_FILES=()

# Cleanup function to handle all temporary files
cleanup() {
  # Remove git diff file if it exists
  [ -f source-info/git_diff.txt ] && rm -f source-info/git_diff.txt

  # Clean up all temporary files
  for temp_file in "${TEMP_FILES[@]}"; do
    [ -f "$temp_file" ] && rm -f "$temp_file"
  done

  echo "Cleaned up temporary files"
}

# Register cleanup on exit, interrupt, and error
trap cleanup EXIT INT TERM

IMAGE="${USER}-image"
RUN_LOCAL=0
BUILD_APPTAINER=0

usage_str="Usage: build-image.sh [-i IMAGE] [-l] [-a]
  -i: Docker tag of image to build; defaults to \"$IMAGE\".
  -a: Build Apptainer image (cannot use with -l).
  -l: Build Docker image locally (defaults to Cloud Build)."

print_usage() {
  echo "$usage_str"
}

while getopts 'i:la' flag; do
  case "${flag}" in
  a)
    if [ "$RUN_LOCAL" -ne 0 ]; then
      print_usage
      exit 1
    else
      BUILD_APPTAINER=1
    fi
    ;;
  i) IMAGE="${OPTARG}" ;;
  l)
    if [ "$BUILD_APPTAINER" -ne 0 ]; then
      print_usage
      exit 1
    else
      RUN_LOCAL=1
    fi
    ;;
  *)
    print_usage
    exit 1
    ;;
  esac
done

GIT_HASH=$(git rev-parse HEAD)
GIT_BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null || echo "detached")
TIMESTAMP=$(date '+%Y%m%d.%H%M%S')
mkdir -p source-info
git diff HEAD >source-info/git_diff.txt

if [ "$RUN_LOCAL" -ne 0 ]; then
  echo "=== Locally building Docker Image ${IMAGE} ==="
  echo "=== git hash ${GIT_HASH}, git branch ${GIT_BRANCH} ==="
  docker build -f runscripts/container/Dockerfile -t "${IMAGE}" \
    --build-arg git_hash="${GIT_HASH}" \
    --build-arg git_branch="${GIT_BRANCH}" \
    --build-arg timestamp="${TIMESTAMP}" .
elif [ "$BUILD_APPTAINER" -ne 0 ]; then
  # Create a temporary Singularity definition file
  TEMP_DEF=$(mktemp)
  TEMP_FILES+=("$TEMP_DEF")

  # Create a temporary file for find exclude patterns
  EXCLUDE_PATTERNS=$(mktemp)
  TEMP_FILES+=("$EXCLUDE_PATTERNS")

  # Function to process ignore files
  process_ignore_file() {
    local ignore_file="$1"

    if [ -f "$ignore_file" ]; then
      echo "Processing patterns from $ignore_file"
      grep -v "^#" "$ignore_file" | grep -v "^$" | grep -v "^!" | while read -r pattern; do
        # Handle patterns starting with / (root-relative)
        if [[ "$pattern" == /* ]]; then
          echo ".${pattern}" >>"$EXCLUDE_PATTERNS"
          echo ".${pattern}/*" >>"$EXCLUDE_PATTERNS"
        # Handle directory patterns ending with /
        elif [[ "$pattern" == */ ]]; then
          echo "./${pattern}*" >>"$EXCLUDE_PATTERNS"
          echo "./*/${pattern}*" >>"$EXCLUDE_PATTERNS"
        # Handle other patterns
        else
          echo "./*/${pattern}" >>"$EXCLUDE_PATTERNS"
          echo "./${pattern}" >>"$EXCLUDE_PATTERNS"
          echo "./${pattern}/*" >>"$EXCLUDE_PATTERNS"
          echo "./*/${pattern}/*" >>"$EXCLUDE_PATTERNS"
        fi
      done
    fi
  }

  process_ignore_file .dockerignore

  # Start building the find command
  FIND_CMD="find . -type f"

  # Add exclusion patterns to the find command
  while read -r pattern; do
    FIND_CMD="$FIND_CMD ! -path \"$pattern\""
  done <"$EXCLUDE_PATTERNS"

  # Create a temporary file for our list of files
  TEMP_FILES_LIST=$(mktemp)
  TEMP_FILES+=("$TEMP_FILES_LIST")

  echo "Executing: $FIND_CMD"
  # Execute the dynamically generated find command
  eval "$FIND_CMD" >"$TEMP_FILES_LIST"

  # Debug output
  echo "Generated $(wc -l <"$TEMP_FILES_LIST") files to include in the image"

  # Initialize environment variables string
  DOT_ENV_VARS=""
  # Check if .env file exists
  if [ -f ".env" ]; then
      echo "Processing .env for Singularity environment..."
      # Read .env file line by line
      while IFS= read -r line || [ -n "$line" ]; do
          # Skip empty lines and comments
          if [[ -n "$line" && ! "$line" =~ ^\s*# ]]; then
              # Strip any existing 'export ' prefix
              line=${line#export }
              # Add to environment variables string with export prefix
              DOT_ENV_VARS+="    export $line"$'\n'
          fi
      done < ".env"
      echo "Found $(echo "$DOT_ENV_VARS" | grep -c 'export ') environment variables"
  else
      echo "Warning: .env not found"
  fi

  # Read the Singularity file line by line
  while IFS= read -r line; do
    if [[ "$line" == *"FILES_TO_ADD"* ]]; then
      # For the line containing FILES_TO_ADD, replace with formatted file paths
      while IFS= read -r file; do
        echo "    $file /vEcoli/$file" >>"$TEMP_DEF"
      done <"$TEMP_FILES_LIST"
    elif [[ "$line" == *"DOT_ENV_VARS"* ]]; then
      echo "$DOT_ENV_VARS" >> "$TEMP_DEF"
    else
      # Otherwise just add the line as-is
      echo "$line" >>"$TEMP_DEF"
    fi
  done <runscripts/container/Singularity

  echo "Using temporary definition file: $TEMP_DEF"
  echo "=== Building Apptainer Image: ${IMAGE} ==="
  echo "=== git hash ${GIT_HASH}, git branch ${GIT_BRANCH} ==="

  # Retry 10 times to handle inconsistent failures on Sherlock
  MAX_ATTEMPTS=10
  ATTEMPT=1
  until apptainer build --force \
    --build-arg git_hash="${GIT_HASH}" \
    --build-arg git_branch="${GIT_BRANCH}" \
    --build-arg timestamp="${TIMESTAMP}" \
    "${IMAGE}" "$TEMP_DEF"; do
    if [ $ATTEMPT -ge $MAX_ATTEMPTS ]; then
        echo "ERROR: Apptainer build failed after $MAX_ATTEMPTS attempts."
        exit 1
    fi
    echo "Apptainer build attempt $ATTEMPT failed."
    ATTEMPT=$((ATTEMPT + 1))
  done
  echo "Apptainer build successful after $ATTEMPT attempt(s)!"

else
  echo "=== Cloud-building Docker Image ${IMAGE} ==="
  echo "=== git hash ${GIT_HASH}, git branch ${GIT_BRANCH} ==="
  # For this script to work on a Compute Engine VM, you must
  # - Set default Compute Engine region and zone for your project
  # - Set access scope to "Allow full access to all Cloud APIs" when
  #   creating VM
  # - Run gcloud init in VM
  REGION=$(gcloud config get compute/region)
  # This needs a config file to identify the project files to upload and the
  # Dockerfile to run.
  gcloud builds submit --timeout=3h --region="$REGION" \
    --config runscripts/container/cloud_build.json \
    --ignore-file=.dockerignore \
    --substitutions="_IMAGE=${IMAGE},_GIT_HASH=${GIT_HASH},_GIT_BRANCH=${GIT_BRANCH},\
_TIMESTAMP=${TIMESTAMP}"
fi
