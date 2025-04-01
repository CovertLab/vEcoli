#!/bin/sh
# Use Google Cloud Build or local Docker to build a personalized image with
# current state of the vEcoli repo. If using Cloud Build, store the built
# image in the "vecoli" repository in Artifact Registry.
#
# ASSUMES: The current working dir is the vEcoli/ project root.

set -eu
trap 'rm -f source-info/git_diff.txt' EXIT

RUNTIME_IMAGE="${USER}-runtime"
CODE_IMAGE="${USER}-code"
RUN_LOCAL=0
BUILD_APPTAINER=0

usage_str="Usage: build-code-image.sh [-r RUNTIME_IMAGE] [-w CODE_IMAGE] [-l] [-a]
  -r: Docker tag of runtime image to build from; defaults to \"$USER-runtime\" 
      (must exist in Artifact Registry).
  -a: Build Apptainer image (cannot use with -l, -r should be path to runtime 
      image to build from).
  -w: Docker tag of full code image to build; defaults to \"$USER-code\".
  -l: Build image locally."

print_usage() {
  echo "$usage_str"
}

while getopts 'r:w:la' flag; do
  case "${flag}" in
    r) RUNTIME_IMAGE="${OPTARG}" ;;
    a)
      if [ "$RUN_LOCAL" -ne 0 ]; then
        print_usage
        exit 1
      else
        BUILD_APPTAINER=1
      fi
      ;;
    w) CODE_IMAGE="${OPTARG}" ;;
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
GIT_BRANCH=$(git symbolic-ref --short HEAD)
TIMESTAMP=$(date '+%Y%m%d.%H%M%S')
mkdir -p source-info
git diff HEAD > source-info/git_diff.txt

if [ "$RUN_LOCAL" -ne 0 ]; then
    echo "=== Locally building code Docker Image ${CODE_IMAGE} on ${RUNTIME_IMAGE} ==="
    echo "=== git hash ${GIT_HASH}, git branch ${GIT_BRANCH} ==="
    docker build -f runscripts/container/wholecell/Dockerfile -t "${CODE_IMAGE}" \
        --build-arg from="${RUNTIME_IMAGE}" \
        --build-arg git_hash="${GIT_HASH}" \
        --build-arg git_branch="${GIT_BRANCH}" \
        --build-arg timestamp="${TIMESTAMP}" .
elif [ "$BUILD_APPTAINER" -ne 0 ]; then
    # Create a temporary Singularity definition file
    TEMP_DEF=$(mktemp)
    
    # Create a temporary file for find exclude patterns
    EXCLUDE_PATTERNS=$(mktemp)
    
    # Function to process ignore files
    process_ignore_file() {
        local ignore_file="$1"
        
        if [ -f "$ignore_file" ]; then
            echo "Processing patterns from $ignore_file"
            grep -v "^#" "$ignore_file" | grep -v "^$" | grep -v "^!" | while read -r pattern; do
                # Handle patterns starting with / (root-relative)
                if [[ "$pattern" == /* ]]; then
                    echo ".${pattern}" >> "$EXCLUDE_PATTERNS"
                    echo ".${pattern}/*" >> "$EXCLUDE_PATTERNS"
                # Handle directory patterns ending with /
                elif [[ "$pattern" == */ ]]; then
                    echo "./${pattern}*" >> "$EXCLUDE_PATTERNS"
                    echo "./*/${pattern}*" >> "$EXCLUDE_PATTERNS"
                # Handle other patterns
                else
                    echo "./*/${pattern}" >> "$EXCLUDE_PATTERNS"
                    echo "./${pattern}" >> "$EXCLUDE_PATTERNS"
                    echo "./${pattern}/*" >> "$EXCLUDE_PATTERNS"
                    echo "./*/${pattern}/*" >> "$EXCLUDE_PATTERNS"
                fi
            done
        fi
    }
    
    # Process both ignore files
    for ignore_file in .gitignore .gcloudignore; do
        process_ignore_file "$ignore_file"
    done
    
    # Create a temporary file for the find command
    FIND_CMD=$(mktemp)
    
    # Start building the find command
    FIND_CMD="find . -type f"
    
    # Add exclusion patterns to the find command
    while read -r pattern; do
        FIND_CMD="$FIND_CMD ! -path \"$pattern\""
    done < "$EXCLUDE_PATTERNS"
    
    # Create a temporary file for our list of files
    TEMP_FILES=$(mktemp)
    echo $FIND_CMD
    # Execute the dynamically generated find command
    eval "$FIND_CMD" > "$TEMP_FILES"
    
    # Debug output
    echo "Generated $(wc -l < "$TEMP_FILES") files to include in the image"
    
    # Read the Singularity file line by line
    while IFS= read -r line; do
        if [[ "$line" == *"FILES_TO_ADD"* ]]; then
            # For the line containing FILES_TO_ADD, replace with formatted file paths
            while IFS= read -r file; do
                echo "    $file /vEcoli/$file" >> "$TEMP_DEF"
            done < "$TEMP_FILES"
        else
            # Otherwise just add the line as-is
            echo "$line" >> "$TEMP_DEF"
        fi
    done < runscripts/container/wholecell/Singularity
    
    # Clean up
    rm -f "$TEMP_FILES" "$EXCLUDE_PATTERNS"
    
    echo "Using temporary definition file: $TEMP_DEF"
    echo "=== Building code Apptainer Image: ${CODE_IMAGE} on ${RUNTIME_IMAGE} ==="
    echo "=== git hash ${GIT_HASH}, git branch ${GIT_BRANCH} ==="
    
    apptainer build --force \
        --build-arg from="${RUNTIME_IMAGE}" \
        --build-arg git_hash="${GIT_HASH}" \
        --build-arg git_branch="${GIT_BRANCH}" \
        --build-arg timestamp="${TIMESTAMP}" \
        "${CODE_IMAGE}" "$TEMP_DEF"
    
    # Clean up the temporary file
    rm -f "$TEMP_DEF"
else
    echo "=== Cloud-building code Docker Image ${CODE_IMAGE} on ${RUNTIME_IMAGE} ==="
    echo "=== git hash ${GIT_HASH}, git branch ${GIT_BRANCH} ==="
    # For this script to work on a Compute Engine VM, you must
    # - Set default Compute Engine region and zone for your project
    # - Set access scope to "Allow full access to all Cloud APIs" when
    #   creating VM
    # - Run gcloud init in VM
    REGION=$(gcloud config get compute/region)
    # This needs a config file to identify the project files to upload and the
    # Dockerfile to run.
    gcloud builds submit --timeout=15m --region="$REGION" \
      --config runscripts/container/cloud_build.json \
      --substitutions="_RUNTIME_IMAGE=${RUNTIME_IMAGE},\
_CODE_IMAGE=${CODE_IMAGE},_GIT_HASH=${GIT_HASH},_GIT_BRANCH=${GIT_BRANCH},\
_TIMESTAMP=${TIMESTAMP}"
fi

rm source-info/git_diff.txt
