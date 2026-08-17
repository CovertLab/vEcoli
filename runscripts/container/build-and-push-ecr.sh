#!/bin/bash
# Build a Docker image and push it to AWS ECR.
#
# ASSUMES:
#   - Running on an EC2 instance with Amazon Linux 2023
#   - IAM role or credentials configured with ECR permissions
#   - The current working dir is the vEcoli/ project root
#
# Usage: runscripts/container/build-and-push-ecr.sh [-i IMAGE_TAG] [-r REPO_NAME] [-R REGION]

set -eu

# Keep track of all temporary files
TEMP_FILES=()

# Cleanup function
cleanup() {
  [ -f source-info/git_diff.txt ] && rm -f source-info/git_diff.txt
  for temp_file in "${TEMP_FILES[@]}"; do
    [ -f "$temp_file" ] && rm -f "$temp_file"
  done
  # Only print message if not in URI-only mode
  if [ "${URI_ONLY:-0}" -eq 0 ]; then
    echo "Cleaned up temporary files"
  fi
}

trap cleanup EXIT INT TERM

# Default values
IMAGE_TAG="${USER}-image"
REPO_NAME="vecoli"
AWS_REGION="${AWS_DEFAULT_REGION:-us-gov-west-1}"
URI_ONLY=0

usage_str="Usage: build-and-push-ecr.sh [-i IMAGE_TAG] [-r REPO_NAME] [-R REGION] [-u]
  -i: Docker image tag; defaults to \"${IMAGE_TAG}\".
  -r: ECR repository name; defaults to \"${REPO_NAME}\".
  -R: AWS region; defaults to \"${AWS_REGION}\" (or AWS_DEFAULT_REGION env var).
  -u: URI only mode - just output the full ECR image URI without building."

print_usage() {
  echo "$usage_str"
}

while getopts 'i:r:R:uh' flag; do
  case "${flag}" in
    i) IMAGE_TAG="${OPTARG}" ;;
    r) REPO_NAME="${OPTARG}" ;;
    R) AWS_REGION="${OPTARG}" ;;
    u) URI_ONLY=1 ;;
    h)
      print_usage
      exit 0
      ;;
    *)
      print_usage
      exit 1
      ;;
  esac
done

# Ensure required tools are installed
check_dependencies() {
  local missing_deps=()

  # Docker only needed if building
  if [ "$URI_ONLY" -eq 0 ] && ! command -v docker &> /dev/null; then
    missing_deps+=("docker")
  fi

  if ! command -v aws &> /dev/null; then
    missing_deps+=("aws-cli")
  fi

  # Git only needed if building
  if [ "$URI_ONLY" -eq 0 ] && ! command -v git &> /dev/null; then
    missing_deps+=("git")
  fi

  if [ ${#missing_deps[@]} -ne 0 ]; then
    echo "Error: Missing required dependencies: ${missing_deps[*]}"
    echo ""
    echo "To install on Amazon Linux 2023:"
    echo "  sudo dnf install -y docker git"
    echo "  sudo systemctl start docker"
    echo "  sudo usermod -aG docker \$USER"
    echo "  # Log out and back in for group changes to take effect"
    echo ""
    echo "AWS CLI v2 should be pre-installed on AL2023 EC2 instances."
    exit 1
  fi

  # Check if Docker daemon is running (only if building)
  if [ "$URI_ONLY" -eq 0 ] && ! docker info &> /dev/null; then
    echo "Error: Docker daemon is not running or you don't have permission."
    echo ""
    echo "Try:"
    echo "  sudo systemctl start docker"
    echo "  sudo usermod -aG docker \$USER"
    echo "  # Then log out and back in"
    exit 1
  fi
}

check_dependencies

# Get or create ECR repository and extract the repository URI directly
if [ "$URI_ONLY" -eq 0 ]; then
  echo "=== Ensuring ECR repository exists ==="
fi
REPO_INFO=$(aws ecr describe-repositories --repository-names "${REPO_NAME}" --region "${AWS_REGION}" 2>/dev/null || true)

if [ -z "$REPO_INFO" ]; then
  if [ "$URI_ONLY" -ne 0 ]; then
    echo "Error: ECR repository '${REPO_NAME}' does not exist." >&2
    echo "Create it first with build_image=true or manually create the repository." >&2
    exit 1
  fi
  echo "Creating ECR repository: ${REPO_NAME}"
  REPO_INFO=$(aws ecr create-repository \
    --repository-name "${REPO_NAME}" \
    --region "${AWS_REGION}" \
    --image-scanning-configuration scanOnPush=true \
    --encryption-configuration encryptionType=AES256)
  echo "Repository created successfully"
elif [ "$URI_ONLY" -eq 0 ]; then
  echo "Repository ${REPO_NAME} already exists"
fi

# Extract repository URI directly from ECR response (includes account ID)
REPO_URI=$(echo "$REPO_INFO" | grep -o '"repositoryUri": "[^"]*"' | cut -d'"' -f4)
if [ -z "$REPO_URI" ]; then
  echo "Error: Could not determine repository URI. Check your AWS credentials and permissions."
  exit 1
fi

# Extract registry from repository URI (everything before the repo name)
ECR_REGISTRY="${REPO_URI%/*}"
FULL_IMAGE_URI="${REPO_URI}:${IMAGE_TAG}"

# If URI-only mode, just output the URI and exit
if [ "$URI_ONLY" -ne 0 ]; then
  echo "${FULL_IMAGE_URI}"
  exit 0
fi

echo ""
echo "=== AWS ECR Build Configuration ==="
echo "AWS Region: ${AWS_REGION}"
echo "ECR Registry: ${ECR_REGISTRY}"
echo "Repository URI: ${REPO_URI}"
echo "Image Tag: ${IMAGE_TAG}"
echo "Full Image URI: ${FULL_IMAGE_URI}"
echo ""

# Authenticate Docker with ECR
echo "=== Authenticating with ECR ==="
aws ecr get-login-password --region "${AWS_REGION}" | \
  docker login --username AWS --password-stdin "${ECR_REGISTRY}"

# Get git metadata
GIT_HASH=$(git rev-parse HEAD)
GIT_BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null || echo "detached")
TIMESTAMP=$(date '+%Y%m%d.%H%M%S')

# Save git diff for reproducibility
mkdir -p source-info
git diff HEAD > source-info/git_diff.txt

echo ""
echo "=== Building Docker Image ==="
echo "Git Hash: ${GIT_HASH}"
echo "Git Branch: ${GIT_BRANCH}"
echo "Timestamp: ${TIMESTAMP}"
echo ""

# Build the image
docker build \
  -f runscripts/container/Dockerfile \
  -t "${FULL_IMAGE_URI}" \
  --build-arg git_hash="${GIT_HASH}" \
  --build-arg git_branch="${GIT_BRANCH}" \
  --build-arg timestamp="${TIMESTAMP}" \
  .

# Also tag with latest for convenience
docker tag "${FULL_IMAGE_URI}" "${REPO_URI}:latest"

echo ""
echo "=== Pushing Image to ECR ==="
docker push "${FULL_IMAGE_URI}"
docker push "${REPO_URI}:latest"

echo ""
echo "=== Build Complete ==="
echo "Image pushed successfully to:"
echo "  ${FULL_IMAGE_URI}"
echo "  ${REPO_URI}:latest"
echo ""
echo "To use this image in your Nextflow config, set:"
echo "  container_image = '${FULL_IMAGE_URI}'"
