#!/bin/bash -e

# Environment check
required_vars=(
  "COVERITY_SCAN_PROJECT_NAME"
  "COVERITY_SCAN_NOTIFICATION_EMAIL"
  "COVERITY_SCAN_BUILD_COMMAND"
  "COVERITY_SCAN_TOKEN"
  "COVERITY_SCAN_BRANCH_PATTERN"
)

for var in "${required_vars[@]}"; do
  if [ -z "${!var}" ]; then
    echo "ERROR: $var must be set"
    exit 1
  fi
done

PLATFORM="linux64"
TOOL_ARCHIVE=/tmp/cov-analysis-${PLATFORM}.tgz
TOOL_URL=https://scan.coverity.com/download/cxx/${PLATFORM}
TOOL_BASE=/tmp/coverity-scan-analysis
COVERITY_UPLOAD_URL="https://scan.coverity.com/builds"
COVERITY_SCAN_URL="https://scan.coverity.com"
COVERITY_RESULTS_DIR="cov-int"
COVERITY_RESULTS_ARCHIVE="analysis-results.tgz"

# Check if the current branch matches the Coverity scan branch pattern
if [[ "$GITHUB_REF_NAME" =~ ^$COVERITY_SCAN_BRANCH_PATTERN$ ]]; then
  echo -e "Coverity Scan configured to run on branch ${GITHUB_REF_NAME}"
else
  echo -e "Coverity Scan is NOT configured to run on branch ${GITHUB_REF_NAME}"
  exit 1
fi

# Verify upload is permitted. See https://scan.coverity.com/faq#frequency
AUTH_RES=$(curl -s --form project="$COVERITY_SCAN_PROJECT_NAME" --form token="$COVERITY_SCAN_TOKEN" "$COVERITY_SCAN_URL/api/upload_permitted")
if [ "$AUTH_RES" = "Access denied" ]; then
  echo "Coverity Scan API access denied. Check COVERITY_SCAN_PROJECT_NAME and COVERITY_SCAN_TOKEN."
  exit 1
fi

AUTH=$(echo "$AUTH_RES" | jq -r '.upload_permitted')
if [ "$AUTH" = "true" ]; then
  echo "Coverity Scan analysis authorized per quota."
else
  WHEN=$(echo "$AUTH_RES" | jq -r '.next_upload_permitted_at')
  echo "Coverity Scan analysis NOT authorized until $WHEN."
  exit 0
fi

# Download and set up Coverity tool if not already done
if [ ! -d "$TOOL_BASE" ]; then
  echo "Downloading Coverity Scan Analysis Tool..."
  wget -q -nv -O "$TOOL_ARCHIVE" "$TOOL_URL" --post-data "project=$COVERITY_SCAN_PROJECT_NAME&token=$COVERITY_SCAN_TOKEN"
  
  echo "Extracting Coverity Scan Analysis Tool..."
  mkdir -p "$TOOL_BASE"
  tar xzf "$TOOL_ARCHIVE" -C "$TOOL_BASE"
fi

echo "Adding Coverity to PATH..."
TOOL_DIR=$(find "$TOOL_BASE" -type d -name 'cov-analysis*')
export PATH="$TOOL_DIR/bin:$PATH"

# Clean up the build directory
cd "$GITHUB_WORKSPACE"
cmake --build build --target clean

# Build
echo "Capturing ISPC build by Coverity Scan and importing source code information..."
cov-build --dir $COVERITY_RESULTS_DIR $COVERITY_SCAN_BUILD_COMMAND
cov-import-scm --dir $COVERITY_RESULTS_DIR --scm git --log $COVERITY_RESULTS_DIR/scm_log.txt 2>&1

# Upload results
echo "Tarring Coverity Scan results..."
tar czf "$COVERITY_RESULTS_ARCHIVE" "$COVERITY_RESULTS_DIR"
SHA=$(git rev-parse --short HEAD)

echo "Uploading Coverity Scan results..."
response=$(curl \
  --silent --write-out "\n%{http_code}\n" \
  --form project="$COVERITY_SCAN_PROJECT_NAME" \
  --form token="$COVERITY_SCAN_TOKEN" \
  --form email="$COVERITY_SCAN_NOTIFICATION_EMAIL" \
  --form file=@"$COVERITY_RESULTS_ARCHIVE" \
  --form version="$SHA" \
  --form description="Github Action build" \
  "$COVERITY_UPLOAD_URL")

status_code=$(echo "$response" | tail -n 1)
if [ "$status_code" != "200" ]; then
  echo "Coverity Scan upload failed: $response."
  exit 1
fi

echo "Coverity Scan completed successfully."
