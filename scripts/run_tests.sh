#!/usr/bin/env bash
# Copyright (c) 2026 Data-Blitz Inc. All rights reserved.
# License: Proprietary. See NOTICE.md.
# Author: Paul Harvener.

set -euo pipefail

REPORT_DIR="reports"
JUNIT_XML="${REPORT_DIR}/junit.xml"
HTML_REPORT="${REPORT_DIR}/tests.html"

mkdir -p "${REPORT_DIR}"

python -m pytest \
  --cov=backend/app \
  --cov-report=term-missing \
  --cov-fail-under=95 \
  --junitxml="${JUNIT_XML}" \
  --html="${HTML_REPORT}" \
  --self-contained-html

echo "JUnit report: ${JUNIT_XML}"
echo "HTML report: ${HTML_REPORT}"
