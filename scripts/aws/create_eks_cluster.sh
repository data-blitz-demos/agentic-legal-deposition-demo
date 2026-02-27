#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${CONFIG_FILE:-deploy/eks/cluster.eksctl.yaml}"

if ! command -v eksctl >/dev/null 2>&1; then
  echo "eksctl is required" >&2
  exit 1
fi

eksctl create cluster -f "${CONFIG_FILE}"

echo "Cluster created from ${CONFIG_FILE}"
