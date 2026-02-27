#!/usr/bin/env bash
set -euo pipefail

KUSTOMIZE_DIR="${KUSTOMIZE_DIR:-deploy/k8s/aws}"
TMP_RENDERED="$(mktemp -t deposition-k8s-XXXXXX.yaml)"
trap 'rm -f "${TMP_RENDERED}"' EXIT

kubectl kustomize "${KUSTOMIZE_DIR}" > "${TMP_RENDERED}"
python - <<'PY' "${TMP_RENDERED}"
from __future__ import annotations

from pathlib import Path
import sys

import yaml

rendered = Path(sys.argv[1]).read_text(encoding="utf-8")
docs = [doc for doc in yaml.safe_load_all(rendered) if isinstance(doc, dict)]
if not docs:
    raise SystemExit("Rendered manifest is empty")

required = {
    ("Namespace", "deposition-demo"),
    ("ConfigMap", "api-config"),
    ("StatefulSet", "couchdb"),
    ("Deployment", "api"),
    ("Deployment", "ollama"),
    ("Ingress", "api"),
}
existing = {(doc.get("kind"), (doc.get("metadata") or {}).get("name")) for doc in docs}
missing = sorted(required - existing)
if missing:
    raise SystemExit(f"Rendered manifest missing required resources: {missing}")
PY

bash -n scripts/aws/create_eks_cluster.sh
bash -n scripts/aws/build_and_push.sh
bash -n scripts/aws/deploy_k8s.sh

echo "Kubernetes bundle validation passed"
