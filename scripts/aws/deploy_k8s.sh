#!/usr/bin/env bash
set -euo pipefail

: "${API_IMAGE:?Set API_IMAGE to the full ECR image URI}"
KUSTOMIZE_DIR="${KUSTOMIZE_DIR:-deploy/k8s/aws}"
NAMESPACE="${NAMESPACE:-deposition-demo}"

kubectl apply -k "${KUSTOMIZE_DIR}"
kubectl -n "${NAMESPACE}" set image deployment/api api="${API_IMAGE}"

kubectl -n "${NAMESPACE}" rollout status statefulset/couchdb --timeout=10m
kubectl -n "${NAMESPACE}" rollout status deployment/ollama --timeout=20m
kubectl -n "${NAMESPACE}" rollout status deployment/api --timeout=10m

kubectl -n "${NAMESPACE}" delete job ollama-model-loader --ignore-not-found
kubectl -n "${NAMESPACE}" apply -f "${KUSTOMIZE_DIR}/ollama-model-loader-job.yaml"
kubectl -n "${NAMESPACE}" wait --for=condition=complete --timeout=45m job/ollama-model-loader

echo "Deployment complete in namespace ${NAMESPACE}"
