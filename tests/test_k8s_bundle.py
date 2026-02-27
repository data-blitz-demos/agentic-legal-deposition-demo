from __future__ import annotations

import subprocess
from pathlib import Path
import shutil

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
KUSTOMIZE_DIR = ROOT / "deploy" / "k8s" / "aws"


def _render_manifests() -> list[dict]:
    if shutil.which("kubectl") is None:
        pytest.skip("kubectl is required for Kubernetes manifest rendering tests")
    rendered = subprocess.run(
        ["kubectl", "kustomize", str(KUSTOMIZE_DIR)],
        check=True,
        capture_output=True,
        text=True,
    )
    docs = [doc for doc in yaml.safe_load_all(rendered.stdout) if isinstance(doc, dict)]
    assert docs, "kustomize output was empty"
    return docs


def _find_doc(docs: list[dict], kind: str, name: str) -> dict:
    for doc in docs:
        if doc.get("kind") != kind:
            continue
        metadata = doc.get("metadata") or {}
        if metadata.get("name") == name:
            return doc
    raise AssertionError(f"Missing manifest kind={kind} name={name}")


def test_kustomize_renders_expected_resource_set() -> None:
    docs = _render_manifests()

    kinds = {doc.get("kind") for doc in docs}
    for required_kind in {
        "Namespace",
        "ConfigMap",
        "Secret",
        "PersistentVolumeClaim",
        "StatefulSet",
        "Deployment",
        "Service",
        "Ingress",
        "Job",
    }:
        assert required_kind in kinds


def test_api_deployment_wiring_is_correct() -> None:
    docs = _render_manifests()
    deployment = _find_doc(docs, "Deployment", "api")

    containers = deployment["spec"]["template"]["spec"]["containers"]
    api_container = next(item for item in containers if item.get("name") == "api")

    env_by_name = {item["name"]: item for item in api_container.get("env", [])}
    assert env_by_name["COUCHDB_URL"]["value"] == "http://$(COUCHDB_USER):$(COUCHDB_PASSWORD)@couchdb:5984"

    readiness = api_container["readinessProbe"]["httpGet"]
    assert readiness["path"] == "/api/thought-streams/health"

    mounts = {item["name"]: item for item in api_container.get("volumeMounts", [])}
    assert mounts["deposition-files"]["mountPath"] == "/data/depositions"


def test_ollama_deployment_requests_gpu() -> None:
    docs = _render_manifests()
    deployment = _find_doc(docs, "Deployment", "ollama")

    pod_spec = deployment["spec"]["template"]["spec"]
    assert pod_spec["nodeSelector"]["accelerator"] == "nvidia"

    containers = pod_spec["containers"]
    ollama_container = next(item for item in containers if item.get("name") == "ollama")
    requests = ollama_container["resources"]["requests"]
    limits = ollama_container["resources"]["limits"]

    assert requests["nvidia.com/gpu"] == "1"
    assert limits["nvidia.com/gpu"] == "1"


def test_storage_and_ingress_manifests_have_aws_settings() -> None:
    docs = _render_manifests()

    storage_class = _find_doc(docs, "StorageClass", "efs-sc")
    assert storage_class["provisioner"] == "efs.csi.aws.com"
    assert storage_class["parameters"]["fileSystemId"].startswith("fs-")

    deposition_pvc = _find_doc(docs, "PersistentVolumeClaim", "deposition-files")
    assert "ReadWriteMany" in deposition_pvc["spec"]["accessModes"]
    assert deposition_pvc["spec"]["storageClassName"] == "efs-sc"

    ingress = _find_doc(docs, "Ingress", "api")
    annotations = ingress["metadata"]["annotations"]
    assert annotations["kubernetes.io/ingress.class"] == "alb"
    assert "alb.ingress.kubernetes.io/certificate-arn" in annotations


def test_shell_scripts_are_syntax_valid() -> None:
    for script in [
        ROOT / "scripts" / "aws" / "create_eks_cluster.sh",
        ROOT / "scripts" / "aws" / "build_and_push.sh",
        ROOT / "scripts" / "aws" / "deploy_k8s.sh",
        ROOT / "scripts" / "aws" / "test_k8s_bundle.sh",
    ]:
        subprocess.run(["bash", "-n", str(script)], check=True)
