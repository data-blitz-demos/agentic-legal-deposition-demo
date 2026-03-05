# Copyright (c) 2026 Data-Blitz Inc. All rights reserved.
# License: Proprietary. See NOTICE.md.
# Author: Paul Harvener.

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TERRAFORM_DIR = ROOT / "terraform"


def test_terraform_scaffold_files_exist() -> None:
    for path in [
        TERRAFORM_DIR / "README.md",
        TERRAFORM_DIR / "versions.tf",
        TERRAFORM_DIR / "providers.tf",
        TERRAFORM_DIR / "variables.tf",
        TERRAFORM_DIR / "locals.tf",
        TERRAFORM_DIR / "main.tf",
        TERRAFORM_DIR / "outputs.tf",
        TERRAFORM_DIR / "terraform.tfvars.example",
    ]:
        assert path.exists(), f"Missing Terraform scaffold file: {path}"


def test_terraform_main_includes_aws_infra_for_eks_gpu_and_efs() -> None:
    content = (TERRAFORM_DIR / "main.tf").read_text(encoding="utf-8")

    assert 'module "vpc"' in content
    assert 'module "eks"' in content
    assert 'eks_managed_node_groups' in content
    assert 'AL2023_x86_64_STANDARD' in content
    assert 'AL2023_x86_64_NVIDIA' in content
    assert 'aws_ecr_repository' in content
    assert 'aws_efs_file_system' in content
    assert 'aws_efs_mount_target' in content
    assert 'accelerator = "nvidia"' in content
    assert 'nvidia.com/gpu' in content


def test_terraform_outputs_cover_cluster_registry_and_storage() -> None:
    content = (TERRAFORM_DIR / "outputs.tf").read_text(encoding="utf-8")

    for required_output in [
        'output "cluster_name"',
        'output "cluster_endpoint"',
        'output "kubectl_config_command"',
        'output "api_repository_url"',
        'output "llm_runtime_repository_url"',
        'output "deposition_efs_file_system_id"',
        'output "gpu_nodegroup_name"',
    ]:
        assert required_output in content


def test_gitignore_excludes_terraform_local_state() -> None:
    content = (ROOT / ".gitignore").read_text(encoding="utf-8")

    assert ".terraform/" in content
    assert "*.tfstate" in content
    assert "*.tfstate.*" in content
    assert "terraform.tfvars" in content


def test_readme_references_terraform_aws_flow() -> None:
    content = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "Terraform infra scaffold: `terraform/`" in content
    assert "terraform init" in content
    assert "terraform plan" in content
    assert "terraform apply" in content
