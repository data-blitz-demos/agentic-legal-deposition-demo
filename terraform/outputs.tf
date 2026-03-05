# Copyright (c) 2026 Data-Blitz Inc. All rights reserved.
# License: Proprietary. See NOTICE.md.
# Author: Paul Harvener.

output "aws_region" {
  description = "AWS region used by this Terraform stack."
  value       = var.aws_region
}

output "cluster_name" {
  description = "EKS cluster name."
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "EKS API server endpoint."
  value       = module.eks.cluster_endpoint
}

output "kubectl_config_command" {
  description = "Command to merge this cluster into your local kubeconfig."
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}"
}

output "api_repository_url" {
  description = "ECR repository URL for the FastAPI image."
  value       = aws_ecr_repository.api.repository_url
}

output "llm_runtime_repository_url" {
  description = "ECR repository URL for the LLM runtime image."
  value       = aws_ecr_repository.llm_runtime.repository_url
}

output "deposition_efs_file_system_id" {
  description = "EFS file system ID used for deposition files."
  value       = aws_efs_file_system.depositions.id
}

output "deposition_efs_security_group_id" {
  description = "Security group attached to the deposition EFS file system."
  value       = aws_security_group.deposition_efs.id
}

output "gpu_nodegroup_name" {
  description = "Managed node group name for GPU-backed LLM inference."
  value       = module.eks.eks_managed_node_groups["gpu"].node_group_name
}

output "next_steps" {
  description = "Operational next steps after apply."
  value = {
    update_kubeconfig = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}"
    deploy_manifests  = "kubectl apply -k deploy/k8s/aws"
    patch_efs_id      = "Set deploy/k8s/aws/efs-storageclass.yaml fileSystemId to ${aws_efs_file_system.depositions.id}"
  }
}
