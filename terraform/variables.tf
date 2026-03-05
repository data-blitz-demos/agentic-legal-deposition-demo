# Copyright (c) 2026 Data-Blitz Inc. All rights reserved.
# License: Proprietary. See NOTICE.md.
# Author: Paul Harvener.

variable "app_name" {
  description = "Base name used for AWS resources."
  type        = string
  default     = "deposition-demo"
}

variable "environment" {
  description = "Environment name used for tagging and naming."
  type        = string
  default     = "dev"
}

variable "aws_region" {
  description = "AWS region for all infrastructure resources."
  type        = string
  default     = "us-east-1"
}

variable "cluster_version" {
  description = "Kubernetes version for the EKS cluster."
  type        = string
  default     = "1.30"
}

variable "vpc_cidr" {
  description = "Primary VPC CIDR block."
  type        = string
  default     = "10.42.0.0/16"
}

variable "availability_zones" {
  description = "Optional explicit list of availability zones. Leave empty to use the first three available zones."
  type        = list(string)
  default     = []
}

variable "single_nat_gateway" {
  description = "Use a single NAT gateway to reduce cost in lower environments."
  type        = bool
  default     = true
}

variable "cpu_node_instance_type" {
  description = "EC2 instance type for the primary CPU node group."
  type        = string
  default     = "m6i.large"
}

variable "cpu_node_desired_size" {
  description = "Desired size for the CPU node group."
  type        = number
  default     = 2
}

variable "cpu_node_min_size" {
  description = "Minimum size for the CPU node group."
  type        = number
  default     = 2
}

variable "cpu_node_max_size" {
  description = "Maximum size for the CPU node group."
  type        = number
  default     = 6
}

variable "cpu_capacity_type" {
  description = "Capacity type for CPU nodes (ON_DEMAND or SPOT)."
  type        = string
  default     = "ON_DEMAND"
}

variable "gpu_node_instance_type" {
  description = "EC2 instance type for the GPU node group that runs LLM inference."
  type        = string
  default     = "g5.2xlarge"
}

variable "gpu_node_desired_size" {
  description = "Desired size for the GPU node group."
  type        = number
  default     = 1
}

variable "gpu_node_min_size" {
  description = "Minimum size for the GPU node group."
  type        = number
  default     = 1
}

variable "gpu_node_max_size" {
  description = "Maximum size for the GPU node group."
  type        = number
  default     = 4
}

variable "gpu_capacity_type" {
  description = "Capacity type for GPU nodes (ON_DEMAND or SPOT)."
  type        = string
  default     = "ON_DEMAND"
}

variable "deposition_efs_performance_mode" {
  description = "EFS performance mode for deposition files."
  type        = string
  default     = "generalPurpose"
}

variable "deposition_efs_throughput_mode" {
  description = "EFS throughput mode for deposition files."
  type        = string
  default     = "bursting"
}
