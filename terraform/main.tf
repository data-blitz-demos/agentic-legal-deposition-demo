# Copyright (c) 2026 Data-Blitz Inc. All rights reserved.
# License: Proprietary. See NOTICE.md.
# Author: Paul Harvener.

resource "aws_ecr_repository" "api" {
  name                 = "${local.name_prefix}-api"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "llm_runtime" {
  name                 = "${local.name_prefix}-llm-runtime"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Expire untagged images beyond the newest 10"
        selection = {
          tagStatus   = "untagged"
          countType   = "imageCountMoreThan"
          countNumber = 10
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

resource "aws_ecr_lifecycle_policy" "llm_runtime" {
  repository = aws_ecr_repository.llm_runtime.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Expire untagged images beyond the newest 10"
        selection = {
          tagStatus   = "untagged"
          countType   = "imageCountMoreThan"
          countNumber = 10
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = local.name_prefix
  cidr = var.vpc_cidr

  azs             = local.azs
  private_subnets = [for index, _az in local.azs : cidrsubnet(var.vpc_cidr, 4, index)]
  public_subnets  = [for index, _az in local.azs : cidrsubnet(var.vpc_cidr, 4, index + 8)]

  enable_nat_gateway = true
  single_nat_gateway = var.single_nat_gateway

  enable_dns_hostnames = true
  enable_dns_support   = true

  public_subnet_tags = {
    "kubernetes.io/role/elb" = 1
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = 1
  }
}

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = local.cluster_name
  cluster_version = var.cluster_version

  cluster_endpoint_public_access           = true
  enable_cluster_creator_admin_permissions = true

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  eks_managed_node_group_defaults = {
    disk_size = 100
  }

  eks_managed_node_groups = {
    system = {
      ami_type       = "AL2023_x86_64_STANDARD"
      instance_types = [var.cpu_node_instance_type]
      capacity_type  = var.cpu_capacity_type

      min_size     = var.cpu_node_min_size
      max_size     = var.cpu_node_max_size
      desired_size = var.cpu_node_desired_size

      labels = {
        workload = "platform"
      }
    }

    gpu = {
      ami_type       = "AL2023_x86_64_NVIDIA"
      instance_types = [var.gpu_node_instance_type]
      capacity_type  = var.gpu_capacity_type

      disk_size    = 200
      min_size     = var.gpu_node_min_size
      max_size     = var.gpu_node_max_size
      desired_size = var.gpu_node_desired_size

      labels = {
        accelerator = "nvidia"
        workload    = "llm"
      }

      taints = {
        gpu = {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      }
    }
  }

  cluster_addons = {
    coredns   = {}
    kube-proxy = {}
    vpc-cni   = {}
    aws-ebs-csi-driver = {}
  }
}

resource "aws_security_group" "deposition_efs" {
  name        = "${local.name_prefix}-efs"
  description = "Allow NFS from the VPC so EKS workloads can mount deposition storage"
  vpc_id      = module.vpc.vpc_id

  ingress {
    description = "NFS from private subnets"
    from_port   = 2049
    to_port     = 2049
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_efs_file_system" "depositions" {
  creation_token   = "${local.name_prefix}-depositions"
  encrypted        = true
  performance_mode = var.deposition_efs_performance_mode
  throughput_mode  = var.deposition_efs_throughput_mode
}

resource "aws_efs_mount_target" "depositions" {
  for_each = { for index, subnet_id in module.vpc.private_subnets : index => subnet_id }

  file_system_id  = aws_efs_file_system.depositions.id
  subnet_id       = each.value
  security_groups = [aws_security_group.deposition_efs.id]
}
