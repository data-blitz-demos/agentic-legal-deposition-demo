# Copyright (c) 2026 Data-Blitz Inc. All rights reserved.
# License: Proprietary. See NOTICE.md.
# Author: Paul Harvener.

data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

locals {
  azs = length(var.availability_zones) > 0 ? var.availability_zones : slice(data.aws_availability_zones.available.names, 0, 3)

  name_prefix  = "${var.app_name}-${var.environment}"
  cluster_name = "${local.name_prefix}-eks"

  tags = {
    Application = var.app_name
    Environment = var.environment
    ManagedBy   = "terraform"
    Repository  = "agentic-legal-deposition-demo"
  }
}
