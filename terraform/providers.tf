# Copyright (c) 2026 Data-Blitz Inc. All rights reserved.
# License: Proprietary. See NOTICE.md.
# Author: Paul Harvener.

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = local.tags
  }
}
