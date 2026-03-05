# Terraform AWS Infra (ECR + EKS + GPU LLM Nodes)

This directory provisions the AWS infrastructure layer for the application:

- VPC with public/private subnets
- ECR repositories for the API and LLM runtime images
- EKS cluster
- CPU managed node group for platform services
- GPU managed node group for LLM inference
- EFS for deposition file storage

It is intentionally paired with the existing Kubernetes manifests in:

- `deploy/k8s/aws`

Terraform creates the cloud resources. The Kubernetes manifests remain the workload layer.

## Provisioned resources

- `terraform-aws-modules/vpc/aws`
- `terraform-aws-modules/eks/aws`
- `aws_ecr_repository` (API)
- `aws_ecr_repository` (LLM runtime)
- `aws_efs_file_system`
- `aws_efs_mount_target`

## Node group layout

- `system`
  - CPU nodes for `api`, `couchdb`, `neo4j`, and observability services
- `gpu`
  - NVIDIA GPU nodes for the LLM runtime (`ollama` or later `vLLM`)
  - labels: `accelerator=nvidia`, `workload=llm`
  - taint: `nvidia.com/gpu=true:NoSchedule`

## Usage

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
terraform init
terraform plan
terraform apply
```

## After apply

1. Update kubeconfig using the `kubectl_config_command` output.
2. Patch `deploy/k8s/aws/efs-storageclass.yaml` with the `deposition_efs_file_system_id` output.
3. Push your Docker images to the ECR repositories from the outputs.
4. Apply the Kubernetes manifests in `deploy/k8s/aws`.

## Notes

- This stack uses `AL2023_x86_64_STANDARD` for CPU nodes and `AL2023_x86_64_NVIDIA` for GPU nodes.
- The GPU node group is isolated for LLM workloads only.
- For production, you will still need to install cluster-side components such as:
  - AWS Load Balancer Controller
  - EFS CSI driver
  - NVIDIA device plugin
