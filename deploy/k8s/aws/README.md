# AWS EKS Deployment (Kubernetes + GPU LLM)

This folder deploys the full app stack on EKS:

- `api` (FastAPI app)
- `couchdb` (StatefulSet + EBS)
- `ollama` (GPU-backed LLM service)
- `deposition-files` (EFS-backed RWX volume mounted at `/data/depositions`)
- ALB Ingress in front of the API

## Prerequisites

- AWS account and credentials configured.
- Existing ECR repo for the app image.
- EFS file system for deposition files.
- EKS add-ons/controllers:
  - EBS CSI driver
  - EFS CSI driver
  - AWS Load Balancer Controller

## 1) Create the EKS cluster (CPU + GPU node groups)

Template config:

- `deploy/eks/cluster.eksctl.yaml`

Create cluster:

```bash
scripts/aws/create_eks_cluster.sh
```

## 2) Build and push the API image

```bash
AWS_ACCOUNT_ID=<account-id> \
AWS_REGION=us-east-1 \
ECR_REPOSITORY=agentic-legal-deposition-demo \
IMAGE_TAG=$(git rev-parse --short HEAD) \
scripts/aws/build_and_push.sh
```

Resulting image format:

```text
<account-id>.dkr.ecr.<region>.amazonaws.com/<repo>:<tag>
```

## 3) Update required placeholders

Edit these before deployment:

- `/Users/paulharvener/workspace/demos/agentic-legal-deposition-demo/deploy/k8s/aws/efs-storageclass.yaml`
- `fileSystemId: fs-REPLACE_ME`
- `deploy/k8s/aws/ingress.yaml`
  - `alb.ingress.kubernetes.io/certificate-arn`
  - `spec.rules[0].host`
- `deploy/k8s/aws/couchdb-secret.yaml`
  - `COUCHDB_PASSWORD`
- `deploy/k8s/aws/api-secret.yaml`
  - `OPENAI_API_KEY` (optional if only using Ollama)

## 4) Deploy to EKS

```bash
API_IMAGE=<full-ecr-image-uri> \
scripts/aws/deploy_k8s.sh
```

The deploy script applies kustomize manifests, patches the API image, and waits for rollouts.

## 5) Upload deposition files to EFS

Place `.txt` files in directories on the EFS mount. The API reads from `/data/depositions` in the container.

Examples expected by the UI:

- `/data/depositions/default`
- `/data/depositions/oj_simpson`
- `/data/depositions/trumpVsCarroll`

## 6) Verify

```bash
kubectl -n deposition-demo get pods
kubectl -n deposition-demo get svc
kubectl -n deposition-demo get ingress
kubectl -n deposition-demo logs deploy/api --tail=100
kubectl -n deposition-demo logs deploy/ollama --tail=100
```

API readiness endpoint:

```bash
kubectl -n deposition-demo port-forward svc/api 8000:80
curl -sSf http://127.0.0.1:8000/api/thought-streams/health
```

## Notes

- `DEFAULT_LLM_PROVIDER=ollama` is set in `api-configmap.yaml` so startup does not require OpenAI.
- GPU scheduling is enforced with `nodeSelector: accelerator=nvidia` and a GPU toleration in `ollama-deployment.yaml`.
- If `nvidia.com/gpu` is not allocatable on the GPU nodes, install the NVIDIA device plugin on the cluster before deploying `ollama`.
- Re-run the model loader job when changing models:

```bash
kubectl -n deposition-demo delete job ollama-model-loader --ignore-not-found
kubectl -n deposition-demo apply -f deploy/k8s/aws/ollama-model-loader-job.yaml
```
