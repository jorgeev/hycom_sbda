# hycom_sbda training/sampling image for AWS GPU instances.
# Build from the REPO ROOT (which is also the package root):
#   docker build -f aws/Dockerfile -t hycom_sbda .
#
# GPU ARCH MATTERS. The base tag must carry kernels for the instance's GPU, or
# every CUDA op fails at runtime:
#   * A100 / H100 (sm_80 / sm_90)           -> CUDA 12.6, the default below
#   * Blackwell: B200, RTX PRO (sm_100/120) -> needs CUDA 12.8+; cu126 wheels
#     ship no sm_100/sm_120 kernels at all.
# Override without editing this file:
#   docker build -f aws/Dockerfile \
#     --build-arg BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime \
#     -t hycom_sbda .
ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime
FROM ${BASE_IMAGE}

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Deps before source, for layer caching. The torch lines are stripped on
# purpose: requirements.txt pins torch off the cu126 wheel index (correct
# on-prem), and honouring it here would overwrite the base image's arch-matched
# build -- silently reintroducing the incompatibility on any other arch.
# The get_arch_list() print is a self-verifying smoke check baked into the layer:
# confirm the target GPU's sm_XX appears in the build log.
COPY requirements.txt .
RUN grep -v -e '^torch==' -e 'download.pytorch.org' requirements.txt > /tmp/requirements.docker.txt \
 && pip install -r /tmp/requirements.docker.txt \
 && python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_arch_list())"

# Source (see .dockerignore for exclusions).
COPY . .

# No ENTRYPOINT, only CMD -- so `docker run <img> bash aws/sample.sh` swaps the
# job with no wrapper machinery.
#
# REQUIRED run flags, both of them:
#   --ipc=host    Docker's default private IPC namespace starves the DataLoader
#                 workers of the shared memory and semaphores they need to
#                 bootstrap. They then hang SILENTLY at startup -- no crash, no
#                 error, no worker PIDs ever appear -- even with a large
#                 --shm-size. This flag is the fix.
#   --shm-size=16g  DataLoader workers use shared memory; the 64 MB default
#                 crashes them.
#
# Example (store streamed straight from S3 -- no mount needed):
#   docker run --gpus all --ipc=host --shm-size=16g \
#     -e HYCOM_STORE=s3://my-bucket/gom_nemo_diffusive.zarr \
#     -e CONFIG=diffusion/configs/prior_genda_masked.yaml \
#     -e OUT_DIR=/mnt/runs -e CACHE_DIR=/mnt/runs/cache -v /mnt/runs:/mnt/runs \
#     hycom_sbda
#
# A store on a HOST PATH must be bind-mounted, or the container sees nothing:
#   docker run --gpus all --ipc=host --shm-size=16g \
#     -e HYCOM_STORE=/mnt/data/gom_nemo_diffusive.zarr -v /mnt/data:/mnt/data:ro \
#     -e OUT_DIR=/mnt/runs -e CACHE_DIR=/mnt/runs/cache -v /mnt/runs:/mnt/runs \
#     hycom_sbda
#
# See aws/README.md -- in particular the RAM sizing note, which is the thing
# most likely to surprise you on a first real run.
CMD ["bash", "aws/launch.sh"]
