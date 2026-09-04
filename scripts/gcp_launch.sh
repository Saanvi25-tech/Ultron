#!/bin/bash
# scripts/gcp_launch.sh
# Create a GCP VM suitable for QLoRA/PEFT training of a 13B model (A100 40GB)
# NOTE: You must have the Google Cloud SDK installed and authenticated (gcloud auth login)
# Make sure your project has GPU quota for the chosen zone.

set -e
PROJECT=${PROJECT:-$(gcloud config get-value project)}
ZONE=${ZONE:-us-central1-a}
INSTANCE=${INSTANCE:-ultron-train-1}
MACHINE_TYPE=${MACHINE_TYPE:-a2-highgpu-1g}
IMAGE_FAMILY=ubuntu-2004-lts
IMAGE_PROJECT=ubuntu-os-cloud

if [ -z "$PROJECT" ]; then
  echo "No GCP project configured. Run: gcloud config set project <PROJECT_ID>"
  exit 1
fi

echo "Creating instance $INSTANCE in project $PROJECT zone $ZONE..."

gcloud compute instances create $INSTANCE \
  --project=$PROJECT \
  --zone=$ZONE \
  --machine-type=$MACHINE_TYPE \
  --accelerator=type=nvidia-tesla-a100,count=1 \
  --image-family=$IMAGE_FAMILY \
  --image-project=$IMAGE_PROJECT \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=200GB \
  --metadata=startup-script='#!/bin/bash\napt-get update\napt-get install -y --no-install-recommends build-essential dkms gcc make curl git python3 python3-venv python3-pip ca-certificates\n# Install NVIDIA drivers (may require a reboot, handled interactively if needed)\napt-get install -y nvidia-driver-525 || true\n# Install docker\napt-get install -y apt-transport-https ca-certificates gnupg lsb-release\ncurl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg\necho \"deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable\" > /etc/apt/sources.list.d/docker.list\napt-get update && apt-get install -y docker-ce docker-ce-cli containerd.io\n# Install nvidia container toolkit
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -\ndistribution=$(. /etc/os-release;echo $ID$VERSION_ID)\ncurl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list\napt-get update && apt-get install -y nvidia-docker2\nsystemctl restart docker || true\n' \
  --scopes=https://www.googleapis.com/auth/cloud-platform

echo "Instance created. Connect with: gcloud compute ssh $INSTANCE --zone=$ZONE"

echo "After SSHing, clone your repo and run the training steps described in scripts/gcp_training.md"
