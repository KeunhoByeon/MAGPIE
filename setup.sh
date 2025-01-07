sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
sudo docker build -t magpie2025_sample -f Dockerfile .