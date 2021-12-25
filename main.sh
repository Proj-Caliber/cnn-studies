# main.sh
# linux --> container(Dockerfile) 할 때 참고용
export CUDA_VISIBLE_DEVICES=1
export WORKSPACE=/Volumes/GitHub/organization-caliber/recycle_image_project/after_plastic/ai-challenge
# export BASE_DIR=/content/drive/MyDrive/Task/plastic-segmentation/Data
export NUM_CLASSES=5

# 
export MOMENTUM=0.9
export LEARNING_RATE=1e-3
export WEIGHT_DECAY=0.0005

# export STEP=5 # lr_scheduler 구성 완료 시 주석해제

# cd models
# bash vision.sh
# zsh vision.sh

# init
# pyenv shell gh-3.7.7
# chsh -s /bin/zsh
# poetry shell
# python main.py
# python main.py --mode update --epoch 3 --batch 5
# python main.py --
# python main.py --
# python main.py --
