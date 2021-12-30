# main.sh
# linux --> container(Dockerfile) 할 때 참고용
export CUDA_VISIBLE_DEVICES=1
# local
# export WORKSPACE=/Volumes/GitHub/organization-caliber/recycle_image_project/after_plastic/ai-challenge
# gdrive
export WORKSPACE=/content/drive/Shareddrives/DataAnalytics/caliber-cnn
export NUM_CLASSES=5

# 
export MOMENTUM=0.9
export LEARNING_RATE=1e-3
export WEIGHT_DECAY=0.0005

export STEP=5 
export GAMMA=0.1
# lr_scheduler 구성 완료 시 주석해제

# # 코랩 환경이나 docker 등에선 하기의 내용 주석 해제(최종 업로드 시, zsh ~ 지우기)
# cd models
# bash vision.sh
# zsh vision.sh

# ./Dockerfile 수정중인 부분이랑 연계?
# pyenv shell gh-3.7.7
# chsh -s /bin/zsh
# poetry shell

python main.py
# python main.py --mode update --epoch 3 --batch 5
# python main.py --
# python main.py --
# python main.py --
