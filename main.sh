# main.sh
# linux --> container(Dockerfile) 할 때 참고용
export CUDA_VISIBLE_DEVICES=1
export BASE_DIR=/content/drive/MyDrive/Task/plastic-segmentation/Data
export NUM_CLASSES=5

# 
MOMENTUM = 0.9
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005

# export STEP=5 # lr_scheduler 구성 완료 시 주석해제

cd models
bash vision.sh
cd ../

# init
python main.py
python main.py --mode update --epoch 3 --batch 5
# python main.py --
# python main.py --
# python main.py --
