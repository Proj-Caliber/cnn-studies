# -*- python-mode -*-
# -*- coding: utf-8 -*-
import os
import argparse
import torch
import collections
# print(os.path.abspath(os.path.dirname(__file__)))
# # /Volumes/GitHub/organization-caliber/recycle_image_project/after_plastic/ai-challenge
parser = argparse.ArgumentParser(description='model params')
parser.add_argument('--mode', type=str, default='init', help="'init', 'update' or 'inference'")
parser.add_argument('--epoch', type=int, default=5, help='')
parser.add_argument('--batch', type=int, default=1, help='')
# 현재 수준에로는 lr_scheduler로 lr, momentum을 맞추는 것보다 임의 지정을 한 뒤, 나중에 수정하는 방향으로
# parser.add_argument('--lr-scheduler', type=float, default=, help='')
args = parser.parse_args()

try:
    assert args.mode in ['update', 'inference', 'init']
except AssertionError:
    print("mode로 올 수 있는 값은 init, update, inference 가 유일합니다.")

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # parallel가능 여부에 따라, logic은 달라야 함.
    # 현재 모델은 Multi-GPU에 대한 고려없이 작성한 코드임.

    base_dir = "/content/drive/MyDrive/Task/plastic-segmentation/Data"
    num_classes = 5
    momentum = 0.9
    learning_rate = 0.001
    step_size = 5
    # dockerfile 셋업 끝나면 하단의 주석 해제 후 상기의 config 제거
    weight_decay = 0.0005
    gamma = 0.1
    # base_dir = os.environ["BASE_DIR"]
    # num_classes = os.environ["NUM_CLASSES"]   
    # lr_scheduler = os.environ["LR_SCHEDULER"] # 해당 부분 코드를 제대로 작성한 경우 사용, 아니면 위의 값 사용.
    # step = os.environ["STEP"]

    # Data Pipelines
    from models.dataset.CaliberDataset import CustomDataset
    if args.mode == 'inference':
        # 이 부분을 통상 test로 하는 것으로 아는데.....
        dataset = CustomDataset()
    else:   # train(data >> train, validation, test)
        # {base_dir}/train
        dataset = CustomDataset()
        # {base_dir}/test
        test_dataset = CustomDataset() 
    
    # Model Pipelines
    from models.model import CaliberM
    model = CaliberM()
    # model extension 파라미터값을 입력하기 보단, 현재 있는 값 중 torch로 학습한 파일(pt, pth)만 가능하도록하고 예외처리구문 작성하기
    # 사용 모델에 따라 normalization이 다를 수 있음 -> transformer?? 그럼 기본은 데이터 로드는 tensor? or PIL?
