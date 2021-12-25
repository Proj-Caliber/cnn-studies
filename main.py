# -*- python-mode -*-
# -*- coding: utf-8 -*-
import os
import argparse
import torch
import collections
# print(os.path.abspath(os.path.dirname(__file__)))
# # /Volumes/GitHub/organization-caliber/recycle_image_project/after_plastic/ai-challenge
parser = argparse.ArgumentParser(prog='caliber', description='model params', argument_default=argparse.SUPPRESS)
parser.add_argument('--mode', type=str, default='init', help="'init', 'update' or 'inference'")
parser.add_argument('--epoch', type=int, default=5, help='')
parser.add_argument('--batch', type=int, default=1, help='')
parser.add_argument('--save-dir', type=str, default='assets/weights', help='')
# 현재 수준에로는 lr_scheduler로 lr, momentum을 맞추는 것보다 임의 지정을 한 뒤, 나중에 수정하는 방향으로
# parser.add_argument('--lr-scheduler', type=float, default=, help='')
args = parser.parse_args()

try:
    assert args.mode in ['update', 'inference', 'init']
except AssertionError:
    print("mode로 올 수 있는 값은 init, update, inference 가 유일합니다.")

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # parallel가능 여부에 따라, logic은 달라야 함. 현재 모델은 Multi-GPU에 대한 고려없이 작성한 코드임.

    # Data Pipelines
    from models.dataset.CaliberDataset import CustomDataset
    if args.mode == 'inference':
        # inference가 이미지 하나일 경우와 폴더일 경우를 고려하고 함수 수정해야할 듯
        test_dataset = CustomDataset(root = os.path.join(os.environ["WORKSPACE"], 'assets/data/inference'), mode = args.mode)
    else:   # train(data >> train, validation, test)
        # {base_dir}/train
        dataset = CustomDataset(root = os.path.join(os.environ["WORKSPACE"], 'assets/data/train'), mode = args.mode)
        # {base_dir}/test
        test_dataset = CustomDataset(root = os.path.join(os.environ["WORKSPACE"], 'assets/data/test'), mode = args.mode)
        
        torch.manual_seed(777)
        indices = torch.randperm(len(dataset)).tolist()
        
        train_dataset = torch.utils.data.Subset(dataset, indices[:-((dataset.__len__)-(test_dataset.__len__))])
        valid_dataset = torch.utils.data.Subset(dataset, indices[-((dataset.__len__)-(test_dataset.__len__)):])
    
    # 오늘 Data Pipelines
        
    
    # Model Pipelines
    from models.model import CaliberM
    model = CaliberM()
    # model extension 파라미터값을 입력하기 보단, 현재 있는 값 중 torch로 학습한 파일(pt, pth)만 가능하도록하고 예외처리구문 작성하기
    # 사용 모델에 따라 normalization이 다를 수 있음 -> transformer?? 그럼 기본은 데이터 로드는 tensor? or PIL?
