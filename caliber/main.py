import os
import torch
from config.envinfos import EnvInfos    #, engine
# from config.train import *
# from config.models import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

EnvInfos().memCheck()

# inference는 따로 .json으로 나와야 함.
# from config.resultForm import Result