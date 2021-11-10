import os
from config.envinfos import EnvInfos    #, engine

# from config.train import *
# from config.models import *

EnvInfos().memCheck()

# inference는 따로 .json으로 나와야 함.
# from config.resultForm import Result