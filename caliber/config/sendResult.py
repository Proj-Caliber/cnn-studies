# config.sendResult.py
import os
import json
from collections import defaultdict
# 해야할 부분 : 그들이 원한 방식으로 맞출 내용

class SendResult:
    # model 결과 받으면, 각 label별 cnt=1이 튜플 형식으로 쌓이게끔 작성
    # label: c_9, count: 9
    # answersheet_4_03_Caliber.json
    def __init__(self, path=os.getcwd(), output=None):
        self.path = os.path.join(path, "answersheet_4_03_Caliber.json")
        self.output = output
        self.result_form = self.Transfer()
    def Transfer(self):
        # tensor 내에 이미지 위치 정보도 있다는 가정하에 작성
        # defaultdict로 작성함이 맞는건가...?
        # self.file_name
        # self.result
        result_form = {"answer":[dict() for num in range(len(self.file_name))]}
        # {
        #     "answer":[{
        #         "file_name":"t3_9999.jpg",
        #         "result":[
        #             {"label":"c_9", "count":"9"}
        #         ]
        #     }]
        # }
        return result_form
    
    def FIN(self):
        with open(self.path, 'w') as jsf:
            jsf.write(json.dumps(self.result_form, indent=4, Type=str))