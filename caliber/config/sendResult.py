# config.sendResult.py
import os
import json
from collections import defaultdict

class SendResult:
    def __init__(self, path=None, output=None):
        '''output==predicted model'''
        super(SendResult, self).__init__()
        self.file_name = "answersheet_4_03_youngpyoryu.json"
        if path != None:
            self.path = os.path.join(path, self.file_name)
        elif os.environ['HOME'] != None:
            self.path = os.path.join(os.environ['HOME'], self.file_name)
        elif os.environ['USER'] != None:
            self.path = os.path.join(os.environ['USER'], self.file_name)
        else:
            self.path = f"./{self.file_name}"
        self.output = output
        self.result_form = self.Transfer()
        
    def Transfer(self, test_image=None):
        '''test_image는 나중에 지우기!'''
        # result_form = {"answer":[dict([("file_name",None), ("result", [])] for num in range(len(self.output))]}
        # prediction, type : list --> prediction[0].items(), "labels" -> tensor.1d(float32)
        # prediction[0].keys() = ['boxes', 'labels', 'scores'] <= f_name 추가 작성
        if test_image:
            result_form = {"answer":[dict([("file_name",None), ("result", [])]) for num in range(len(test_image))]}
        else:
            result_form = {"answer":[dict([("file_name",None), ("result", [])]) for num in range(len(os.listdir("'/home/agc2021/dataset'/t3_*")))]}
        for num, fname in enumerate(test_image):
            result_form["answer"][num]["file_name"] = fname.lower()
            
            labels = [(0, 1), (1, 1), (2, 1), (3, 1), (3, 1)]
            result = defaultdict(int)
            for k, v in labels:
                result[f'c_{str(k+1)}'] += v
            sorted(result.items())
            result_form["answer"][num]["result"] = [dict([("label",k), ("count",str(result.get(k)))]) for k in result.keys()]
        return result_form
    
    def FIN(self):
        with open(self.path, 'w') as jsf:
            jsf.write(json.dumps(self.result_form, indent=4))