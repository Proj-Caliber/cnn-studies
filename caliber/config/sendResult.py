# config.sendResult.py
import os
import json
from collections import defaultdict

class SendResult:
    def __init__(self, path=None, output=None):
        '''output==predicted model'''
        super(SendResult, self).__init__()
        if path != None:
            self.path = os.path.join(path, "answersheet_4_03_youngpyoryu.json")
        else:
            self.path = "./answersheet_4_03_youngpyoryu.json"
        self.output = output
        self.result_form = self.Transfer()
        
    def Transfer(self, test_image):
        '''test_image는 나중에 지우기!'''
        # result_form = {"answer":[dict([("file_name",None), ("result", [])] for num in range(len(self.output))]}
        # prediction, type : list --> prediction[0].items(), "labels" -> tensor.1d(float32)
        result_form = {"answer":[dict([("file_name",None), ("result", [])]) for num in range(len(test_image))]}
        for num, fname in enumerate(test_image):
            result_form["answer"][num]["file_name"] = fname
            
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