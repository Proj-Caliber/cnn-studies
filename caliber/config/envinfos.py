import os
import psutil
import numpy as np
import pandas as pd
import torch
import GPUtil as GPU
import humanize

class EnvInfos:
    def __init__(self):
        self.pid = os.getpid()                                              # port id
        self.memory_usage_dict = dict(psutil.virtual_memory()._asdict())    # general RAM usage(svmem)
        self.process = psutil.Process(self.pid)                             # current process RAM usage
        self.GPUs = GPU.getGPUs()
        self.iGPU = self.GPUinfos()

    def memCheck(self):
        print(f"""
        {'+'*50}\tmemory usage check\t{'+'*50}
        {'+'*10}\t\t\tGen RAM Free : {humanize.naturalsize(self.memory_usage_dict['free'])}\t\t||\t\tProc size : {humanize.naturalsize(self.process.memory_info().rss)}\t\t\t{'+'*10}
        {'+'*10}\t\t\tGPU NAME : {self.iGPU['GDevNM']}\t||\t\tGPU RAM Util : {self.iGPU['GUtil']}\t\t\t{'+'*10}
        {'+'*10}\t\tGPU RAM Free : {self.iGPU['GFree']}\t||\tUsed : {self.iGPU['GUsed']}\t\t||\tTotal : {self.iGPU['GTotal']}\t\t\t{'+'*10}
        {'+'*50}\t\tEnd\t\t{'+'*50}""")

    def CPUinfos(self):
        '''
        # VMRAM general RAM usage
        # total, availabel, percent, used, free, active, inactive, buffers, cached, shared, slab
        '''
        VRAM = self.cvtDICTtoDF(self.memory_usage_dict)
        return VRAM

    def PROCinfos(self):
        '''
        # VM -> process size
        # rss, vms, shared, text, lib, data, dirty <- memory_info
        # user, system, children_user, children_system <- cpu_times
        '''
        pmem = dict(self.process.memory_info()._asdict())
        PMEM = self.cvtDICTtoDF(pmem)

        # cpu
        pcpu_infos = {"pcpu_num" : self.process.cpu_num(), "pcpu_affinity" : self.process.cpu_affinity(),
                      "pcpu_percent" : self.process.cpu_percent(), "pcpu_times" : dict(self.process.cpu_times()._asdict())} # affinity : [0,...,11]    # 12
        # get memory and GPU usage of this PID as well
        with self.process.oneshot():
            print(pcpu_infos)
        
    def GPUinfos(self):
        '''
        # units <- MB, 9.99... => *(1024**2), *100 => humanize.naturalsize(size), "%"
        # device_name, capacity
        # time을 구하기까지는 시간이 부족함
        '''
        GFree = humanize.naturalsize(self.GPUs[0].memoryFree*(1024**2))
        GUsed = humanize.naturalsize(self.GPUs[0].memoryUsed*(1024**2))
        GUtil = str(round(self.GPUs[0].memoryUtil*(100), 4)) + "%"
        GTotal = humanize.naturalsize(self.GPUs[0].memoryTotal*(1024**2))
        try:
            GDevNM = torch.cuda.get_device_name(0)
        except:
            GDevNM = "Not Connected"
        # Gresult = GPU.showUtilization()
        return {"GDevNM" : GDevNM, "GFree" : GFree, "GUsed" : GUsed, "GUtil" : GUtil, "GTotal" : GTotal}
    
    def cvtDICTtoDF(self, dict_object):
        cvt_units = np.array([humanize.naturalsize(val).split(' ') for val in dict_object.values()])
        datas = {0:np.array([val for val in dict_object.values()]).astype('float'), 
                 1:np.array([float(val[0]) for val in cvt_units]), 
                 2:np.array([val[1] if val[1] != "Bytes" else "%(or None)" for val in cvt_units])}
        DF = pd.DataFrame(data=datas, index=dict_object.keys()).rename(columns={0:"org_size", 1:"cvt_size", 2:"units"})
        return DF