import os
import sys
import subprocess
from pynvml import *

def use_gpu(used_percentage=0.5):
    
    nvmlInit()
    gpu_num = nvmlDeviceGetCount()
    out = ""
    for i in range(gpu_num):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        used_percentage_real = info.used / info.total
        if out == "":
            if used_percentage_real < used_percentage:
                out += str(i)
        else:
            if used_percentage_real < used_percentage:
                out += "," + str(i)
    nvmlShutdown()
    
    return out

def set_rtx():
    os.environ["CUDA_VISIBLE_DEVICES"] = use_gpu(0.5) 
    
    if os.getenv("CUDA_VISIBLE_DEVICES") is None:
        print("It is not a good time!")
        sys.exit(-1)
    
    os.environ["OLLAMA_MAX_LOADED_MODELS"] = '4'
    os.environ["OLLAMA_NUM_PARALLEL"] = '4'
    command = "ollama serve"
    subprocess.run(command, shell=True)
    
if __name__ == "__main__":
    set_rtx()
