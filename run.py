import main 
import time
import os
from os import listdir
from os.path import isfile, join

cwd =  os.getcwd() +'/configs/model_configs/'#
json_files = [f for f in os.listdir(cwd) if os.path.isfile(os.path.join(cwd, f))]

for file in json_files:
    for iter in range(1):
        main.main('configs/model_configs/'+file , iter_no = iter )
        
    if os.path.exists('configs/model_configs/'+file):
       os.remove('configs/model_configs/'+file)
#     time.sleep(60) # 60 seconds to upload the results to comet
 
 

