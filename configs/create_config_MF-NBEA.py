import json
import random
import numpy as np

'''
This file is used to create all required settings and hyperparameters for MF-NBEA
''' 

samplePerDay = 27
overlapping = False  #if want to use overlapping window or siliding window 

window = [  1,2,3 ,4,5,6,7]
h = 1
if overlapping :
    for item in range(len(window)) :
       window[item] = int(window[item]  *samplePerDay)
    print(type(window))
    h = h *samplePerDay 


batch_size= [64]#,128]
random.seed(444)
seeds =  [random.randrange(1, 9999 ) for i in range(10)]
for stacktypes in ['I']:#,'I']: 

    if stacktypes =='G':
        stack_types_list = [  ( "generic" ,"generic","generic","generic","generic","generic","generic") ]
    elif stacktypes =='I':
        stack_types_list = [( "trend" ,"seasonality")]

        ''' 
       power , map , cov  , cal
        '''
    model_input = [

         
      # [True, False, False, False] ,
       [False, True, False, False]   ,
     #           [False, False, True, False]   , 
     #            [False, True, True, False]#,
                  # [False, False, False, True] ,
               #   [False, True, True, True], 
               #  [False, False, True, True],
                # [False, True, False, True]

                  ]

    datasets= [("NSW1","solar_30min_filtered_NSW1.csv", 12)   , ("QLD1","solar_30min_filtered_QLD1.csv",17)]
    
    
    configurations=0



    for model_inputs in model_input:
        for stack_types in stack_types_list:
            for seed_value in seeds:
                for dataset in datasets:
                    for w in window:
                        for batch in batch_size:
                            dic=  {
                            "exp": {
                            "name": "MFNBEA_"+dataset[0],
                                "seed_value": seed_value,
                                "config_n":configurations
                            },

                            "comet_key":{
                                "comet_api_key": "",
                                "workspace": ""
                            },
                            "model":{
                                "learning_rate": 0.001,
                                "optimizer": "adam",
                                "maps_coding_size": 8,
                                "f2": 8, 
                                "f3":16,
                                "f4":8,
                                "kernels":5

                            },
                            "trainer":{
                                "num_epochs": 1000,
                                "batch_size": batch,
                                "validation_split":0.20,
                                "verbose_training": True
                            },
                            "callbacks":{
                                "checkpoint_monitor": "val_loss",
                                "checkpoint_mode": "min",
                                "patience":50,
                                "checkpoint_save_best_only": True,
                            "checkpoint_save_weights_only": True,
                                "checkpoint_verbose": True,
                                "tensorboard_write_graph": True
                            },
                            "dataset_file" :  {
                                "file_name" : dataset[1],
                                "siteName" :dataset[0],
                                "samplePerDay"  : samplePerDay
                                },
                            "model_data" :  {
                                "window" : w,
                                "horizon" :h,
                                "locations_n":dataset[2],
                                #"split_data_train"  : 0.74,
                                #"split_data_valid"  : 0,
                                "normlize": 0,
                                "use_power": True,
                                "use_weather": True,
                                "pca_comp": 0.9,
                                "correction_power_only":False,
                                "descriptive_data":False,
                                "model_power_only": model_inputs[0],
                                "model_weather_maps":model_inputs[1],
                                "model_weather_covariate":model_inputs[2],
                                "model_calendar_cov":model_inputs[3],
                                "stack_types":stack_types,
                                "overlapping_window":overlapping


                            },
                            "auto_corr": { "cal": False }
                            ,
                            "features" :  ['AlbedoDaily', 'CloudOpacity','AirTemp', 'Ghi','Dhi','Dni','Ebh','PrecipitableWater', 'RelativeHumidity',
                                            'SurfacePressure','WindSpeed10m' ]

                        }

                            
                            configurations +=1
                            if overlapping:
                                conf = str(configurations) +'_overlap'
                            else:
                                conf = configurations
                            if model_inputs[0] and not model_inputs[1] and not model_inputs[2]and not model_inputs[3]  :
                                file_name = 'model_configs/'+str(conf)+'NN_config_power'+stacktypes+'.json'
                            elif not model_inputs[0] and model_inputs[1] and not model_inputs[2]and not model_inputs[3]  :
                                file_name = 'model_configs/'+str(conf)+'NN_config_powerMap'+stacktypes+'.json'
                            elif not model_inputs[0] and not model_inputs[1] and  model_inputs[2]and not model_inputs[3]  :
                                file_name = 'model_configs/'+str(conf)+'NN_config_powerCov'+stacktypes+'.json'
                            elif not model_inputs[0] and not model_inputs[1] and not  model_inputs[2]and  model_inputs[3]  :
                                file_name = 'model_configs/'+str(conf)+'NN_config_powerCal'+stacktypes+'.json'


                            elif not model_inputs[0] and model_inputs[1] and  model_inputs[2]and not model_inputs[3]  :
                                file_name = 'model_configs/'+str(conf)+'NN_config_powerMapCov'+stacktypes+'.json'
                            elif not model_inputs[0] and model_inputs[1] and  model_inputs[2]and  model_inputs[3]  :
                                file_name = 'model_configs/'+str(conf)+'NN_config_powerMapCovCal'+stacktypes+'.json'

                                
                            elif not model_inputs[0] and not  model_inputs[1] and  model_inputs[2] and  model_inputs[3]  :
                                file_name = 'model_configs/'+str(conf)+'NN_config_powerCovCal'+stacktypes+'.json'                                           
                            elif not model_inputs[0] and   model_inputs[1] and not model_inputs[2] and  model_inputs[3]  :
                                file_name = 'model_configs/'+str(conf)+'NN_config_powerMapCal'+stacktypes+'.json'
                            
                                
                            
                                

                            with open(file_name, 'w') as outfile:
                                json.dump(dic, outfile,indent=4)