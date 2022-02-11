import json
import random
import numpy as np

# example dictionary that contains data like you want to have in json
overlapping = False

window = [  1,2,3 ,4,5,6,7]
h = 1
if overlapping :
    for item in range(len(window)) :
       window[item] = int(window[item]  *27)
    print(type(window))
    h = h *27 

layers = [2]#,4,5]
neurons = [32]#,64,512,256]
batch_size= [64]#,128]

random.seed(444)
# seeds =  [random.randrange(1, 9999 ) for i in range(5)]
 
seeds=[111]
print(seeds)
print(window)


regulizers = [None]#,'l1','l2']
dropouts = [0]#,0.2]
for stacktypes in ['']:#['G','I']: 

    if stacktypes =='G':
        stack_types_list = [  ( "generic" ,"generic","generic","generic","generic","generic","generic") ]
    elif stacktypes =='I':
        stack_types_list = [( "trend" ,"seasonality","trend" ,"seasonality","trend" ,"seasonality")]# , ( "generic" ,"generic","generic","generic","generic","generic","generic") ]

        ''' 
       power , map , cov  , cal
        '''
    model_input = [

         
       [True, False, False, False] ,
                #    [False, True, False, False]   ,
     #           [False, False, True, False]   , 
     #            [False, True, True, False]#,
                  # [False, False, False, True] ,
               #   [False, True, True, True], 
               #  [False, False, True, True],
                # [False, True, False, True]

                  ]
    #datasets=[("synth","solar_30min_filtered_synth.csv")] # ]

    datasets=[ ("QLD1","solar_30min_filtered_QLD1.csv",17)]#("QLD1","solar_30min_filtered_QLD1.csv")]#("synth","solar_30min_filtered_synth.csv")]
    # datasets= [("NSW1","solar_30min_filtered_NSW1.csv", 12)   , ("QLD1","solar_30min_filtered_QLD1.csv",17)]
    
    
    configurations=0



    for model_inputs in model_input:
        for stack_types in stack_types_list:
            for seed_value in seeds:
                for dataset in datasets:
                    for drop_rate in dropouts: 
                        for regulizer in regulizers:
                            for layer in layers: 
                                for w in window:
                                    for n_neuron in neurons:
                                        for batch in batch_size:
                                            dic=  {
                                            "exp": {
                                            "name": 'Ridge', #"NN_best_"+dataset[0],
                                                "seed_value": seed_value,
                                                "config_n":configurations
                                            },

                                            "comet_key":{
                                                "comet_api_key": "xG8s5XdJ2GdmGKVJSHfEHNTFK",
                                                "workspace": "sarah-almaghrabi"
                                            },
                                            "model":{
                                                "learning_rate": 0.001,
                                                "optimizer": "adam",
                                                "dense_units":n_neuron,
                                                "n_dense_layers": layer,
                                                "regulizer": regulizer,
                                                "dropout": drop_rate,
                                                "maps_coding_size": 16

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
                                                "patience":20,
                                                "checkpoint_save_best_only": True,
                                            "checkpoint_save_weights_only": True,
                                                "checkpoint_verbose": True,
                                                "tensorboard_write_graph": True
                                            },
                                            "dataset_file" :  {
                                                "file_name" : dataset[1],
                                                "siteName" :dataset[0],
                                                "samplePerDay"  : 27
                                                },
                                            "model_data" :  {

                                                "window" : w,
                                                "horizon" :h,
                                                "locations_n":dataset[2],
                                                "split_data_train"  : 0.74,
                                                "split_data_valid"  : 0,
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
                                                          #'SnowWater',
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