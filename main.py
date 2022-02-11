
from re import I
# from comet_ml import Experiment
import os
import random
import pandas as pd 
import tensorflow as tf
from keras import  Model 
from tensorflow import keras
from evaluater.model_evaluater import Evaluater
from models import MF_NBEA, autoencoder
from trainers.trainer import Trainer
import trainers.trainer_autoencoder  
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
import numpy as np 
from models.residual_learner import *
import copy
from pathlib import Path
from tensorboard.plugins.hparams import api as hp

def reshape_2d(data):
    if len( data.shape)  == 4:                
        data = data.reshape(data.shape[0],data.shape[1]*data.shape[2]*data.shape[3])
    elif len( data.shape)  == 3:
        data = data.reshape(data.shape[0],data.shape[1]*data.shape[2] )
    return data

def main(conf_file=0, iter_no=0):
    exp_description = conf_file.split('.')[0].split('_')[-1]


    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        if conf_file == 0 :
            args = get_args()
            config = process_config(args.config)

        else :
            print('run using python script -- config: ',conf_file)
            config = process_config(conf_file)
        
        #uncomment if you want to use comet_ml 
        experiment =  '' #Experiment(
            # api_key=config.comet_key.comet_api_key, 
            # project_name=config.exp.name, 
            # workspace=config.comet_key.workspace,
            # auto_histogram_tensorboard_logging=True,
            # auto_histogram_gradient_logging=True,
            
            # log_code=False,
            # log_graph=False,
            # auto_output_logging =False,
            # disabled=True
            
            # )

        # experiment.set_name(config.exp.config_n)
        
    except:
        print("missing or invalid arguments")
        exit(0)

    #[actual , predictions, acc ] 
    res_dec  = {}
    evaluater  = Evaluater( config=config,experiment=experiment)

    # create the experiments dirs
    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir,config.callbacks.checkpoint_dir_autenc])


    ovelapping= config.model_data.overlapping_window
    if ovelapping:
         from data_loader.Data_util_overlapping  import DataUtil
         windowing = 'overalap'
    else: 
        from data_loader.Data_util import DataUtil 
        windowing = ''

    print('Create the data generator.')
    data_loader = DataUtil(config,experiment) 
    
    
 
    my_file = Path('encoder_'+config.dataset_file.siteName+'.h5')

    if  not my_file.is_file():
        autoencoder_model =  autoencoder.AutoEncoder(config,experiment)


        ##autoencode the map features
        auto_trainer = trainers.trainer_autoencoder.Trainer(model= autoencoder_model.model, data=data_loader.get_train_data(), config=config,experiment=experiment )
        print('start train')
        

        if  not config.model_data.model_power_only :
            auto_trainer.train( plot=False)
            encoder = Model(inputs=autoencoder_model.model.input, outputs=autoencoder_model.model.get_layer('encoder').output)
            
            # save the encoder to file
            encoder.save('encoder_'+config.dataset_file.siteName+'.h5')
        
            print('finish train')
            

        #'''
    else:
        print('load encoder')
        encoder = keras.models.load_model('encoder_'+config.dataset_file.siteName+'.h5')
        encoder.compile(optimizer='adam', loss='mse')

    
    
    if  not config.model_data.model_power_only :
        ##encode train and test maps 
        data_loader.encode_maps_data()

    #check data shapes 
    for data in data_loader.get_train_data()[0]:
        print('train\n',data.shape)
    
    print('maps data are encoded ')


    print('Create the model.')
    model_name =   config.exp.name.split('_')[0].upper()
    print(model_name)
   
    if( model_name== "MFNBEA"):
        ### hyperparams for optimization 
        HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([32]))#, 64]))
        HP_NUM_IN_DIM = hp.HParam('num_in_dim', hp.Discrete([64]))
        HP_NUM_OUT_DIM = hp.HParam('num_out_dim', hp.Discrete([32]))
        HP_BLOCKS_PER_STACK = hp.HParam('nb_blocks_per_stack', hp.Discrete([1]))
        HP_THETAS_DIM = hp.HParam('thetas_dim', hp.Discrete([3]))
        HP_NBEATS_UNITS = hp.HParam('nbeats_units', hp.Discrete([128]))
        HP_NB_HARMONICS = hp.HParam('nb_harmonics', hp.Discrete([8]))
        HP_DELTA = hp.HParam('delta', hp.RealInterval(1.5,1.55))
        
        METRIC_ACCURACY = 'mse'
        HPARAMS=[HP_NUM_UNITS,HP_NUM_IN_DIM,HP_NUM_OUT_DIM,HP_BLOCKS_PER_STACK, HP_THETAS_DIM ,HP_NBEATS_UNITS,HP_NB_HARMONICS,HP_DELTA]  
       
        log_dir = os.path.normpath( config.callbacks.tensorboard_log_dir)+ 'HyperParams' 

        with tf.summary.create_file_writer(log_dir).as_default():
          hp.hparams_config(hparams=HPARAMS, metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')])

        def call_existing_code(units,in_dim,out_dim , nb_blocks_per_stack,thetas_dim, nbeats_units ,nb_harmonics ,delta):
             
            model = MF_NBEA.MF_NBEA(config,experiment,units,in_dim,out_dim,nb_blocks_per_stack,thetas_dim, nbeats_units ,nb_harmonics )#.model


            from tensorflow.keras.losses import Huber
            # from keras.optimizers import Adam
            model.compile(
                loss= Huber(delta=delta) ,
                optimizer=config.model.optimizer,
                metrics=['mae']
            )
            
            return model

    else:
        print("Not valid experiment name in the config")
        exit(0)




    print('Create the trainer')
    print('start train')
    
    rng = random.Random(config.exp.seed_value)
    session_index = 0
    # random search
    num_session_groups = 1
    sessions_per_group = 1
    #number of search space to consider 
    for group_index in range(num_session_groups):
      print('##############group_index (',str(group_index),' ) starated:##########')  
      hparams = {h: h.domain.sample_uniform(rng) for h in HPARAMS}
      print(hparams)
      print('selected_params:')
      selected_params = {}
      for k in hparams.keys():

          print(k.name ,' >>>>  ' , hparams[k] )
          selected_params[k.name ] = hparams[k]
      hparams_string = str(hparams)
      ##run the selected paramerters multiple times 
      for repeat_index in range(sessions_per_group):
        keras.backend.clear_session()  
        print('-----------repeat_index (',str(repeat_index),' ) --------------------:')  

        session_id = str(session_index)
        session_index += 1
        trainer = trainers.trainer.Trainer(
                model= call_existing_code( 
                   units=selected_params['num_units'],
                   in_dim = selected_params['num_in_dim'],
                   out_dim = selected_params['num_out_dim'] ,
                   nb_blocks_per_stack =selected_params['nb_blocks_per_stack'] ,
                   thetas_dim =selected_params['thetas_dim'] ,
                   nbeats_units  =selected_params['nbeats_units'] ,
                   nb_harmonics  =selected_params['nb_harmonics'] , 
                   delta =  selected_params['delta']  
               
                   )  
               , data=data_loader.get_train_data(), config=config,experiment=experiment , session_id = '_s'+str(session_id))

        #fitness_func(hparams, session_id)
        model = trainer.train(hparams)

        print('finish train')
        
        use_residual = True
        single_model = True

        if single_model : 
            if config. model_data.descriptive_data:
    
                details='_DS_single'+'_Pfeat_'+str(config.model_data.use_power)+'_Wfeat_'+str(config.model_data.use_weather)+windowing
            else:
                details='_single'+'_Pfeat_'+str(config.model_data.use_power)+'_Wfeat_'+str(config.model_data.use_weather)+windowing

            ##-------evaluate training data -----------------

            trainX =   data_loader.get_train_data()[0]
            trainy =  data_loader.get_train_data()[1]


            print('Start prediction (training).')
            predictions=np.zeros(shape=trainy.shape)


            for ind in range(trainX[0].shape[0]): #loop all examples 
                x=[] 
                for input_item  in trainX: ## for each input kind in the input list 
                    inp_shape= [1]
                    for sh in input_item[ind].shape :
                        inp_shape.append (sh) 

                    x.append(input_item[ind].reshape(inp_shape) )
            
                predictions[ind] = model.predict(x)[0,:,0] ##  outputshpe will be [1,27,1]


            predictions.shape    =  predictions.shape[0], predictions.shape[1]
            #if data normlized apply the inverse 
            # if 0 not normlize, 1 normalize 
            if config.model_data.normlize == 1: 
                #de normlize the predictions   
                predictions = data_loader.denormilse_data(data = predictions, fitted_scalar = data_loader.scalars )
                actuals = data_loader.denormilse_data(data = data_loader.get_train_data()[1], fitted_scalar = data_loader.scalars )

            actuals =  data_loader.get_train_data()[1]
    
            train_res = evaluater .computeAccuracy(actData=actuals, predData= predictions ,maxCapacity=data_loader.maxCapacity,
            details= details+'_training_No_'+str(iter_no)+exp_description+'_'+str(config.exp.seed_value))




            ##-------evaluate testing data -----------------
            testX =   data_loader.get_test_data()[0]
            testy =  data_loader.get_test_data()[1]

            print('Start prediction (testing).')
            predictions=np.zeros(shape=testy.shape)
            print('predictions shape',predictions.shape)

            for ind in range(testX[0].shape[0]): #loop all examples 
                x=[] 
                for input_item  in testX: ## for each input kind in the input list 
                    inp_shape= [1]
                    for sh in input_item[ind].shape :
                        inp_shape.append (sh) 

                    x.append(input_item[ind].reshape(inp_shape) )
            
                predictions[ind] = model.predict(x)[0,:,0] ##  outputshpe will be [1,27,1]

            predictions.shape    =  predictions.shape[0], predictions.shape[1]

            #if data normlized apply the inverse 
            # if 0 not normlize, 1 normalize 
            if config.model_data.normlize == 1: 
                #de normlize the predictions   
                predictions = data_loader.denormilse_data(data = predictions, fitted_scalar = data_loader.scalars )
                actuals = data_loader.denormilse_data(data = data_loader.get_test_data()[1], fitted_scalar = data_loader.scalars )

                print('after de scale ')
                print('predictions shape',predictions.shape)
            else:
                actuals = data_loader.get_test_data()[1]  


            print('Evalaute predictions ')

            test_res = evaluater .computeAccuracy(actData=actuals, predData= predictions ,maxCapacity=data_loader.maxCapacity,
            details=  details+ '_testing_No_'+str(iter_no)+exp_description+'_'+str(config.exp.seed_value))
            res_dec [model_name]= { 'train': train_res , 'test':  test_res}


 
            #print(res_dec)
            #exit()
        if use_residual :
            print('start ensemble part ')
            pca_comp = config.model_data.pca_comp
            correction_power_only = config.model_data.correction_power_only  ##use only power to train the residual model 

            ##build predictions processing part 
            gbnn= [] 
            # for n in range(1,ensemble_n ):
            for n in [1]: # 1 residual learner used 

                #information for saving results 
                if config. model_data.descriptive_data:
                    details='_DS_ens_'+str(n) +'_pca_'+str(pca_comp).replace('.','')+'_Pfeat_'+str(config.model_data.use_power)+'_Wfeat_'+str(config.model_data.use_weather)+'_onlyP_'+str(correction_power_only)+windowing
                else:
                    details='_ens_'+str(n) +'_pca_'+str(pca_comp).replace('.','')+'_Pfeat_'+str(config.model_data.use_power)+'_Wfeat_'+str(config.model_data.use_weather)+'_onlyP_'+str(correction_power_only)+windowing

                
                trainX =   data_loader.get_train_data()[0]
                trainy =  data_loader.get_train_data()[1]


                prediction_processing  =  get_ensemble( trainX, trainy,model = model ,pca_comp=pca_comp, n=n,correction_power_only =correction_power_only )
                print('----------------')
                final_predictions= np.zeros( shape=(len(prediction_processing),data_loader.get_train_data()[1].shape[0], data_loader.get_train_data()[1].shape[1] ))

                ##Evaluate training data
                #'''
                for i , _model in enumerate(prediction_processing): 
                    if i ==0 : 
                        continue

                    
                    else:

                        if correction_power_only: 
                            trainX_2D = trainX[0] #only power 

                        elif config.auto_corr.cal :  #power, weather and claendar             
                            trainX_2D = np.concatenate([trainX[2],trainX[0],trainX[3] ], axis=-1)
                   
                        else: #power and weather   
                            trainX_2D = np.concatenate([trainX[2],trainX[0] ], axis=-1)
                        
                        trainX_2D = reshape_2d(data=trainX_2D) 
                    
                        if pca_comp>0 :
                            pca =  PCA(n_components=pca_comp).fit(trainX_2D)
                            trainX_2D =pca.transform(trainX_2D)
 

                ##Evaluate testing data
                final_predictions= np.zeros( shape=(len(prediction_processing),data_loader.get_test_data()[1].shape[0], data_loader.get_test_data()[1].shape[1] ))
                 
                testX =  data_loader.get_test_data()[0] 
                testy =  data_loader.get_test_data()[1]

                ## by example  
                for test_index in range(testX[0].shape[0]):
                    x =[] 
                    
                    for input_item  in testX:
                        inp_shape= [1]
                        for sh in input_item[test_index].shape :
                            inp_shape.append (sh) 
                        #print('inp_shape:',inp_shape)
                        x.append(input_item[test_index].reshape(inp_shape) )
             
                    for i , _model in enumerate(prediction_processing): 
                        if i ==0 : 
                            final_predictions[i][test_index] = _model.predict( x )[:,:,0]                      
                        else:
                            if correction_power_only: 
                                testX_2D = x[0] #only power 

                            elif config.auto_corr.cal :  #power, weather and claendar             
                                testX_2D = np.concatenate([x[2],x[0],x[3] ], axis=-1)
                    
                            else: #power and weather   
                                testX_2D = np.concatenate([x[2],x[0] ], axis=-1)
                        
                        

                            testX_2D = reshape_2d(data=testX_2D) 
                            if pca_comp>0 :
                                testX_2D =pca.transform(testX_2D)

                            final_predictions[i][test_index] = _model.predict( testX_2D)[0]

            
                predictions = final_predictions.sum(axis=0)
                print( 'predictions:',predictions.shape)

                #if data normlized apply the inverse 
                # if 0 not normlize, 1 normalize 
                if config.model_data.normlize == 1: 
                    #de normlize the predictions   
                    predictions = data_loader.denormilse_data(data = predictions, fitted_scalar = data_loader.scalars )
                    actuals = data_loader.denormilse_data(data = data_loader.get_test_data()[1], fitted_scalar = data_loader.scalars )
                else:
                    actuals =  data_loader.get_test_data()[1]
            
                
                    test_res = evaluater .computeAccuracy(actData=actuals, predData= predictions ,maxCapacity=data_loader.maxCapacity,details=details+'_testing_No_'+str(iter_no)+exp_description+'_'+str(config.exp.seed_value)+windowing)
            

                res_dec ['GB'+model_name+'_'+str(n)]= { 'train': train_res , 'test':  test_res}



        


                gbnn.append(res_dec['GB'+model_name+'_'+str(n)]['test'][-1]['MRE'].values[-1])
    

                 

        print(res_dec)
    #exit()

















if __name__ == '__main__':
    main()
    

    