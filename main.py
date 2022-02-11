
from re import I
# from comet_ml import Experiment
import os

import pandas as pd 
import tensorflow as tf
# from tensorflow import keras 
from keras import  Model 

from tensorflow.keras.models import load_model

from keras.utils.vis_utils import plot_model
from tensorflow import keras

from evaluater.model_evaluater import Evaluater
from models_grid import NN, autoencoder
from trainers.trainer import Trainer
import trainers.trainer_autoencoder  
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
import numpy as np 
from models_grid.ensemble_utils import *
import matplotlib.pyplot as plt
import copy
from sklearn.linear_model import Ridge,RidgeCV
from pathlib import Path
from tensorboard.plugins.hparams import api as hp

# function optimized to run on gpu 
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

    #  dic for the results of the different models [  model_name: [accuracy_training , predictions_training, accuracy_testing, prediction_testing  ]]
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

    if  True:#not my_file.is_file():
        autoencoder_model =  autoencoder.AutoEncoder(config,experiment)


        ##autoencode the map features
        auto_trainer = trainers.trainer_autoencoder.Trainer(model= autoencoder_model.model, data=data_loader.get_train_data(), config=config,experiment=experiment )
        print('start train')
        ## print('Build a new model with the best weights ')
        # auto_trainer.model.fit(
        #     # self.data[0], self.data[1],
        #     data_loader.get_train_data()[0][1], data_loader.get_train_data()[0][1],
        #     batch_size=1,
        #     verbose=1,
        #     epochs=1)

        if  not config.model_data.model_power_only :
            auto_trainer.train( plot=False)
            #checkpoint_filepath = os.path.join(config.callbacks.checkpoint_dir, '%s-{epoch:02d}.ckpt' % config.exp.name)
            #checkpoint_dir = os.path.dirname(checkpoint_filepath)
            # define an encoder model (without the decoder)
            #recostructed = autoencoder_model.model.predict(  data_loader.get_train_data()[0][1] )

            encoder = Model(inputs=autoencoder_model.model.input, outputs=autoencoder_model.model.get_layer('encoder').output)
            
            plot_model(encoder, 'encoder_no_compress.png', show_shapes=True)
            # save the encoder to file
            encoder.save('encoder_'+config.dataset_file.siteName+'.h5')
        
            print('finish train')
        #plt.plot(data_loader.get_train_data()[0][1][0].flatten() , label='actual')
        #plt.plot(recostructed[0].flatten() , label='reconstructed')
        #plt.legend()
        #plt.show()

        #'''
    else:
        print('load encoder')
        encoder = keras.models.load_model('encoder_'+config.dataset_file.siteName+'.h5')
        encoder.compile(optimizer='Nadam', loss='mse')

    

    # print((np.square(data_loader.get_train_data()[0][1] - recostructed)).mean(axis=1) )
    # print((np.square(data_loader.get_train_data()[0][1] - recostructed)).mean(axis=0) )
    
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
   
    if( model_name== 'NN'):
        ### hyperparams for optimization 
        HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([32]))#, 64]))
        HP_NUM_IN_DIM = hp.HParam('num_in_dim', hp.Discrete([64]))
        HP_NUM_OUT_DIM = hp.HParam('num_out_dim', hp.Discrete([32]))
        HP_BLOCKS_PER_STACK = hp.HParam('nb_blocks_per_stack', hp.Discrete([1]))
        HP_THETAS_DIM = hp.HParam('thetas_dim', hp.Discrete([3]))
        HP_NBEATS_UNITS = hp.HParam('nbeats_units', hp.Discrete([128]))
        HP_NB_HARMONICS = hp.HParam('nb_harmonics', hp.Discrete([8]))
        HP_DELTA = hp.HParam('delta', hp.RealInterval(1.5,1.55))
        #HP_DELTA = hp.HParam('delta', hp.RealInterval(0.2 , 3.0) )




        #HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.0 , 0.2))
        #HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd', 'rmsprop']))
        METRIC_ACCURACY = 'mse'
        HPARAMS=[HP_NUM_UNITS,HP_NUM_IN_DIM,HP_NUM_OUT_DIM,HP_BLOCKS_PER_STACK, HP_THETAS_DIM ,HP_NBEATS_UNITS,HP_NB_HARMONICS,HP_DELTA]  
       
        #log_dir = os.path.normpath( config.callbacks.tensorboard_log_dir)+ '\\'+config.dataset_file.siteName +'_HyperParams' 
        log_dir = os.path.normpath( config.callbacks.tensorboard_log_dir)+ 'HyperParams' 

        with tf.summary.create_file_writer(log_dir).as_default():
          hp.hparams_config(hparams=HPARAMS, metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')])

        def call_existing_code(units,in_dim,out_dim , nb_blocks_per_stack,thetas_dim, nbeats_units ,nb_harmonics ,delta):
             
            model = NN.NN(config,experiment,units,in_dim,out_dim,nb_blocks_per_stack,thetas_dim, nbeats_units ,nb_harmonics )#.model
            
            #compile the model 

            '''def build_model(hp):
                kernel_size = hp.Choice("kernel", [5])
                model  = NN.NN(config,experiment)
                print(model)
                print(model.models)
                complete_model = model
                return complete_model
            #model = model.build_model  
            '''
            from tensorflow.keras.losses import Huber

            # from keras.optimizers import Adam
            model.compile(
                loss= Huber(delta=delta) ,
                optimizer=config.model.optimizer,
                metrics=['mae']
            )
            #print(model.summary())

            plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

            return model


    else:
        print("Not valid experiment name in the config")
        exit(0)




    print('Create the trainer')

    # ## grid search
    #trainer = trainers.trainer_grid.Trainer(model= build_model  , data=data_loader.get_train_data(), config=config,experiment=experiment,model_name=model_name)
    print('start train')
    # print('Build a new model with the best weights ')
    
    import random
    rng = random.Random(config.exp.seed_value)
    session_index = 0
    # random search
    num_session_groups = 1
    sessions_per_group = 1
    #number of search space to consider 
    for group_index in range(num_session_groups):
      print('###########################################################################group_index (',str(group_index),' ) starated:##########')  
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
        print('---------------------------------------------------------repeat_index (',str(repeat_index),' ) --------------------:')  

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
                #    delta = np.round(selected_params['delta'],2)
                   delta =  selected_params['delta']  
               
                   )  
               , data=data_loader.get_train_data(), config=config,experiment=experiment , session_id = '_s'+str(session_id))

        #fitness_func(hparams, session_id)
        model = trainer.train(hparams)

        print('finish train')

     
        #checkpoint_filepath = os.path.join(config.callbacks.checkpoint_dir, '%s-{epoch:02d}.ckpt' % config.exp.name)
        #checkpoint_dir = os.path.dirname(checkpoint_filepath)
        ## latest = tf.train.latest_checkpoint(checkpoint_dir)
        ##checkpoint_dir = os.path.normpath('./NN')

        #model = load_model(checkpoint_dir)

        if 'trend' in config.model_data.stack_types :
            number_stacks = 6 
            curent_stack = 0 
            outputs_layers_forecast = []
            outputs_layers_backcast = []
            while curent_stack <  number_stacks:                          
                outputs_layers_forecast.append(model.get_layer(str(curent_stack)+'/0/trend/forecast-block' ).output) 
                if curent_stack < number_stacks-1:
                    outputs_layers_backcast.append(model.get_layer(str(curent_stack)+'/0/trend/backcast-block' ).output)
                curent_stack+=1


                outputs_layers_forecast.append(model.get_layer(str(curent_stack)+'/0/seasonality/forecast-block' ).output) 
                if curent_stack < number_stacks-1:
                    outputs_layers_backcast.append(model.get_layer(str(curent_stack)+'/0/seasonality/backcast-block' ).output)
                    
                curent_stack+=1

            partial_model = Model(inputs = [model.get_layer('input_1').input,model.get_layer('input_2').input ,model.get_layer('input_3').input, model.get_layer('input_4').input  ] ,
                                 #outputs = [model.layers[-4].output,model.layers[-3].output]  )
                                 outputs = [outputs_layers_forecast, outputs_layers_backcast] )

                                 
            #print(partial_model.summary())
            # plot_model(partial_model, show_shapes=True,to_file='partial.png',     show_dtype=True,    show_layer_names=True)



        ensemble_model = True
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
            print('predictions shape',predictions.shape)
            #predictions= model.predict( [ data_loader.get_train_data()[0][0] , data_loader.get_train_data()[0] [1] , data_loader.get_train_data()[0] [2]]  )
            for ind in range(trainX[0].shape[0]): #loop all examples 
                x=[] 
                for input_item  in trainX: ## for each input kind in the input list 
                    inp_shape= [1]
                    for sh in input_item[ind].shape :
                        inp_shape.append (sh) 
                    #print('inp_shape:',inp_shape)
                    x.append(input_item[ind].reshape(inp_shape) )
            
                predictions[ind] = model.predict(x)[0,:,0] ##  outputshpe will be [1,27,1]


            #predictions= model.predict( data_loader.get_train_data()[0]  )



            predictions.shape    =  predictions.shape[0], predictions.shape[1]
            #if data normlized apply the inverse 
            # if 0 not normlize, 1 normalize 
            if config.model_data.normlize == 1: 
                #de normlize the predictions   
                predictions = data_loader.denormilse_data(data = predictions, fitted_scalar = data_loader.scalars )
                actuals = data_loader.denormilse_data(data = data_loader.get_train_data()[1], fitted_scalar = data_loader.scalars )


                print('after de scale ')
                print('predictions shape',predictions.shape)
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
                    #print('inp_shape:',inp_shape)
                    x.append(input_item[ind].reshape(inp_shape) )
            
                predictions[ind] = model.predict(x)[0,:,0] ##  outputshpe will be [1,27,1]




            #predictions= model.predict( data_loader.get_test_data()[0]  ) 

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
        if ensemble_model :
            print('start ensemble part ')
            pca_comp = config.model_data.pca_comp
            correction_power_only = config.model_data.correction_power_only  ##use only power to train the residual model 
            #model = model.model
            ##build ensemble models 
            # ensemble_n = 4
            gbnn= [] 
            # for n in range(1,ensemble_n ):
            for n in [1]:
                #information for saving results 
                if config. model_data.descriptive_data:
                    details='_DS_ens_'+str(n) +'_pca_'+str(pca_comp).replace('.','')+'_Pfeat_'+str(config.model_data.use_power)+'_Wfeat_'+str(config.model_data.use_weather)+'_onlyP_'+str(correction_power_only)+windowing
                else:
                    details='_ens_'+str(n) +'_pca_'+str(pca_comp).replace('.','')+'_Pfeat_'+str(config.model_data.use_power)+'_Wfeat_'+str(config.model_data.use_weather)+'_onlyP_'+str(correction_power_only)+windowing
                # trainX =  [ data_loader.get_train_data()[0][0] , data_loader.get_train_data()[0] [2]]
                
                trainX =   data_loader.get_train_data()[0]
                trainy =  data_loader.get_train_data()[1]

                ensemble_models  =  get_ensemble( trainX, trainy,model = model ,pca_comp=pca_comp, n=n,correction_power_only =correction_power_only )
                print('----------------')
                ensmble_predictions= np.zeros( shape=(len(ensemble_models),data_loader.get_train_data()[1].shape[0], data_loader.get_train_data()[1].shape[1] ))
                print('ensmble_predictions',ensmble_predictions.shape)
                ##Evaluate training data
                #'''
                for i , ensemble_model in enumerate(ensemble_models): 
                    if i ==0 : 
                        continue
                        #ensmble_predictions[i] = ensemble_model.predict(  trainX)  [:,:,0]
                    
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


                        #ensmble_predictions[i] = ensemble_model.predict( trainX_2D) 
            
            
                #predictions = ensmble_predictions.sum(axis=0)
                #print( 'predictions:',predictions.shape)

                #if data normlized apply the inverse 
                # if 0 not normlize, 1 normalize 
                '''
                if config.model_data.normlize == 1: 
                    #de normlize the predictions   
                    predictions = data_loader.denormilse_data(data = predictions, fitted_scalar = data_loader.scalars )
                    actuals = data_loader.denormilse_data(data = data_loader.get_train_data()[1], fitted_scalar = data_loader.scalars )
                else:
                    actuals =  data_loader.get_train_data()[1]
                train_res = evaluater .computeAccuracy(actData=actuals, predData= predictions ,maxCapacity=data_loader.maxCapacity,details=details+'_training_No_'+str(iter_no)+exp_description+'_'+str(config.exp.seed_value)+windowing)
                '''


                ##Evaluate testing data
                ensmble_predictions= np.zeros( shape=(len(ensemble_models),data_loader.get_test_data()[1].shape[0], data_loader.get_test_data()[1].shape[1] ))
                interpretable_predictions = []


                testX =  data_loader.get_test_data()[0] 
                testy =  data_loader.get_test_data()[1]

                ## by example because of no memory 
                for test_index in range(testX[0].shape[0]):
                    x =[] 
                    
                    for input_item  in testX:

                        #print('input_item[test_index]:',input_item[test_index].shape)
                        inp_shape= [1]
                        for sh in input_item[test_index].shape :
                            inp_shape.append (sh) 
                        #print('inp_shape:',inp_shape)
                        x.append(input_item[test_index].reshape(inp_shape) )
             
                    for i , ensemble_model in enumerate(ensemble_models): 
                        if i ==0 : 
                            ensmble_predictions[i][test_index] = ensemble_model.predict( x )[:,:,0]
                        
                            if 'trend' in config.model_data.stack_types :
                                temp= partial_model.predict(  x )
                                temp= temp [0] #forecast_interpret
                                #temp_back=temp[1] #backcast

                                #print("len temp",len(temp))
                                if len(interpretable_predictions ) == 0 :
                                    interpretable_predictions = temp
                                else: 
                                    interpretable_predictions = np.concatenate((interpretable_predictions, temp),axis=1)
                            # interpretable_prediction = np.array(partial_model.predict(  np.expand_dims(x_test[0:3], axis=0)   ))

                            # interpretable_prediction.shape = interpretable_prediction.shape[0],interpretable_prediction.shape[2]
                            # print( pd.dataframe( np.array(interpretable_prediction)) ) 
                            #plt.plot(interpretable_predictions[0].flatten(),label='trend' )
                            #plt.plot(interpretable_predictions[1].flatten(),label='seasonality')
                            #plt.plot(interpretable_predictions[0].flatten()+interpretable_predictions[1].flatten(),label = 'prediction')
                            #plt.plot(testy.flatten(),label = 'actual')
                            #plt.legend()
                            #plt.show()



                        
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

                            #pred_ = ensemble_model.predict( x)
                            #print("pred_:",pred_.shape)
                            ensmble_predictions[i][test_index] = ensemble_model.predict( testX_2D)[0]
                #print("interpretable_predictions",interpretable_predictions.shape)

            
                predictions = ensmble_predictions.sum(axis=0)
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
            
           
                if 'trend' in config.model_data.stack_types :
                    print("actuals.flatten().shape:",actuals.flatten().shape)
                    print(" interpretable_predictions[com_i].flatten()", interpretable_predictions[0].flatten().shape)
                    #save interpretable_predictions and residau
                    interpret_dic = {'actuals': actuals.flatten()}  
                    backast_dic = {}      
                    for com_i , component in enumerate( interpretable_predictions):
                        #if com_i < number_stacks : 
                        #    interpret_dic['forecast_comp_'+str(com_i)] = interpretable_predictions[com_i].flatten()
                        #else:
                        #    backast_dic['backcast_comp_'+str(com_i)] = interpretable_predictions[com_i].flatten()

                        interpret_dic['forecast_comp_'+str(com_i)] = interpretable_predictions[com_i].flatten()
    
                    print('interpretable_predictions:',interpretable_predictions.shape)
                    interpretable_predictions_df = pd.DataFrame(  interpret_dic) 
                    interpretable_results_folder = 'interpretation'
                    if not os.path.exists(interpretable_results_folder):
                        os.makedirs(interpretable_results_folder)
                    interpretable_predictions_df.to_csv(interpretable_results_folder + '/interpret_' + config.dataset_file.siteName+ "_lag_" + str(config.model_data.window )+  "_"+config.exp.name.split('_')[-1].upper()+details +'_testing_No_'+str(iter_no)+exp_description+'_'+str(config.exp.seed_value)+windowing+'.csv' )
                    
                    interpretable_backast_df = pd.DataFrame(  backast_dic) 
                    backcast_results_folder = 'backcast'
                    if not os.path.exists(backcast_results_folder):
                        os.makedirs(backcast_results_folder)
                    interpretable_backast_df.to_csv(backcast_results_folder + '/interpret_' + config.dataset_file.siteName+ "_lag_" + str(config.model_data.window )+  "_"+config.exp.name.split('_')[-1].upper()+details +'_testing_No_'+str(iter_no)+exp_description+'_'+str(config.exp.seed_value)+windowing+'.csv' )
                   

                ##save the residuals predictions 
                dic_res = {'actuals': actuals.flatten()}
                for i in range(1,ensmble_predictions.shape[0]):
                    dic_res['res_'+str(i)]= ensmble_predictions[i].flatten()
                    print(dic_res['res_'+str(i)].shape)
            
                residuals_predictions_df = pd.DataFrame( dic_res ) 
            
                residuals_predictions_folder = 'residuals'
                if not os.path.exists(residuals_predictions_folder):
                    os.makedirs(residuals_predictions_folder)
                residuals_predictions_df.to_csv(residuals_predictions_folder + '/residuals_' + config.dataset_file.siteName+ "_lag_" + str(config.model_data.window )+  "_"+config.exp.name.split('_')[-1].upper()+details +'_testing_No_'+str(iter_no)+exp_description+'_'+str(config.exp.seed_value)+windowing+'.csv' )
            

 

                res_dec ['GB'+model_name+'_'+str(n)]= { 'train': train_res , 'test':  test_res}



        


                gbnn.append(res_dec['GB'+model_name+'_'+str(n)]['test'][-1]['MRE'].values[-1])
    

                 

        print(res_dec)
    #exit()

















if __name__ == '__main__':
    main()
    

    