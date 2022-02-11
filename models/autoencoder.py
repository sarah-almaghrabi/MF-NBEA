

from base.base_model import BaseModel
from keras.models import Sequential
 

import pandas as pd
import os
import random
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
#tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)

import numpy as np
from keras.layers import Input,Cropping3D,ZeroPadding3D,  UpSampling3D, Conv3DTranspose,Conv3D,MaxPooling3D, Conv2D,Dense,Flatten,Concatenate,AveragePooling2D,TimeDistributed, RepeatVector,Reshape,Dropout,Activation  ,ConvLSTM2D,Conv1D,MaxPooling2D,LSTM,BatchNormalization, GlobalAveragePooling2D,GlobalMaxPooling2D, AveragePooling1D
from keras import initializers, Model
from tensorflow.keras.regularizers import l2

from tensorflow.keras import backend as K
from keras import layers as L

import random
import numpy as np
from keras.utils.vis_utils import plot_model
from tensorflow.keras.constraints import non_neg, Constraint

from models_grid import Attention




class AutoEncoder(BaseModel):
    def __init__(self, config , experiment ):
        self.experiment = experiment
        super(AutoEncoder, self).__init__(config)
        self.config = config
        ''' seed setings '''
        # Set seed value
        seed_value = self.config.exp.seed_value
        os.environ['PYTHONHASHSEED']=str(seed_value)

        # Set `python` built-in pseudo-random generator at a fixed value
        random.seed(seed_value)

        # Set `numpy` pseudo-random generator at a fixed value
        np.random.seed(seed_value)

        # Set `tensorflow` pseudo-random generator at a fixed value
        tf.random.set_seed(seed_value)


        self.build_model()


    def build_model(self):

        #Set Tensorflow to grow GPU memory consumption instead of grabbing all of it at once
        K.clear_session()
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True

        tfsess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(tfsess)

        # hp_n_layers = hb.Int('layers', min_value=0, max_value=2, step=1)
        # hp_units = hb.Choice(name='units' ,values=[32,64,128])#,64,128,256,512])
        # hp_dropout = hb.Choice(name='droput' ,values=[0.0,0.2])
        
        

        #number of weather features wanted 
        n_wanted_features =len( self.config.features)
        optimal_map_size = int(np.ceil(np.sqrt(self.config.model_data.locations_n))) 

        ovelapping= self.config.model_data.overlapping_window
        if ovelapping :                    
            stepsIn = self.config.model_data.window
            stepsOut= self.config.model_data.horizon
            input_imag = Input(shape=(n_wanted_features , self.config.model_data.window, optimal_map_size,optimal_map_size  ) , name = 'map_inputs') #weather 

        else:          
            stepsIn = self.config.dataset_file.samplePerDay * self.config.model_data.window
            stepsOut= self.config.dataset_file.samplePerDay
            input_imag = Input(shape=(n_wanted_features , self.config.model_data.window*self.config.dataset_file.samplePerDay, optimal_map_size,optimal_map_size  ) , name = 'map_inputs') #weather 



        print('input_imag',input_imag.get_shape())

        #if use all the features of all the locations 
    
        if optimal_map_size == 4 : 
   
            x= BatchNormalization()(input_imag)
            x= input_imag
            x = Conv3D(64, (5, 3, 3), data_format='channels_first', activation='relu', padding='same')(x)
            x = MaxPooling3D((1, 2, 2), data_format='channels_first', padding='same')(x)
            #x= BatchNormalization()(x)
            x = Conv3D(32, (5, 3, 3), data_format='channels_first', activation='relu', padding='same')(x)
            x = MaxPooling3D((1, 2, 2), data_format='channels_first', padding='same')(x)

            x= BatchNormalization()(x)
            x = Conv3D(self.config.model.maps_coding_size, (5, 3, 3), data_format='channels_first', activation='relu', padding='same')(x)
            encoded = MaxPooling3D((1, 2, 2), data_format='channels_first', padding='same', name='encoder')(x)
        
            #x= BatchNormalization()(encoded)
            x = Conv3D(self.config.model.maps_coding_size, (5, 3, 3), data_format='channels_first', activation='relu', padding='same')(encoded)
            x = UpSampling3D((1, 1, 1), data_format='channels_first')(x)
            x= BatchNormalization()(x)
            x = Conv3D(32, (5, 3, 3), data_format='channels_first', activation='relu', padding='same')(x)
            x = UpSampling3D((1, 2, 2), data_format='channels_first')(x)
            #x= BatchNormalization()(x)
            x = Conv3D(64, (5, 3, 3), data_format='channels_first', activation='relu', padding='same')(x)
            x = UpSampling3D((1, 2, 2), data_format='channels_first')(x)
            #x= BatchNormalization()(x)
            decoded_out = Conv3D(n_wanted_features, (5, 3, 3), data_format='channels_first', activation='relu', padding='same')(x)
        
            
        elif optimal_map_size == 5:
            print("optimal_map_size",optimal_map_size)
            x= input_imag
            x = Conv3D(64, (5, 3, 3), data_format='channels_first', activation='relu',kernel_initializer=initializers.he_normal(), padding='same')(x)


            x = MaxPooling3D((1, 2, 2), data_format='channels_first', padding='same')(x)
            #x= BatchNormalization()(x)
            x = Conv3D(32, (5, 3, 3), data_format='channels_first', activation='relu', padding='same',kernel_initializer=initializers.he_normal())(x)
            x = MaxPooling3D((1, 2, 2), data_format='channels_first', padding='same')(x)

            x= BatchNormalization()(x)
            x = Conv3D(self.config.model.maps_coding_size, (5, 3, 3), data_format='channels_first', activation='relu', padding='same',kernel_initializer=initializers.he_normal())(x)
            encoded = MaxPooling3D((1, 2, 2), data_format='channels_first', padding='same', name='encoder')(x)
        
            #x= BatchNormalization()(encoded)
            x = UpSampling3D((1, 2, 2), data_format='channels_first' )(encoded)
            x= BatchNormalization()(x)
            x = Conv3D(self.config.model.maps_coding_size, (5, 3, 3), data_format='channels_first', activation='relu', kernel_initializer=initializers.he_normal(),padding='same')(x)
            x = UpSampling3D((1, 2, 2), data_format='channels_first' )(x)
            x = Cropping3D(cropping=((0, 0), (1, 0), (1, 0)), data_format='channels_first')(x)  
            x = Conv3D(32, (5, 3, 3), data_format='channels_first', activation='relu', padding='same',kernel_initializer=initializers.he_normal())(x)
            x = UpSampling3D((1, 2, 2), data_format='channels_first' )(x)
            x = Cropping3D(cropping=((0, 0), (1, 0), (1, 0)), data_format='channels_first')(x)  
        
        
            #x= BatchNormalization()(x)
            x = Conv3D(64, (5, 3, 3), data_format='channels_first', activation='relu', padding='same',kernel_initializer=initializers.he_normal())(x)
            #x= BatchNormalization()(x)
            decoded_out = Conv3D(n_wanted_features, (5, 3, 3), data_format='channels_first', activation='relu', padding='same')(x)

        #optimal_map_size =5
        #dim = optimal_map_size   

        
        #input_imag = Input(shape=(3, 81, optimal_map_size, optimal_map_size))
        #x = input_imag #ZeroPadding3D(((0, 0), (1, 1), (1, 1)) , data_format='channels_first') (input_imag)
        #x = Conv3D(16, (5, dim, dim), data_format='channels_first', activation='relu',padding='same')(x)
        #x = MaxPooling3D((1, 2, 2), data_format='channels_first',padding='same' )(x)
        #x = Conv3D(8, (5, dim, dim), data_format='channels_first', activation='relu',padding='same')(x)
        #x = MaxPooling3D((1, 2, 2), data_format='channels_first',padding='same')(x)
        #x = Conv3D(4, (5, dim, dim), data_format='channels_first', activation='relu',padding='same')(x)
        #encoded = MaxPooling3D((1, 1, 1), data_format='channels_first', padding='same', name='encoder')(x)
        
        #padd =2
        #x = Conv3DTranspose(4, (5, dim, dim), strides=(1,padd+1,padd+1),output_padding = (0,padd,padd) ,data_format='channels_first', padding='same',activation='relu')(encoded)
        ##x = Conv3DTranspose(4, (5, dim, dim), strides=(1,2,2),data_format='channels_first', padding='same',activation='relu')(encoded)
        #x = Conv3DTranspose(8, (5, dim, dim), data_format='channels_first', padding='same',activation='relu')(x)
        #x = Conv3DTranspose(16, (5, dim, dim) ,strides=(1,padd+1,padd+1),data_format='channels_first', padding='same',activation='relu')(x)
        #x = Conv3DTranspose(3, (5, dim, dim), strides=(1,padd+1,padd+1), data_format='channels_first', padding='same',activation='relu')(x)
        ##x = Conv3D(8, (5, dim, dim), data_format='channels_first', padding='same',activation='relu')(x)
        ##x = Conv3D(16, (5, dim, dim), data_format='channels_first', activation='relu')(x)
        ##x = UpSampling3D((3, 2, 2), data_format='channels_first')(x)
        #decoded_out = x#Conv3D(3, (5, dim, dim), data_format='channels_first', activation='relu')(x)
 

         
        self.model = Model(input_imag, decoded_out)
        self.model.compile(optimizer='Nadam', loss='mse')

        



        
        # self.model.compile(
        #     loss= hb.Choice(name='loss' ,values=['huber'])  ,
        #     optimizer=self.config.model.optimizer,
        #     metrics=['mae']
        # )
        # self.experiment.log_parameters(reg)
        print(self.model.summary(expand_nested=True))
        plot_model(self.model, to_file='model_auto.png', show_shapes=True, show_layer_names=True)
        # tfsess.close()
        #return self.model
        

        
