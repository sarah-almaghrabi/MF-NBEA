

from base.base_model import BaseModel
import pandas as pd
import os
import random
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import numpy as np
from keras.layers import Input,Cropping3D,  UpSampling3D,Conv3D,MaxPooling3D
from keras import initializers, Model
from tensorflow.keras import backend as K






class AutoEncoder(BaseModel):
    def __init__(self, config , experiment ):
        self.experiment = experiment
        super(AutoEncoder, self).__init__(config)
        self.config = config
        ''' seed setings '''
        seed_value = self.config.exp.seed_value
        os.environ['PYTHONHASHSEED']=str(seed_value)
        random.seed(seed_value)
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)


        self.build_model()


    def build_model(self):

        #Set Tensorflow to grow GPU memory consumption instead of grabbing all of it at once
        K.clear_session()
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True

        tfsess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(tfsess)
        

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



    
        if optimal_map_size == 4 : 
   
            x= input_imag
            x = Conv3D(64, (5, 3, 3), data_format='channels_first', activation='relu', padding='same')(x)
            x = MaxPooling3D((1, 2, 2), data_format='channels_first', padding='same')(x)
            x = Conv3D(32, (5, 3, 3), data_format='channels_first', activation='relu', padding='same')(x)
            x = MaxPooling3D((1, 2, 2), data_format='channels_first', padding='same')(x)
            x = Conv3D(self.config.model.maps_coding_size, (5, 3, 3), data_format='channels_first', activation='relu', padding='same')(x)
            encoded = MaxPooling3D((1, 2, 2), data_format='channels_first', padding='same', name='encoder')(x)
        
            x = Conv3D(self.config.model.maps_coding_size, (5, 3, 3), data_format='channels_first', activation='relu', padding='same')(encoded)
            x = UpSampling3D((1, 1, 1), data_format='channels_first')(x)
            x = Conv3D(32, (5, 3, 3), data_format='channels_first', activation='relu', padding='same')(x)
            x = UpSampling3D((1, 2, 2), data_format='channels_first')(x)
            x = Conv3D(64, (5, 3, 3), data_format='channels_first', activation='relu', padding='same')(x)
            x = UpSampling3D((1, 2, 2), data_format='channels_first')(x)
            decoded_out = Conv3D(n_wanted_features, (5, 3, 3), data_format='channels_first', activation='relu', padding='same')(x)
        
            
        elif optimal_map_size == 5:
            print("optimal_map_size",optimal_map_size)
            x= input_imag
            x = Conv3D(64, (5, 3, 3), data_format='channels_first', activation='relu',kernel_initializer=initializers.he_normal(), padding='same')(x)
            x = MaxPooling3D((1, 2, 2), data_format='channels_first', padding='same')(x)
            x = Conv3D(32, (5, 3, 3), data_format='channels_first', activation='relu', padding='same',kernel_initializer=initializers.he_normal())(x)
            x = MaxPooling3D((1, 2, 2), data_format='channels_first', padding='same')(x)
            x = Conv3D(self.config.model.maps_coding_size, (5, 3, 3), data_format='channels_first', activation='relu', padding='same',kernel_initializer=initializers.he_normal())(x)
            encoded = MaxPooling3D((1, 2, 2), data_format='channels_first', padding='same', name='encoder')(x)
        
            x = UpSampling3D((1, 2, 2), data_format='channels_first' )(encoded)
            x = Conv3D(self.config.model.maps_coding_size, (5, 3, 3), data_format='channels_first', activation='relu', kernel_initializer=initializers.he_normal(),padding='same')(x)
            x = UpSampling3D((1, 2, 2), data_format='channels_first' )(x)
            x = Cropping3D(cropping=((0, 0), (1, 0), (1, 0)), data_format='channels_first')(x)  
            x = Conv3D(32, (5, 3, 3), data_format='channels_first', activation='relu', padding='same',kernel_initializer=initializers.he_normal())(x)
            x = UpSampling3D((1, 2, 2), data_format='channels_first' )(x)
            x = Cropping3D(cropping=((0, 0), (1, 0), (1, 0)), data_format='channels_first')(x)  
        
        
            x = Conv3D(64, (5, 3, 3), data_format='channels_first', activation='relu', padding='same',kernel_initializer=initializers.he_normal())(x)
            decoded_out = Conv3D(n_wanted_features, (5, 3, 3), data_format='channels_first', activation='relu', padding='same')(x)

         
        self.model = Model(input_imag, decoded_out)
        self.model.compile(optimizer='adam', loss='mse')

        



        
