
from warnings import filters
from base.base_model import BaseModel
 

import pandas as pd
import os
import random
import tensorflow.compat.v1 as tf
import numpy as np
from keras.layers import Input, LeakyReLU,Lambda,Subtract,Add, Conv2D,Dense,Flatten,Concatenate,Reshape,Dropout,Conv1D,BatchNormalization 
from keras import initializers, Model
 
from keras import backend as K





class MF_NBEA(BaseModel):
    _BACKCAST = 'backcast'
    _FORECAST = 'forecast'


    def __init__(self, config , experiment, nb_blocks_per_stack=1,thetas_dim=4, nbeats_units=128 ,nb_harmonics = 8  ):
        
        self.experiment = experiment

        super(MF_NBEA, self).__init__(config)
        self.config = config
        ''' seed setings '''
        # Set seed value
        seed_value = self.config.exp.seed_value
        os.environ['PYTHONHASHSEED']=str(seed_value)
        random.seed(seed_value)
        np.random.seed(seed_value)
        tf.random.set_random_seed(seed_value)

        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)



         
        f2 = self.config.model.f2
        f3 = self.config.model.f3
        kernel_size= self.config.model.kernels
        f4 = self.config.model.f4
        

        ''' OVERLAPPING WINDOW '''
        #aggregated pv 
        ovelapping= self.config.model_data.overlapping_window
        if ovelapping : 
                    
            stepsIn = self.config.model_data.window
            stepsOut= self.config.model_data.horizon
        
        
            inputs1 = Input(shape=(self.config.model_data.window, 1 )) #power 
            #number of weather features wanted 
            #this is the number of features learned by the autoencoder 
            n_wanted_features = self.config.model.maps_coding_size# len(self.config.features)+1

            #if use all the features of all the locations 
            #maps representaion of exogenous data 
            if  not self.config.model_data.model_power_only :   ##the weather data is mapped if used as inputs otherwise ignored 
                inputs2 = Input(shape=( self.config.model_data.window, 1,1,n_wanted_features   )) #weather maps
            else: 
                #number of weather features wanted 
                n_wanted_features =len( self.config.features)
                optimal_map_size = int(np.ceil(np.sqrt(self.config.model_data.locations_n))) 
                inputs2 = Input(shape=(n_wanted_features, self.config.model_data.window, optimal_map_size,optimal_map_size   )) #weather maps

            #covariate representaion of exogenous data 
            inputs3 = Input(shape=( self.config.model_data.window, int(self.config.model_data.locations_n) * len(self.config.features)   )) #weather cov
            input4 =  Input(shape=(self.config.model_data.window, 5 )) #calendar 

        else: 

            
            ''' SLIDING WINDOW '''
                    
            stepsIn = self.config.dataset_file.samplePerDay * self.config.model_data.window
            stepsOut= self.config.dataset_file.samplePerDay
        
        
            #aggregated pv 
            inputs1 = Input(shape=(self.config.model_data.window,self.config.dataset_file.samplePerDay , 1 )) #power 
        
            #maps representaion of exogenous data 
            #number of weather features wanted - this is the number of features learned by the autoencoder 
            n_wanted_features = self.config.model.maps_coding_size 
            inputs2 = Input(shape=( self.config.model_data.window,self.config.dataset_file.samplePerDay, 1,n_wanted_features   )) #weather maps
        
            #covariate representaion of exogenous data 
            inputs3 = Input(shape=( self.config.model_data.window,self.config.dataset_file.samplePerDay, int(self.config.model_data.locations_n) * len(self.config.features)   )) #weather cov

            input4 =  Input(shape=(self.config.model_data.window,self.config.dataset_file.samplePerDay , 1 )) #calendar 



        power = inputs1
        power = BatchNormalization()(power)
        weather = inputs2
        weather = BatchNormalization()(weather)

        ## Temporal Encoders with residual connections 
        ## extract features from power series 
        concat_input_ = power 
        random.seed(seed_value)
        seed_value = [random.randrange(1, 9999 ) for i in range(1)][0]
        concat_input =   Conv2D(filters=f2, kernel_size= kernel_size , padding= "same" ,kernel_initializer=initializers.he_normal(seed=seed_value),  name= 'latent_power_features')(concat_input_) 
        concat_input = LeakyReLU()(concat_input)
        concat_input = BatchNormalization() (concat_input)
        ## residual connection for power 
        concat_input = Concatenate() ([concat_input,concat_input_])
        

        ## extract features from encoded map weather 
        random.seed(seed_value)
        seed_value = [random.randrange(1, 9999 ) for i in range(1)][0]
        weather_ = Conv2D(filters=f3,  kernel_size=kernel_size, padding= "same",kernel_initializer=initializers.he_normal( seed=seed_value), name= 'latent_weather_features')(weather) 
        weather_ = LeakyReLU()(weather_)
        weather_ = BatchNormalization() (weather_)
        seed_value = seed_value+1
        weather_ = Dropout(.15,seed =seed_value ) (weather_)
        ## residual features for encoded map weather 
        weather_ = Concatenate() ([weather_,weather])


        # 
        if  ovelapping :
            s1 =weather_.get_shape()[1]  
            s2 =weather_.get_shape()[2] 
            s3 =weather_.get_shape()[3]
            s4 = weather_.get_shape()[4]
            weather_ = Flatten()(weather_)
            weather_ = Reshape( (s1,s2*s3 *s4) , name = 'reshape_latent_maps')(weather_)
            
        else:    
            #reshape the power to batch, steps,features 
            s1 =concat_input.get_shape()[1]  
            s2 =concat_input.get_shape()[2] 
            s3 =concat_input.get_shape()[3]
            concat_input = Reshape((s1*s2,s3))(concat_input)
            print('weather_.get_shape()',weather_.get_shape())

            s1 =weather_.get_shape()[1]  
            s2 =weather_.get_shape()[2] 
            s3 =weather_.get_shape()[3]
            s4 =weather_.get_shape()[4]
            weather_ = Reshape((s1*s2,s3*s4))(weather_)




        concat_input  = Concatenate( ) ([concat_input,weather_])
        concat_input = Conv1D(filters=f4,  kernel_size=kernel_size, padding= "same",kernel_initializer=initializers.he_normal( seed=seed_value), name= 'reduce_dim_of_power_wether_maps1')(concat_input) 
        # concat_input = Conv1D(filters=f4//2,  kernel_size=kernel_size, padding= "same",kernel_initializer=initializers.he_normal( seed=seed_value), name= 'reduce_dim_of_power_wether_maps2')(concat_input) 
        power_to_cocat = Flatten()(power)
        power_to_cocat = Reshape( (stepsIn,1)) (power_to_cocat)
        concat_input = Concatenate()([concat_input, power_to_cocat ])

        seed_value = seed_value+1
        layer = Dropout(.3, seed = seed_value)(concat_input ) 
 
        layer = Flatten()(layer)
        generated_seq = Dense(stepsIn, activation='relu' ,  kernel_initializer=initializers.HeNormal(seed=seed_value) )(layer)
        generated_seq = Reshape((stepsIn,1),name= 'input_variable' )(generated_seq)

        
        '''
        The following code is from: https://github.com/philipperemy/n-beats with some modifications 
        '''    
        self.stack_types = tuple(self.config.model_data.stack_types)
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.thetas_dim = tuple( [ thetas_dim for _ in range(len(self.stack_types))]) 
        self.units = nbeats_units
        self.share_weights_in_stack = False  if 'generic' in self.stack_types else True  # share_weights_in_stack
        self.backcast_length = stepsIn  #backcast_length
        self.forecast_length = stepsOut #forecast_length
        self.input_dim = generated_seq.get_shape()[-1]# input_dim
        self.input_shape = (self.backcast_length, self.input_dim)

        self.exo_dim  = 0 
        self.exo_shape = (self.backcast_length, self.exo_dim)
        self.output_shape = (self.forecast_length, self.input_dim)
        self.weights = {}
        self.nb_harmonics =  nb_harmonics

        assert len(self.stack_types) == len(self.thetas_dim)

        x = generated_seq #Input(shape=self.input_shape, name='input_variable')
        x_ = {}
        for k in range(self.input_dim):
            x_[k] = Lambda(lambda z: z[..., k])(x)
        e_ = {}



        e = None
        y_ = {}

        for stack_id in range(len(self.stack_types)):
            stack_type = self.stack_types[stack_id]
            nb_poly = self.thetas_dim[stack_id]
            for block_id in range(self.nb_blocks_per_stack):
                backcast, forecast = self.create_block(x_, e_, stack_id, block_id, stack_type, nb_poly)
                for k in range(self.input_dim):
                    x_[k] = Subtract()([x_[k], backcast[k]])
                    if stack_id == 0 and block_id == 0:
                        y_[k] = forecast[k]
                    else:
                        y_[k] = Add(name ='add'+str(stack_id)+'-'+str(block_id))([y_[k], forecast[k]])
        for k in range(self.input_dim):

            y_[k] = Reshape(target_shape=(self.forecast_length, 1))(y_[k])
            x_[k] = Reshape(target_shape=(self.backcast_length, 1))(x_[k])
        if self.input_dim > 1:
            y_ = Concatenate()([y_[ll] for ll in range(self.input_dim)])
            x_ = Concatenate()([x_[ll] for ll in range(self.input_dim)])
        else:
            y_ = y_[0]
            x_ = x_[0]


        n_beats_forecast = Model([inputs1, inputs2,inputs3,input4], y_, name=self._FORECAST)
        n_beats_backcast = Model([inputs1, inputs2,inputs3,input4,x_ ], name=self._BACKCAST)

        self.models = {model.name: model for model in [n_beats_backcast, n_beats_forecast]}
        self.cast_type = self._FORECAST



    def has_exog(self):
        # exo/exog is short for 'exogenous variable', i.e. any input
        # features other than the target time-series itself.
        return self.exo_dim > 0

    @staticmethod
    def load(filepath, custom_objects=None, compile=True):
        from tensorflow.keras.models import load_model
        return load_model(filepath, custom_objects, compile)

    def _r(self, layer_with_weights, stack_id):
        # mechanism to restore weights when block share the same weights.
        # only useful when share_weights_in_stack=True.
        if self.share_weights_in_stack:
            layer_name = layer_with_weights.name.split('/')[-1]
            try:
                reused_weights = self.weights[stack_id][layer_name]
                return reused_weights
            except KeyError:
                pass
            if stack_id not in self.weights:
                self.weights[stack_id] = {}
            self.weights[stack_id][layer_name] = layer_with_weights
        return layer_with_weights

    def create_block(self, x, e, stack_id, block_id, stack_type, nb_poly):
        # register weights (useful when share_weights_in_stack=True)
        def reg(layer):
            return self._r(layer, stack_id)

        # update name (useful when share_weights_in_stack=True)
        def n(layer_name):
            return '/'.join([str(stack_id), str(block_id), stack_type, layer_name])

        backcast_ = {}
        forecast_ = {}
        d1 = reg(Dense(self.units, activation='relu', name=n('d1')))
        d2 = reg(Dense(self.units, activation='relu', name=n('d2')))
        d3 = reg(Dense(self.units, activation='relu', name=n('d3')))
        d4 = reg(Dense(self.units, activation='relu', name=n('d4')))
        if stack_type == 'generic':
            theta_b = reg(Dense(nb_poly, activation='linear', use_bias=False, name=n('theta_b')))
            theta_f = reg(Dense(nb_poly, activation='linear', use_bias=False, name=n('theta_f')))
            backcast = reg(Dense(self.backcast_length, activation='linear', name=n('backcast')))
            forecast = reg(Dense(self.forecast_length, activation='linear', name=n('forecast')))
        elif stack_type == 'trend':
            theta_f = theta_b = reg(Dense(nb_poly, activation='linear', use_bias=False, name=n('theta_f_b')))

            backcast = Lambda(trend_model , name= n('backcast-block'), 
                     arguments={'is_forecast': False, 'backcast_length': self.backcast_length,
                                                      'forecast_length': self.forecast_length})
            forecast = Lambda(trend_model, name= n('forecast-block') ,
                                            arguments={'is_forecast': True, 'backcast_length': self.backcast_length,
                                                      'forecast_length': self.forecast_length})
        else:  # 'seasonality'
            if self.nb_harmonics:
                theta_b = reg(Dense(self.nb_harmonics, activation='linear', use_bias=False, name=n('theta_b')))
            else:
                theta_b = reg(Dense(self.forecast_length, activation='linear', use_bias=False, name=n('theta_b')))
            theta_f = reg(Dense(self.forecast_length, activation='linear', use_bias=False, name=n('theta_f')))
            backcast = Lambda(seasonality_model,
                              name= n('backcast-block'),
                              arguments={'is_forecast': False, 'backcast_length': self.backcast_length,
                                         'forecast_length': self.forecast_length} )
            forecast = Lambda(seasonality_model,name= n('forecast-block') ,
                              arguments={'is_forecast': True, 'backcast_length': self.backcast_length,
                              
                                         'forecast_length': self.forecast_length})
        for k in range(self.input_dim):
            if self.has_exog():
                d0 = Concatenate()([x[k]] + [e[ll] for ll in range(self.exo_dim)])
            else:
                d0 = x[k]
            d1_ = d1(d0)
            d2_ = d2(d1_)
            d3_ = d3(d2_)
            d4_ = d4(d3_)
            theta_f_ = theta_f(d4_)
            theta_b_ = theta_b(d4_)
            backcast_[k] = backcast(theta_b_)
            forecast_[k] = forecast(theta_f_)

        return backcast_, forecast_

    def __getattr__(self, name):
        # https://github.com/faif/python-patterns
        # model.predict() instead of model.n_beats.predict()
        # same for fit(), train_on_batch()...
        attr = getattr(self.models[self._FORECAST], name)

        if not callable(attr):
            return attr

        def wrapper(*args, **kwargs):
            cast_type = self._FORECAST
            if attr.__name__ == 'predict' and 'return_backcast' in kwargs and kwargs['return_backcast']:
                del kwargs['return_backcast']
                cast_type = self._BACKCAST
            return getattr(self.models[cast_type], attr.__name__)(*args, **kwargs)

        return wrapper

        

def linear_space(backcast_length, forecast_length, is_forecast=True):
    ls = K.arange(-float(backcast_length), float(forecast_length), 1) / forecast_length
    return ls[backcast_length:] if is_forecast else K.abs(K.reverse(ls[:backcast_length], axes=0))


def seasonality_model(thetas, backcast_length, forecast_length, is_forecast):
    p = thetas.get_shape().as_list()[-1]
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    t = linear_space(backcast_length, forecast_length, is_forecast=is_forecast)
    s1 = K.stack([K.cos(2 * np.pi * i * t) for i in range(p1)])
    s2 = K.stack([K.sin(2 * np.pi * i * t) for i in range(p2)])
    if p == 1:
        s = s2
    else:
        s = K.concatenate([s1, s2], axis=0)
    s = K.cast(s, np.float32)
    return K.dot(thetas, s)


def trend_model(thetas, backcast_length, forecast_length, is_forecast):
    
    p = thetas.shape[-1]
    print('trend_model:',p) 
    t = linear_space(backcast_length, forecast_length, is_forecast=is_forecast)
    t = K.transpose(K.stack([t ** i for i in range(p)]))
    t = K.cast(t, np.float32)
    return K.dot(thetas, K.transpose(t))

# '''    