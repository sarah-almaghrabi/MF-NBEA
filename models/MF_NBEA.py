
from base.base_model import BaseModel
 

import pandas as pd
import os
import random
import tensorflow.compat.v1 as tf
import numpy as np
from keras.layers import Input, LeakyReLU,Lambda,Subtract,Add, Conv2D,Dense,Flatten,Concatenate,Reshape,Dropout,Conv1D,BatchNormalization 
from keras import initializers, Model
 
from tensorflow.keras import backend as K




class MF_NBEA(BaseModel):
    _BACKCAST = 'backcast'
    _FORECAST = 'forecast'


    def __init__(self, config , experiment,units ,in_dim,out_dim , nb_blocks_per_stack=1,thetas_dim=4, nbeats_units=128 ,nb_harmonics = 8  ):
        model_power_only = config.model_data.model_power_only
        weather_maps = config.model_data.model_weather_maps
        weather_covariate = config.model_data.model_weather_covariate
        calendar_cov = config.model_data.model_calendar_cov
        

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



        



        def custom_weights( shape,dtype=None):
                    #kernel = pd.read_csv('/Users/sarahalmaghrabi/OneDrive - RMIT University/experiments/Multivariate_experiments/DL_with_autocorrection/ridge_coef.csv', index_col=0).values
                    kernel = pd.read_csv('ridge_coef_lag_'+str(self.config.model_data.window)+'_' +self.config.dataset_file.siteName+'.csv').values
                    # change value here
                    kernel = K.variable(kernel)
                    return kernel

        def add_dim(data):
            """Reshape the context vectors to 3D vector"""
            return K.reshape(x=data, shape=(K.shape(data)[0], K.shape(data)[1], 1))

         
        filters = units 
        kernel_size=5 
        
        
        def flatten_first_dimentions(tensor_x, dim_to_flat):
            tensor_dim_len = len(tensor_x.get_shape())
            dims = np.zeros( shape= (tensor_dim_len)) #skip the batches dim
            for dim in range(1,tensor_dim_len): #skip the batches dim
                dims[dim] = tensor_x.get_shape()[dim]
            print(dims)
            new_shape = [ int(np.product(dims[dim_to_flat]))   ] 
            for rest_dim in dims[dim_to_flat[-1]+1:]:
                new_shape.append( int(rest_dim))
            print('new_shape:',new_shape)
            tensor_x= Reshape(new_shape)(tensor_x)
            return tensor_x

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

            # if use mean values of features 
            # inputs2 = Input(shape=(self.config.model_data.window,self.config.dataset_file.samplePerDay, n_wanted_features )) #weather 
            # inputs2 = Input(shape=(self.config.model_data.window,self.config.dataset_file.samplePerDay, 1 )) #weather 

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



        input_list = []
        power = inputs1
        power = BatchNormalization()(power)
        weather = inputs2
        weather = BatchNormalization()(weather)


        # concat_input =Concatenate( ) ([inputs2,inputs1])

        ## Temporal Encoders 
        ## extract features from power series 
        concat_input_ = power #weather #Concatenate( ) ([weather,power])
        random.seed(seed_value)
        seed_value = [random.randrange(1, 9999 ) for i in range(1)][0]
        concat_input =   Conv2D(filters=filters, kernel_size= kernel_size , padding= "same" ,kernel_initializer=initializers.he_normal(seed=seed_value),  name= 'latent_power_features')(concat_input_) 
        concat_input = LeakyReLU()(concat_input)
        concat_input = BatchNormalization() (concat_input)
        ## residual connection for power 
        concat_input = Concatenate() ([concat_input,concat_input_])
        
        ## extract features from calendar series 
        random.seed(seed_value)
        seed_value = [random.randrange(1, 9999 ) for i in range(1)][0]
        calendar =  Conv1D(filters=filters, kernel_size= kernel_size , padding= "same" ,kernel_initializer=initializers.he_normal(seed=seed_value ), name= 'latent_calendar_features')(input4)
        calendar =  LeakyReLU()(calendar)
        calendar = BatchNormalization() (calendar)
        ## residual connection for calendar
        calendar = Concatenate() ([calendar,input4])


        ## extract features from encoded map weather 
        random.seed(seed_value)
        seed_value = [random.randrange(1, 9999 ) for i in range(1)][0]
        weather_ = Conv2D(filters=filters,  kernel_size=kernel_size, padding= "same",kernel_initializer=initializers.he_normal( seed=seed_value), name= 'latent_weather_features')(weather) 
        weather_ = LeakyReLU()(weather_)
        weather_ = BatchNormalization() (weather_)
        seed_value = seed_value+1
        weather_ = Dropout(.15,seed =seed_value ) (weather_)
        ## residual features for encoded map weather 
        weather_ = Concatenate() ([weather_,weather])
        print('\n\n\n\n\n',weather_.get_shape())


        # 
        if  ovelapping :
            s1 =weather_.get_shape()[1]  
            s2 =weather_.get_shape()[2] 
            s3 =weather_.get_shape()[3]
            s4 = weather_.get_shape()[4]
            weather_ = Flatten()(weather_)
            weather_ = Reshape( (s1,s2*s3 *s4) , name = 'reshape_latent_maps')(weather_)
        else:
            s1 =weather_.get_shape()[1]  
            s2 =weather_.get_shape()[2] 
            s3 =weather_.get_shape()[3]
            s4 = weather_.get_shape()[4]
            weather_ = Flatten()(weather_)
            weather_ = Reshape( (s1,s2,s3 *s4) ,name = 'reshape_latent_maps')(weather_)
        




        if not ovelapping : 
            
            #reshape the power to batch, steps,features 
            s1 =concat_input.get_shape()[1]  
            s2 =concat_input.get_shape()[2] 
            s3 =concat_input.get_shape()[3]
            concat_input = Reshape((s1*s2,s3))(concat_input)

        #reshape the encoded maps to batch, steps,features  (if not use the latent features betwen days and lags )
        # s1 =weather.get_shape()[1]  
        # s2 =weather.get_shape()[2] 
        # s3 =weather.get_shape()[3]
        # s4 = weather.get_shape()[4]

        #reshape the latent features of encoded maps to batch, steps,features  
        print('weather_.get_shape()',weather_.get_shape())
        

        if not ovelapping :  
            weather_ =  flatten_first_dimentions(tensor_x=weather_, dim_to_flat=[1,2])

        print('weather_.get_shape()',weather_.get_shape())

        #random.seed(seed_value)
        #seed_value = [random.randrange(1, 9999 ) for i in range(1)][0]        
        #if  ovelapping:
        #    weather_ = Conv1D(filters=filters, kernel_regularizer=l2(), kernel_size=kernel_size , padding= "same",kernel_initializer=initializers.he_normal(seed=seed_value), name= 'latent_weather_features2d')(weather_) 
        #else:
        #    weather_ = Conv2D(filters=filters, kernel_regularizer=l2(), kernel_size=kernel_size , padding= "same",kernel_initializer=initializers.he_normal(seed=seed_value), name= 'latent_weather_features2d')(weather_) 
                   

        #weather_ = LeakyReLU()(weather_)
        #seed_value = seed_value+1
        #weather_ = Dropout(.15, seed=seed_value) (weather_)
        #weather_ = BatchNormalization() (weather_)

        
        
        
        #random.seed(seed_value)
        #seed_value = [random.randrange(1, 9999 ) for i in range(1)][0] 
        
        #weather_cov = inputs3
        
        #if  ovelapping:
        #    weather_cov =   Conv1D(filters=filters, kernel_size= kernel_size , padding= "same" ,kernel_initializer=initializers.he_normal(seed=seed_value), name= 'latent_weather_cov_features')(weather_cov) 
        #else:
        #    weather_cov =   Conv2D(filters=filters, kernel_size= kernel_size , padding= "same" ,kernel_initializer=initializers.he_normal(seed=seed_value), name= 'latent_weather_cov_features')(weather_cov) 
        #weather_cov = LeakyReLU()(weather_cov)
        #weather_cov = BatchNormalization() (weather_cov)
        #weather_cov = Concatenate() ([weather_cov,inputs3])
        
        if weather_maps and not weather_covariate :
            print('concat_input > ',concat_input.get_shape())  #power
            print('weather_ > ',weather_.get_shape()) #maped 
            
            concat_input  = Concatenate( ) ([concat_input,weather_])
            print('concat_input > ',concat_input.get_shape())  #power

            concat_input = Conv1D(filters=filters,  kernel_size=kernel_size, padding= "same",kernel_initializer=initializers.he_normal( seed=seed_value), name= 'reduce_dim_of_power_wether_maps1')(concat_input) 
            print('concat_input > ',concat_input.get_shape())  #power

            concat_input = Conv1D(filters=filters//2,  kernel_size=kernel_size, padding= "same",kernel_initializer=initializers.he_normal( seed=seed_value), name= 'reduce_dim_of_power_wether_maps2')(concat_input) 

            #concat_input  = Flatten( ) (concat_input)


            print(concat_input.get_shape())

        elif weather_covariate and not weather_maps: 
            concat_input  = Concatenate( ) ([concat_input,weather_cov])
        elif weather_maps and weather_covariate and not calendar_cov: 
            concat_input  = Concatenate( ) ([concat_input,weather_cov,weather_])
        elif weather_maps and weather_covariate and calendar_cov:
            concat_input  = Concatenate( ) ([concat_input,weather_cov,weather_,calendar])

        #print('concat_input >>',concat_input.get_shape())

        ##flaten the last dimention and feeed it to a conv1D


        # concat_input = Concatenate(axis=1) ([concat_input,concat_input_])
        # in1 = Reshape( (self.config.model_data.window*self.config.dataset_file.samplePerDay , 1 )) (inputs1 ) 
        # in2= Reshape( (self.config.model_data.window*self.config.dataset_file.samplePerDay , 12 )) (inputs2 ) 

        # latentdim = hb.Choice(name='latentdim' ,values=[5, 4,3,2,10,15])
        #produce latent features of power 


        input_list.append(power)

        input_list.append(weather)

        use_power = self.config.model_data.use_power
        use_weather = self.config.model_data.use_weather
        
        if use_power and  not use_weather : 
            full_inputs = power
        elif not use_power and   use_weather : 
            full_inputs = weather
        elif use_power and use_weather: 
            full_inputs = concat_input 
        else: 
            print('sorry features to use are not decided !')
            exit()
        
        print('full_inputs',full_inputs.get_shape())
        seed_value = seed_value+1
        layer = Dropout(.3, seed = seed_value)(full_inputs )
 
        unit_fc = stepsIn 
        #to use only power with NBEATS
        if model_power_only:
            exxxo_dim = 0
            if  ovelapping:
                layer = inputs1
            else:
                s1 =inputs1.get_shape()[1]  
                s2 =inputs1.get_shape()[2] 
                s3 =inputs1.get_shape()[3]
                layer = Reshape((s1*s2,1),name= 'input_variable' )(inputs1)
                 

            #layer = BatchNormalization()(layer)
        elif  weather_covariate and not weather_maps and not calendar_cov:  
            s1 =inputs3.get_shape()[1]  
            s2 =inputs3.get_shape()[2] 
            s3 =inputs3.get_shape()[3]

            layer_exo = Reshape((s1*s2,s3),name= 'input_exo_cov' )(inputs3)
            layer_exo = BatchNormalization()(layer_exo)

            s1 =inputs1.get_shape()[1]  
            s2 =inputs1.get_shape()[2] 
            s3 =inputs1.get_shape()[3]
            layer = Reshape((s1*s2,1),name= 'input_power' )(inputs1)
            layer = BatchNormalization()(layer)
        else: #power and weathre maps

            #s3 =layer.get_shape()[3]
            print(layer.get_shape())

            layer_exo = layer
            exxxo_dim = layer_exo.get_shape()[-1]
            s1 =inputs1.get_shape()[1]  
            s2 =inputs1.get_shape()[2] 
            #s3 =inputs1.get_shape()[3]
            layer = Reshape((s1*s2,1),name= 'input_power' )(inputs1)
            # random.seed(seed_value)
            # seed_value = [random.randrange(1, 9999 ) for i in range(1)][0] 

            # stepsIn= layer.get_shape()[1]
            # layer = Reshape((stepsIn,1),name= 'input_variable' )(layer)

            #if ovelapping: 
            #     #layer = Flatten()(layer)
            #     stepsIn= layer.get_shape()[1]
            #     #layer = Dense(stepsIn, activation='linear' ,  kernel_initializer=initializers.HeNormal(seed=seed_value) )(layer)
            #     layer = Reshape((stepsIn,1),name= 'input_variable' )(layer)

            #else:
            #    s1 =layer.get_shape()[1]  
            #    s2 =layer.get_shape()[2] 
            #    layer = Flatten()(layer)

            #    layer = Dense(s1*s2, activation='linear' ,  kernel_initializer=initializers.HeNormal(seed=seed_value))(layer)
            #    layer = Reshape((s1*s2,1),name= 'input_variable' )(layer)



        GENERIC_BLOCK = 'generic'
        TREND_BLOCK = 'trend'
        SEASONALITY_BLOCK = 'seasonality'
        #output  = Dense(27)(layer)
        # output = Reshape((stepsOut,1),name= 'input_variable' )(output)
        # self.model = Model([inputs1, inputs2,inputs3,input4], output, name=self._FORECAST)

        # '''
       
        '''
        The following code is from: https://github.com/philipperemy/n-beats
        '''    
        self.stack_types = tuple(self.config.model_data.stack_types)# (TREND_BLOCK,SEASONALITY_BLOCK)#(GENERIC_BLOCK,GENERIC_BLOCK)# stack_types
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.thetas_dim = tuple( [ thetas_dim for _ in range(len(self.stack_types))]) # (4, 8)# thetas_dim
        self.units = nbeats_units
        self.share_weights_in_stack = False  if 'generic' in self.stack_types else True  # share_weights_in_stack
        self.backcast_length = stepsIn #backcast_length
        self.forecast_length = stepsOut#forecast_length
        self.input_dim = layer.get_shape()[-1]# input_dim
        self.input_shape = (self.backcast_length, self.input_dim)

        if weather_covariate and not weather_maps and not calendar_cov: 
            self.exo_dim = int(self.config.model_data.locations_n) * len(self.config.features) # exo_dim
            
        else:
            self.exo_dim  = exxxo_dim #0 
        self.exo_shape = (self.backcast_length, self.exo_dim)
        self.output_shape = (self.forecast_length, self.input_dim)
        self.weights = {}
        self.nb_harmonics =  nb_harmonics

        assert len(self.stack_types) == len(self.thetas_dim)

        x = layer #Input(shape=self.input_shape, name='input_variable')
        x_ = {}
        for k in range(self.input_dim):
            x_[k] = Lambda(lambda z: z[..., k])(x)
        e_ = {}
        if self.has_exog():
            #e = Input(shape=self.exo_shape, name='exos_variables')
            e =layer_exo 
            exo_filters= 32
            e =  Conv1D(filters=exo_filters, kernel_size= kernel_size , padding= "same" ,kernel_initializer=initializers.he_normal( seed=seed_value), name= 'latent_cov_features')(e)
            self.exo_dim = exo_filters
            #e = BatchNormalization()(e)

            for k in range(self.exo_dim):
                e_[k] = Lambda(lambda z: z[..., k])(e)
        else:
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

        if self.has_exog():
            n_beats_forecast = Model([inputs1, inputs2,inputs3,input4], y_, name=self._FORECAST)
            # n_beats_backcast = Model(x, x_, name=self._BACKCAST)
            n_beats_backcast = Model([inputs1, inputs2,inputs3,input4,x_ ], name=self._BACKCAST)
        else:  
            n_beats_forecast = Model([inputs1, inputs2,inputs3,input4], y_, name=self._FORECAST)
            # n_beats_backcast = Model(x, x_, name=self._BACKCAST)
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