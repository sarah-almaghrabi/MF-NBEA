import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler
import copy
import matplotlib.pyplot as plt
from pathlib import Path
 
import pickle
import pprint 

class DataUtil(object):

    # This class contains data specific information.
    def __init__(self, configg,experiment):
            
        workpath = 'C:/Users/User'
        #workpath = '/Users/sarahalmaghrabi'

        self.config = configg
        # if ( "comet_key" in self.config):
        #     from comet_ml import Experiment
        #     self.experiment = Experiment(api_key=self.config.comet_key.comet_api_key, project_name=self.config.exp.name, workspace='sarah-almaghrabi')
        try:
            self.w =  self.config.model_data.window
            self.h =  self.config.model_data.horizon
            self.samplePerDay = self.config.dataset_file.samplePerDay 




            data_dir = 'data/'
            siteName=self.config.dataset_file.siteName
            dataColName=siteName+'(aggregated)'
            # dataColName= 'GRIFSF1'#'MOREESF1' #'BROKENH1'

            # file = data_dir+'power/' + self.config.dataset_file.file_name

            file = data_dir+'power/2019_2020_2021_' + self.config.dataset_file.file_name
            print(file)
            self.data = pd.read_csv(file,  index_col=0 , parse_dates=True, dayfirst=True,usecols=['Timestamp',dataColName] )
            print( self.data )

            #####
            # 
            # self.data = self.data.loc[:'2019-2-1',: ]
            self.data.index = pd.to_datetime(self.data.index, format = format, utc=True)

            ## create calendar features 
            calendar_feat = pd.DataFrame()
            calendar_feat['month'] = self.data.index.month
            calendar_feat['day_of_week'] = self.data.index.day_of_week
            calendar_feat['day_of_year'] = self.data.index.day_of_year
            calendar_feat['hour'] = self.data.index.hour
            calendar_feat['minute'] = self.data.index.minute
            #print(calendar_feat)

            #
            #####
            #print( self.data.iloc[27*3:27*5] )
            
            #weather features 
            wanted_features = list(self.config.features) # [ 'Ghi' , 'CloudOpacity','AirTemp']
            print('wanted_features len:',len(wanted_features))
            print('wanted_features :',wanted_features)

            wanted_locations = [ 'BERYLSF1', 'BROKENH1','COLEASF1','GRIFSF1','GULLRSF1','MANSLR1','MLSP1','MOREESF1','NYNGAN1','PARSF1','ROYALLA1','WRSF1']
            optimal_map_size = int(np.ceil(np.sqrt(  self.config.model_data.locations_n)))
            #print(self.data.shape)

            ##np array loads  the weather data as maps for each ts
            weather_data = np.zeros( shape = (int(len(wanted_features)), self.data.shape[0],optimal_map_size,optimal_map_size))
            
            # weather_data_df = np.zeros(shape=( len(wanted_features), self.data.shape[0],len(wanted_locations)  ))
            weather_data_df = pd.DataFrame()
            for feature_i , feature in enumerate(wanted_features): 

                temp  =  pd.read_csv(workpath +'/OneDrive - RMIT University/Data_2019_to_2020/weather_data/solcast/combined_features_from_locations/'+siteName+'/'+feature+'.csv',  index_col=0 , parse_dates=True, dayfirst=True)#, usecols=['PeriodStart','Ghi_'+dataColName])
                #get weather of same time indexes in the pv data 
                #print(temp)
                temp = temp.loc[self.data.index , :]
                # weather_data_df[feature_i]=temp.values
                weather_data_df = pd.concat( [weather_data_df ,  temp] ,axis=1)
            # for feature in wanted_locations : 
            #     weather_data =  pd.read_csv(data_dir    +'/weather_per_location/'+feature+'.csv',  index_col=0 , parse_dates=True, dayfirst=True)#, usecols=['PeriodStart','Ghi_'+dataColName])
            #     weather_data.index.name='Timestamp'

                feature_data = np.zeros( shape=(self.data.shape[0],optimal_map_size,optimal_map_size))

                try:
                    #print("inside try ")
         
                    my_file = Path('C:/Users/User/Documents/maps_weather_'+siteName+'.pkl')
                    if  not my_file.is_file():

                        for ts_i, ts in enumerate(self.data.index):
                            ts = str(self.data.index[ts_i]).replace('-',"").replace(':',"")[:-7].replace(' ','_')
                            #print("ts_i:",ts_i)
                            #imputed gaps in the maps 
                            #feature_data[ts_i,:]= pd.read_csv(workpath+"/OneDrive - RMIT University/experiments/Multivariate_experiments/"+siteName+'/'+feature+'/' +ts+'.csv',index_col=0).values
                            feature_data[ts_i,:]= pd.read_csv("C:/Users/User/Documents/imputed_"+siteName+'/'+feature+'/' +ts+'.csv',index_col=0).values
                        
                            #not_imputed 
                            #feature_data[ts_i,:]= pd.read_csv(workpath+"/OneDrive - RMIT University/experiments/Multivariate_experiments/"+"not_imputed_"+siteName[:-1]+'/'+feature+'/' +ts+'.csv',index_col=0).values
                            #feature_data[ts_i,:]= pd.read_csv("/OneDrive - RMIT University/experiments/Multivariate_experiments/"+siteName+'/'+feature+'/' +ts+'.csv',index_col=0).values
                    else: 
                        continue

                        #print('done')
                except IOError as err:
                    print('OO')
                # weather_data.index = self.data .index
                #print(feature)
                #print(feature_data.shape)
                #print(feature_data)

                weather_data [feature_i ] = copy.deepcopy(feature_data)
                #print(weather_data.shape)

            ##save the created maps for eath ts
            if  not my_file.is_file():

                output = open(my_file, 'wb')
                pickle.dump(weather_data, output)
                output.close()
            else: 
                pkl_file = open(my_file, 'rb')
                _ =  pickle.load(pkl_file)
                weather_data = _ .reshape(weather_data.shape)
                #pprint.pprint(weather_data)
                pkl_file.close()

            calendar_feat =calendar_feat.values

            weather_data_df = weather_data_df.values


            # cols = np.array(self.data.columns[:-1].values)
            # cols = cols.reshape(  cols.shape[0]//len(wanted_features),len(wanted_features) )
            # print(cols)
            # print(cols.shape)

            #print(self.data.columns)
            #print(self.data)


           
            self.n_features = len(wanted_features)+1  #weather+pv 

            #change power data shape into 3D shape days, steps , 1
            self.series = self.data.values
            #calendar_feat = calendar_feat.reshape( (   calendar_feat.shape[0]// self.samplePerDay,  self.samplePerDay, calendar_feat.shape[-1]  ))
            ## for map representaion of weather data : n_features, timesteps (samplePerDay*lags) , 4,4 
            #weather_data = weather_data.reshape( ( len(wanted_features), weather_data.shape[1]// self.samplePerDay,  self.samplePerDay, weather_data.shape[-2],weather_data.shape[-1] ))
            ## table representaion of weather data 
            #weather_data_df = weather_data_df.reshape(     weather_data_df.shape[0]// self.samplePerDay, self.samplePerDay,weather_data_df.shape[-1]   )
            
            print('power data:',self.data.shape)
            print('calendar_feat:',calendar_feat.shape)
            print('weather_data data:',weather_data.shape)
            print('weather_data_df :',weather_data_df.shape)
            
            
               
            #split 
            splitindex = 365*self.samplePerDay *2
            # splitindex = 20*self.samplePerDay #*2
            print('splitindex',splitindex)
            print('splitindex - self.w',splitindex - self.w)
            print('splitindex + self.w',splitindex + self.w)
            self.series_train,self.series_test = self.series[ : splitindex + self.w, : ] ,  self.series [ splitindex - self.w :,: ] 
            self.series_train_calendar,self.series_test_calendar =   calendar_feat  [ : splitindex + self.w, : ] , calendar_feat [ splitindex - self.w :,: ]
            
            self.series_train_exo_map,self.series_test_exo_map =weather_data [:, : splitindex + self.w, :,: ] , weather_data [:, splitindex - self.w : , :,: ] 
            self.series_train_exo_df,self.series_test_exo_df =weather_data_df [  : splitindex+ self.w, : ] , weather_data_df [ splitindex - self.w : , : ] 


            

            self.maxCapacity = self.series_train[:,-1].max() 
            print('series_train',self.series_train.shape)
            print('series_test',self.series_test.shape)
            print('series_train_calendar',self.series_train_exo_map.shape)
            print('series_train_maps',self.series_test_exo_map.shape)
            print('series_train_df',self.series_train_exo_df.shape)







            # use list of scalars, more convinent in the case of multiple features 
            self.scalars  = None
            if self.config.model_data.normlize : 
                for col in range(self.series_train.shape[-1]): #for each feature 
                    #scale training data
                    # scalar_=None
                    # self.series_train[:-self.w ,:,col] , scalar_ =  self.normilze_data(data = self.series_train[:-self.w ,:,col]  )
                    self.series_train[: ,col] , scalar_ =  self.normilze_data(data = self.series_train[: ,col]  )
                    self.scalars =  scalar_
                    #scale testing data 
                    # self.series_test[self.w :,:,col] , _ =  self.normilze_data(data =self.series_test[self.w :,:,col] , fitted_scalar=scalar_ )
                    self.series_test[ :,col] , _ =  self.normilze_data(data =self.series_test[ :,col] , fitted_scalar=scalar_ )
                print('data are normalized ')


            
            # prepare x and y   (PV)
            self.x_train, self.y_train = self.get_x_y(data =self.series_train  , lag=self.w ,out=self.h)
            self.x_test, self.y_test = self.get_x_y(data =self.series_test   , lag=self.w  ,out=self.h)
            print('self.x_train:',self.x_train.shape)
            print('self.y_train:',self.y_train.shape)
            print('self.x_test: ',self.x_test.shape)
            print('self.y_test: ',self.y_test.shape)     
            
            # prepare x and y   (calendar)
            self.x_train_cal, _ = self.get_x_y(data =self.series_train_calendar   , lag=self.w ,out=self.h)
            self.x_test_cal, _ = self.get_x_y(data =self.series_test_calendar    , lag=self.w  ,out=self.h)

            print('self.x_train_cal:',self.x_train_cal.shape)
            print('self.x_test_cal:',self.x_test_cal.shape)            
                        
       
          



            
            # prepare x   (exo map)
            self.x_train_exo_map = self.get_x_y_maps(data =self.series_train_exo_map   , lag=self.w ,out=self.h)
            self.x_test_exo_map = self.get_x_y_maps(data =self.series_test_exo_map   ,   lag=self.w ,out=self.h)
            
            print('self.x_train_exo_map:',self.x_train_exo_map.shape)
            print('self.x_test_exo_map:',self.x_test_exo_map.shape)


            # prepare x   (exo df)
            self.x_train_exo_df, _= self.get_x_y(data =self.series_train_exo_df  , lag=self.w ,out=self.h )
            self.x_test_exo_df, _ = self.get_x_y(data =self.series_test_exo_df  , lag=self.w ,out=self.h )
                       

            print('self.x_train_exo_df:',self.x_train_exo_df.shape)
            print('self.x_test_exo_df:',self.x_test_exo_df.shape)


            # actual = denormilse_data (data=y_test, fitted_scalar=scalars[-1])


            
            


        except IOError as err:
            # In case file is not found, all of the above attributes will not have been created            
            text = "Error opening data file ... %s"+ str( err)
            print('\n\n\n\n\n*****************************************\n\n\n\n\n'+text)
            #experiment.log_text(text=text)


    
    def split_data( self,series , split_index = 365 ,lag =7  ):
        '''
        function to split the series into training and testing data 
        inputs: 
            - series : the data shaped into 3D shape
            - split index: index of sequence to split from 
            - lag 
        outputs: 
            series_train, series_test 
        '''
        #split to train-test 
        # consider the first year of the data for training and the second year for testing 
        series_train = series[:split_index+lag] 
        series_test= series[split_index-lag:]

        #print('series_train',series_train.shape)
        #print('series_test',series_test.shape)

        # maxcapacity = series_train.max()
        return series_train,series_test
    
        
    def normilze_data (self,data, fitted_scalar=False):
        '''
        function to normilze the data 
        inputs: 
            - data: data to be normalised 
            - fitted scalar: provide the scalar if it is already fitted  (e.g. to scale testing data using a scalar fitted on train data )
        outputs: 
            -  scaled_data and  fitted_scalar
        '''
        scaled_data = copy.deepcopy(data)
        if not fitted_scalar:
            fitted_scalar = MinMaxScaler(feature_range=(0,1))
            # fitted_scalar = StandardScaler()
            # fitted_scalar = RobustScaler()
            # fitted_scalar = FunctionTransformer(func = np.log1p , inverse_func =np.expm1 )
            
            
            fitted_scalar.fit( scaled_data )
        # else: 
        #     fitted_scalar.fit( scaled_data )
        
        scaled_data = fitted_scalar.transform(scaled_data)
        return scaled_data, fitted_scalar


    def denormilse_data (self,data, fitted_scalar):
        '''
        function to de-normilze the data 
        inputs: 
            - data:   normalised data
            - fitted scalar: provide the fitted scalar  
        outputs: 
            -  data in its normal scale 
        '''
        denormilsed = copy.deepcopy(data)
        return fitted_scalar.inverse_transform(denormilsed)

    def get_x_y(self,data, lag=1 ,   out =1):
        '''
        function to get the x and y samples , this function can work  for univariate and multivariate 
        inputs: 
            - data:   data frame,  the data should be  with 3D shape  as [number of days ,time samplePerDay , noumber of features ]
            - samplePerDay: number of samplePerDay in the sequence, to mimic number of timesamplePerDay per day 
            - features: number of features/channels in the provided data
        outputs: 
            -  x and y samples 
        '''  
        #print(data.shape)
        features = data.shape[-1]
        x = np.zeros( shape= (data.shape[0] - lag -out , lag  , features )) 
        y = np.zeros( shape= (data.shape[0] - lag -out ,  out ))
        print(x.shape)
        print(y.shape)
        for i in range(x.shape[0]):
            #print(i)

            for j in range(features):
                x[i,:,j] = data[i:i+lag,j] #.flatten() 
                # x[i,:,j] = data[i:i+lag,:,j] .flatten() 
                # x[i,:,:,j] = data[i:i+lag,:,j].transpose() #.flatten() 
                y[i] = data[i+lag: i+lag + out  ,-1  ] # labels is the last column in the data 

        # x = x.reshape(*x.shape[:1], -1, *x.shape[-1:])

        return x, y 


    def get_x_y_maps(self,data,  lag=1 ,out=1):
        '''
        function to get the x  samples for the exogenous maps  data 
        inputs: 
            - data:   data frame,  the data should be  with 5D shape  as [number of feattures , number of days ,time samplePerDay , rows,cols ]
            - samplePerDay: number of samplePerDay in the sequence, to mimic number of timesamplePerDay per day 
            - features: number of features/channels in the provided data
        outputs: 
            -  x samples 
        '''  
        #print(data.shape)
        features = data.shape[0]
        x = np.zeros( shape= ( data.shape[1] - lag -out, features,    lag  ,data.shape[-2], data.shape[-1] )) 

        #print(x.shape)
        for i in range(x.shape[0]):
            for j in range(features): 
                x[i,j,:,:,:] = data[j,i:i+lag,:,:] 

        x = x.reshape(*x.shape[:2], -1, *x.shape[-2:])
        return x  
    
     
    def series_3d_reshape(self, df, samplePerDay = 27 , lag = 7 , n_features= 1):
        '''function to reshape the data to match 3D shape [number of days ,time samplePerDay , noumber of features ]
        inputs: 
            - df : pandas dataframe 
            - samplePerDay: default  = 27 # samplePerDay or samples per day
            - lag : default = 7 # length of historical lookback 
            - n_features : the number of features to be used in the regression model 
        '''
        
        df_=copy.deepcopy(df)
        ## univariate model only power data 
        if n_features == 1 :
            series = df_.iloc[:,-1].values 
            # series = df_ .values 
            #print(series)
        else: 
        #multivariate model with weather features
        # df_ = df_.iloc[:,[0,2,-1]] #select some weatehr feature and the power feature 
        # n_features = df_ .shape[-1] # number of features 
            series = df_ .values


        #reshape to match 3D shape [number of days ,time samplePerDay , noumber of features ]
        series = series.reshape((len(series)//samplePerDay, samplePerDay, n_features))

        #print(series.shape) 

        return series

    def get_train_data(self):
        #print(self.y_train .shape)
        return [self.x_train,self.x_train_exo_map ,self.x_train_exo_df, self.x_train_cal ]   , self.y_train 
    def get_test_data(self):
        return [self.x_test,self.x_test_exo_map,self.x_test_exo_df , self.x_test_cal], self.y_test 
    
    def encode_maps_data(self):
        from tensorflow.keras.models import load_model

        print('inside incoding func ')
       
        encoder = load_model( 'encoder_'+self.config.dataset_file.siteName+'.h5', compile=True)

        print('self.x_train_exo_map ',self.x_train_exo_map .shape)

        encoded = [] 
        for inde in range(self.x_train_exo_map.shape[0]):
            example = self.x_train_exo_map[inde,:,:,:,:] 
            example = example[np.newaxis,:,:,:,:]
            encoded.append( encoder.predict(example)[0] ) 

        encoded = np.array(encoded)

        self.x_train_exo_map = encoded

        encoded = [] 
        for inde in range(self.x_test_exo_map.shape[0]):
            example = self.x_test_exo_map[inde,:,:,:,:] 
            example = example[np.newaxis,:,:,:,:]
            encoded.append( encoder.predict(example)[0] ) 
        encoded = np.array(encoded)    
        self.x_test_exo_map = encoded
        #reshape the data to match the other data (lags, samples perday ) shape 
        

        ##not needed for overlapping 
        #self.x_train_exo_map=self.x_train_exo_map.reshape(self.x_train_exo_map.shape[0],self.x_train_exo_map .shape[1],self.w,self.x_train_exo_map.shape[2]//self.w, self.x_train_exo_map .shape[-2]*self.x_train_exo_map .shape[-1])
        #self.x_test_exo_map=self.x_test_exo_map.reshape(self.x_test_exo_map.shape[0],self.x_test_exo_map .shape[1],self.w,self.x_test_exo_map.shape[2]//self.w, self.x_test_exo_map .shape[-2]*self.x_test_exo_map .shape[-1])

        print('before swapping axes:,self.x_train_exo_map,: ',self.x_train_exo_map.shape)
        self.x_train_exo_map = np.moveaxis(self.x_train_exo_map, 1, -1)
        self.x_test_exo_map = np.moveaxis(self.x_test_exo_map, 1, -1)
        print('finish from incoding func ')
        print('self.x_train_exo_map ',self.x_train_exo_map .shape)
        print('self.x_test_exo_map ',self.x_test_exo_map .shape)


        # temp = np.zeros( shape=( self.x_train_exo_map.shape[0], self.x_train_exo_map.shape[0],self.x_train_exo_map.shape[2], self.x_train_exo_map.shape[3], self.x_train_exo_map.shape[4] +self.x_train_exo_map.shape[1] ))
        # for 
