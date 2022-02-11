import os, copy
import pandas as pd
import numpy as np
from sklearn.metrics import *

class Evaluater(object):    
    def __init__(self,config,experiment):
        self.experiment = experiment
        self.config =config
        if not os.path.exists('results'):
            os.makedirs('results')

    def computeAccuracy(self, actData, predData, maxCapacity, timeString=0,details=''):
        MAE=[]
        MRE=[]
        MSE=[]
        RMSE=[]
        R2=[]
        for i in range(actData.shape[1]):
            curTarget=actData[:,i]
            curPred=predData[:,i]

            absError=np.abs(curTarget-curPred)
            relabsError=absError/maxCapacity

            MAE.append(np.nanmean(absError))
            MRE.append(np.nanmean(relabsError) * 100)
            MSE.append(mean_squared_error(y_true=curTarget, y_pred=curPred))
            RMSE.append(np.sqrt(mean_squared_error(y_true=curTarget, y_pred=curPred)))
            R2.append(r2_score(y_true=curTarget, y_pred=curPred))

        actFlat=actData.flatten()
        predFlat=predData.flatten()

        absError=np.abs(actFlat-predFlat)
        relabsError=absError/maxCapacity

        MAE.append(np.nanmean(absError))
        MRE.append(np.nanmean(relabsError) * 100)
        MSE.append(mean_squared_error(y_true=actFlat, y_pred=predFlat))
        RMSE.append(np.sqrt(mean_squared_error(y_true=actFlat, y_pred=predFlat)))
        R2.append(r2_score(y_true=actFlat, y_pred=predFlat))


        accuracy={'MAE': MAE, 'MRE': MRE, 'MSE':MSE, 'RMSE':RMSE, 'R2':R2}
        accDF=pd.DataFrame(accuracy)

        predVsAc = pd.DataFrame({ 'Actual': actFlat, "Predictions": predFlat})
        predVsAc.to_csv( 'results/AvP_DL_'+ self.config. dataset_file.siteName+ "_lag_"+ str(self.config.model_data.window )+  "_"+self.config.exp.name.split('_')[-1].upper()+details +'.csv' )





        return [actFlat , predFlat, accDF ]

 