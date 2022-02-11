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
        # self.result = self.computeAccuracy(actData, predData, maxCapacity)

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

        # rowLabel=copy.deepcopy(timeString)
        # rowLabel.append('overall')
        # accuracy={'Time': rowLabel, 'MAE': MAE, 'MRE': MRE, 'MSE':MSE, 'RMSE':RMSE, 'R2':R2}
        accuracy={'MAE': MAE, 'MRE': MRE, 'MSE':MSE, 'RMSE':RMSE, 'R2':R2}
        accDF=pd.DataFrame(accuracy)
        # self.experiment.log_metric("pred_MAE", MAE[-1])
        # self.experiment.log_metric("pred_MRE", MRE[-1])
        # self.experiment.log_metric("pred_RMSE", RMSE[-1])


        # self.experiment.log_table(filename=self.config.exp.name+'_acc.csv', tabular_data=accDF, headers=True)

        predVsAc = pd.DataFrame({ 'Actual': actFlat, "Predictions": predFlat})
        # self.experiment.log_table(filename=self.config.exp.name+'_AvP.csv', tabular_data=predVsAc, headers=True)
        # self.experiment.log_curve( name= "MAE", y= MAE[:-1] ,x=[i for i in range(27)])
        # self.experiment.log_curve( name= "MRE", y= MRE[:-1] ,x=[i for i in range(27)])
        # self.experiment.log_curve( name= "RMSE", y= RMSE[:-1] ,x=[i for i in range(27)])
        predVsAc.to_csv( 'results/AvP_DL_'+ self.config. dataset_file.siteName+ "_lag_"+ str(self.config.model_data.window )+  "_"+self.config.exp.name.split('_')[-1].upper()+details +'.csv' )



        # print(accDF)     

        return [actFlat , predFlat, accDF ]

 