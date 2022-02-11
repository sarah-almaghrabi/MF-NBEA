import copy 
from sklearn.linear_model import Ridge,RidgeCV
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputRegressor

def get_ensemble(trainX, trainy,model, pca_comp = 0,n=1,correction_power_only=False , calendar=False ):
    print("get_ensemble - trainX.shape")
    print(trainX[0].shape)
    print(trainX[1].shape)
    print(trainX[2].shape)
    print(trainy.shape)
    model_1 = model#copy.copy(model)
    yhat = np.zeros( shape = trainy.shape)
    for ind in range(trainX[0].shape[0]): #loop all examples 
        x=[] 
        for input_item  in trainX: ## for each input kind in the input list 
            inp_shape= [1]
            for sh in input_item[ind].shape :
                inp_shape.append (sh) 
            #print('inp_shape:',inp_shape)
            x.append(input_item[ind].reshape(inp_shape) )
    
        yhat[ind] = model_1.predict(x)[0,:,0] ##  outputshpe will be [1,27,1]


    
    if len(yhat.shape) > 2: 
        yhat.shape    =  yhat.shape[0], yhat.shape[1]

    y_res = trainy - yhat   #y2

    print('here')
    
    models = [model_1]  #this list contains all week model and redisuals 
    # res_model = model_1
    print('ensembles:',n)
    for i in range(n):
        print('i .. ',i)
        # y_res = trainy - yhat


        if correction_power_only:
            print('only use power for correction ')
            trainX_2D = trainX[0] #only power 

        elif calendar  :  #power, weather and claendar             
            trainX_2D = np.concatenate([trainX[2],trainX[0],trainX[3] ], axis=-1)
                   
        else: #power and weather   
            trainX_2D = np.concatenate([trainX[2],trainX[0] ], axis=-1)
        
        print('data is reshaped to ',trainX_2D.shape)

        if len( trainX_2D.shape) == 4:                
            trainX_2D = trainX_2D.reshape(trainX_2D.shape[0],trainX_2D.shape[1]*trainX_2D.shape[2]*trainX_2D.shape[3])
        elif len( trainX_2D.shape)  == 3:
            trainX_2D = trainX_2D.reshape(trainX_2D.shape[0],trainX_2D.shape[1]*trainX_2D.shape[2] )
        print('data is reshaped to ',trainX_2D.shape)

        if pca_comp>0 :
            trainX_2D = PCA(n_components=pca_comp   ).fit_transform(trainX_2D)
        print('data is transformed to ',trainX_2D.shape)

        res_model = fit_res_model(trainX_2D, y_res)
        print('res model  is fitted')

        yhat = res_model.predict(trainX_2D)
        y_res = y_res - yhat
        # print('y_res:',y_res[0])
        #append models fitted on residuals 
        models.append(  res_model) 
        print('res model  is appended')

    print('ensemble model with '+ str(n),' boosts is created')
    return models 


def fit_res_model(trainX, trainy):
    print("fit_res_model - trainX.trainy")
    print(trainX.shape)
    print(trainy.shape)

    # print("fit_res_model after - trainX_2D.shape")
    # print(trainX_2D.shape)
    ##single model to learn residuals 
    # model  = Ridge(alpha=5)
    model  = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1],cv=5)
    # model =KNeighborsRegressor(n_neighbors=5 , weights='distance')
    model = MultiOutputRegressor(model)
#     model = DecisionTreeRegressor(max_depth=1,random_state=0)
    model.fit(  trainX, trainy)

    # similar as the preddciton mdeol 
    # model = fit_model(trainX, trainy)

    ### stacking models 
#     model = get_stacking()
#     model.fit(trainX, trainy)
# #     print('fitted')
#     model.estimator.estimators_ = model.estimator.estimators
#     model.estimator.final_estimator_ = model.estimator.final_estimator
#     model.estimator.stack_method_ = model.estimator.stack_method    
    return model