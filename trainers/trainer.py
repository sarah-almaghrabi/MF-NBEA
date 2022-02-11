
from tensorflow.keras.models import load_model
from base.base_trainer import BaseTrain
import os
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping,ReduceLROnPlateau,LearningRateScheduler
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.plugins.hparams import api as hp
 
    
import random

def save_loss(loss, val_loss, name):
    folder_name = 'train_losses'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    pd.DataFrame({'loss':loss, 'val_loss':val_loss} ).to_csv(folder_name+'/'+name+'_loss.csv')


class Trainer(BaseTrain):
    def __init__(self, model, data, config,experiment,session_id):
        self.experiment =experiment
        super(Trainer, self).__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []


        
        self.checkpoint_dir, self.log_dir = self.init_callbacks(session_id)

        ''' seed setings '''
        seed_value = self.config.exp.seed_value
        os.environ['PYTHONHASHSEED']=str(seed_value)
        random.seed(seed_value)

        

    def init_callbacks(self,session_id):
        checkpoint_filepath = os.path.join(self.config.callbacks.checkpoint_dir,   self.config.exp.name+session_id+ '.ckpt')
 
        checkpoint_dir = os.path.normpath(checkpoint_filepath)
        self.callbacks.append(

            ModelCheckpoint(
                filepath=checkpoint_dir,     
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,                
            )
        )

        self.callbacks.append(

            EarlyStopping(
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                verbose=self.config.callbacks.checkpoint_verbose, 
                patience=self.config.callbacks.patience, 
                restore_best_weights=self.config.callbacks.restore_best_weights
                 )
        )
        import datetime
        log_dir = os.path.normpath( self.config.callbacks.tensorboard_log_dir)+ '\\'+self.config.exp.name +datetime.datetime.now().strftime("%Y%m%d-%H%M%S") +session_id 

        self.callbacks.append(
            TensorBoard(
                log_dir = log_dir,
                histogram_freq=1,
                write_graph=True
            )
        )
                    

        self.callbacks.append
        (
            LearningRateScheduler(schedule= lambda epoch: 1e-8 * 10**(epoch/20)  ,verbose=1 )
        )

         
        #self.callbacks.append(

        #    ReduceLROnPlateau(monitor='val_loss', factor=0.1,
        #                      patience=20, min_lr=0.001)
        #)
        # if ( "comet_key" in self.config):
        #     self.experiment.disable_mp()
        #     self.experiment.log_parameters(self.config.model)
        #     self.experiment.log_parameters(self.config.exp)
        #     self.experiment.log_parameters(self.config.dataset_file)
        #     self.experiment.log_parameters(self.config.model_data)
        #     self.experiment.log_parameters(self.config.trainer)

        #     self.callbacks.append(self.experiment.get_callback('keras'))

        return checkpoint_dir , log_dir

    def train(self,HPARAMS,plot = False):

        self.callbacks.append(hp.KerasCallback( self.log_dir, HPARAMS)) 
        
        history = self.model.fit(
            self.data[0] , 
            self.data[1],
            epochs=self.config.trainer.num_epochs,
            verbose=self.config.trainer.verbose_training,
            batch_size=self.config.trainer.batch_size,
            validation_split=self.config.trainer.validation_split,


            callbacks=self.callbacks  
        )
        self.loss.extend(history.history['loss'])
        self.val_loss.extend(history.history['val_loss'])
        if (plot == True):
            plt.plot(history.history['loss'] , label = 'loss')
            plt.plot(history.history['val_loss'] , label = 'val_loss')
            plt.legend()
            plt.ylabel('Loss')
            plt.xlabel('Epochs')
            plt.show()

        save_loss(loss=self.loss, val_loss=self.val_loss, name= 'powerMapG_with_lag'+str(self.config.model_data.window)  +self.config.dataset_file.siteName)

        print('training completed')
        self.model.load_weights(self.checkpoint_dir)
        return self.model
        

