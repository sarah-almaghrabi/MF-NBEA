
#from tensorflow.compat.v1 import ConfigProto
#from tensorflow.compat.v1 import InteractiveSession
#config = ConfigProto()
#config.gpu_options.allow_growth = True
#session = InteractiveSession(config=config)
# __import__("tensorflow").compat.v1.enable_eager_execution()
# import tensorflow.compat.v1 as tf

#config = ConfigProto()
#config.gpu_options.allow_growth = True
#session = InteractiveSession(config=config)

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
        # Set seed value
        seed_value = self.config.exp.seed_value

        os.environ['PYTHONHASHSEED']=str(seed_value)

        # Set `python` built-in pseudo-random generator at a fixed value
        random.seed(seed_value)
        # Set `tensorflow` pseudo-random generator at a fixed value

        # self.experiment.log_other("random seed", seed_value)
        

    def init_callbacks(self,session_id):
        #work_path = 'C:/Users/User/OneDrive - RMIT University/experiments/Multivariate_experiments/DL_model_with_maps_data/'
       
        #exp = os.path.join( '%s-{epoch:02d}.ckpt' % self.config.exp.name)
        #checkpoint_filepath = work_path  + self.config.callbacks.checkpoint_dir.replace("\\",'/')+ exp

        #print('self.config.callbacks.checkpoint_dir',checkpoint_filepath)
        #checkpoint_filepath = os.path.join(self.config.callbacks.checkpoint_dir, '%s-{epoch:02d}.ckpt' % self.config.exp.name)
        checkpoint_filepath = os.path.join(self.config.callbacks.checkpoint_dir,   self.config.exp.name+session_id+ '.ckpt')
 
        checkpoint_dir = os.path.normpath(checkpoint_filepath)

    #     latest = tf.train.latest_checkpoint(checkpoint_dir)

        #checkpoint_dir = os.path.normpath('/checkpoints')


        #checkpoint_dir=r"D:\data"

        #checkpoint_dir = os.path.normpath(checkpoint_dir)
        # print('checkpoint_dir:',checkpoint_dir)

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

        #log_dir = os.path.normpath( self.config.callbacks.tensorboard_log_dir)+ 'HyperParams' 
        self.callbacks.append(hp.KerasCallback( self.log_dir, HPARAMS)) 
        # print(self.log_dir)
        # print(self.callbacks)
        history = self.model.fit(
            # self.data[0], self.data[1],
            self.data[0] , 
            self.data[1],
            epochs=self.config.trainer.num_epochs,
            verbose=self.config.trainer.verbose_training,
            batch_size=self.config.trainer.batch_size,
            validation_split=self.config.trainer.validation_split,


            callbacks=self.callbacks  
        )
        self.loss.extend(history.history['loss'])
        # self.acc.extend(history.history['mae'])
        self.val_loss.extend(history.history['val_loss'])
        # self.val_acc.extend(history.history['val_mae'])
        if (plot == True):
            plt.plot(history.history['loss'] , label = 'loss')
            plt.plot(history.history['val_loss'] , label = 'val_loss')
            plt.legend()
            plt.ylabel('Loss')
            plt.xlabel('Epochs')
            plt.show()

        #save_loss(loss=self.loss, val_loss=self.val_loss, name= 'powerMapG_with_custome_weghits_lag'+str(self.config.model_data.window)  +self.config.dataset_file.siteName)
        save_loss(loss=self.loss, val_loss=self.val_loss, name= 'powerMapG_with_lag'+str(self.config.model_data.window)  +self.config.dataset_file.siteName)
        # self.experiment.log_metric("epochs", epochs)
        # self.experiment.log_metric("batch_size", batch_size)
        # self.experiment.log_metric("model", self.model)
        # self.experiment.log_histogram_3d(history.history['loss'], 'loss')
        # self.experiment.log_histogram_3d(history.history['val_loss'], 'val_loss')
        # self.experiment.set_model_graph( self.config.callbacks.tensorboard_write_graph )

        
        #self.model.save(os.path.join(self.config.callbacks.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp.name))
        print('training completed')
        self.model.load_weights(self.checkpoint_dir)
        return self.model
    #def get_best_trained_model(self ):
    #    print("inside get_best_trained_model")
    #    from tensorflow.keras.models import load_model

    #    checkpoint_filepath = os.path.join(self.config.callbacks.checkpoint_dir, '%s-{epoch:02d}.ckpt' % self.config.exp.name)
    #    checkpoint_dir = os.path.dirname(checkpoint_filepath)
    #    latest = load_model(checkpoint_dir)
            
    #    #latest = tf.train.latest_checkpoint(checkpoint_dir)
    #    print('latest:',latest.summary())
    #    return latest 



