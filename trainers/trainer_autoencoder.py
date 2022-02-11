
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
 
 

from base.base_trainer import BaseTrain
import os
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
 
import matplotlib.pyplot as plt

    
import random
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

# from kerashypetune import KerasGridSearch
 

class Trainer(BaseTrain):
    def __init__(self, model, data, config,experiment):
        self.experiment =experiment

        super(Trainer, self).__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []

        
        self.checkpoint_dir = self.init_callbacks()

        ''' seed setings '''
        # Set seed value
        seed_value = self.config.exp.seed_value
        os.environ['PYTHONHASHSEED']=str(seed_value)
        os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'
        random.seed(seed_value)
        tf.random.set_seed(seed_value)

        

    def init_callbacks(self):
        checkpoint_filepath = os.path.join(self.config.callbacks.checkpoint_dir_autenc,   self.config.exp.name+ '.ckpt')

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

        log_dir = os.path.normpath( self.config.callbacks.tensorboard_log_dir)+ '\\'+self.config.exp.name +datetime.datetime.now().strftime("%Y%m%d-%H%M%S") +'aucenc' 

        self.callbacks.append(
            TensorBoard(
                log_dir = log_dir,
                histogram_freq=1,
                write_graph=True
            )
        )
            

        return checkpoint_dir

    def train(self,plot = False):

        
        history = self.model.fit(
            # self.data[0], self.data[1],
            self.data[0][1], self.data[0][1],

            epochs=self.config.trainer.num_epochs,
            verbose=self.config.trainer.verbose_training,
            batch_size=self.config.trainer.batch_size,
            validation_split=self.config.trainer.validation_split,
            callbacks=self.callbacks,

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

        # self.experiment.log_metric("epochs", epochs)
        # self.experiment.log_metric("batch_size", batch_size)
        # self.experiment.log_metric("model", self.model)
        # self.experiment.log_histogram_3d(history.history['loss'], 'loss')
        # self.experiment.log_histogram_3d(history.history['val_loss'], 'val_loss')
        # self.experiment.set_model_graph( self.config.callbacks.tensorboard_write_graph )

        
        print('training completed')
        self.model.load_weights(self.checkpoint_dir)


    session.close()
