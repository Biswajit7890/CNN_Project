
import numpy as np 
import random
from keras.callbacks import CSVLogger
from model.model import preprocessing
from model.model import custom_cnn
from model.model import vgg_model
from model.model import res_model
from config.config import config


def model_build():
    model_names=['custom_cnn','vgg_model','res_model']
    model_choice=random.choice(model_names)
    config_dict = config()
    if (model_choice == 'custom_cnn'):
        csv_logger1 = CSVLogger("custom_cnn.csv", append=True)
        train_data,test_data,target= preprocessing(config_dict)
        model1 = custom_cnn(config_dict)
        model1.fit(train_data,test_data,config_dict['Epochs'],callbacks=(csv_logger1))
        model1.save('model/cnn.h5')
    elif (model_choice == 'vgg_model'):
          csv_logger2 = CSVLogger("vgg.csv", append=True)
          train_data,test_data,target= preprocessing(config_dict)
          model2 = vgg_model(config_dict)
          model2.fit(train_data,test_data,config_dict['Epochs'],callbacks=(csv_logger2))
          model2.save('model/vgg.h5')
    elif (model_choice == 'res_model'): 
          csv_logger3 = CSVLogger("res.csv", append=True)
          train_data,test_data,target= preprocessing(config_dict)
          model3 = res_model(config_dict)
          model3.fit(train_data,test_data,config_dict['Epochs'],callbacks=(csv_logger3))
          model3.save('model/res.h5')
               
        
model_build()    
        
    
      
   
  
   













































