import numpy as np
import random


def config():
    config_dict={}
    learning_rate= random.uniform(0,1)
    epochs=random.randint(20,50)
    train_path='train'
    test_path='val'
    optimizer_list=['Adam','Adadelta','Adagrad']
    optimizer=random.choice(optimizer_list)
    model_list=['custom_cnn','vgg_model','resnet']
    model_name=random.choice(model_list)
    config_dict={'learning_rate':learning_rate,'Epochs':epochs,'Train_path':train_path,'Test_path':test_path,'Optimizer':optimizer,'model_name':model_name}
    return(config_dict)










