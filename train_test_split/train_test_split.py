import os
import shutil
import numpy as np


root_path='dataset'
classes=os.listdir(root_path)
train_path='train'
test_path='test'
val_path='val'
#train_ratio=0.7
#val_ratio=0.2

filepath=[] 
for dirname, _, filenames in os.walk(root_path):
    for filename in filenames:
        filepath.append(os.path.join(dirname, filename))


for file in filepath:
    sep = file.split('\\')[1]
    os.makedirs(os.path.join(train_path,sep),exist_ok=True)
    os.makedirs(os.path.join(test_path,sep),exist_ok=True)
    os.makedirs(os.path.join(val_path,sep),exist_ok=True)


train_FileNames, val_FileNames, test_FileNames = np.split(np.array(filepath),[int(len(filepath)*0.7), int(len(filepath)*0.30)])



for file in train_FileNames:
    sep1=file.split('\\')[1]
    shutil.copy(file,os.path.join(train_path,sep1))


for file in test_FileNames:
    sep2=file.split('\\')[1]
    shutil.copy(file,os.path.join(test_path,sep2))


for file in val_FileNames:
    sep3=file.split('\\')[1]
    shutil.copy(file,os.path.join(val_path,sep3))



