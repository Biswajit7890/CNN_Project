from keras.preprocessing.image import ImageDataGenerator 
import warnings
import sys
warnings.filterwarnings('ignore')
from keras import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50



def preprocessing(config_dict):
    train_gen=  ImageDataGenerator(rescale=1./255,samplewise_center=True,zoom_range=0.6,brightness_range=[0.2,0.8])
    train_data= train_gen.flow_from_directory(config_dict['Train_path'],target_size=(64,64) ,batch_size=32,class_mode='categorical',shuffle=True)
    test_gen =  ImageDataGenerator(rescale=1./255)
    test_data = test_gen.flow_from_directory(config_dict['Test_path'],target_size=(64,64),batch_size=1,shuffle=False)
    target = train_data.classes
    return(train_data,test_data,target)

def custom_cnn(config_dict):
    model=Sequential()
    model.add(Conv2D(32,(3,3),strides=2,activation='relu',padding='valid',input_shape=(64,64,3)))
    model.add(Conv2D(50,(3,3),strides=3,activation='relu',padding='same'))
    model.add(MaxPooling2D(2,2))
    model.add(Flatten())
    model.add(Dense(10,activation='sigmoid'))
    model.add(Dense(30,activation='sigmoid'))
    model.add(Dense(5,activation='softmax'))
    print(model.summary())
    model.compile(loss='categorical_crossentropy',optimizer=config_dict['Optimizer'],metrics=['accuracy'])
    return(model)

def vgg_model(config_dict):
    IMAGE_SIZE = [64, 64] 
    vgg = VGG16(input_shape = IMAGE_SIZE + [3], weights = 'imagenet', include_top = False) 
    for layer in vgg.layers:
        layer.trainable = False
    x = Flatten()(vgg.output)
    x = Dense(5, activation = 'softmax')(x)
    model = Model(inputs = vgg.input, outputs = x)
    model.compile(loss='categorical_crossentropy', optimizer=config_dict['Optimizer'], metrics=['accuracy'])
    print(model.summary())
    return(model)

def res_model(config_dict):
    base_model = ResNet50(input_shape=(224, 224,3), include_top=False, weights="imagenet")
    for layer in base_model.layers:
        layer.trainable = False
    base_model = Sequential()
    base_model.add(ResNet50(include_top=False, weights='imagenet', pooling='max'))
    base_model.add(Dense(5, activation='sigmoid'))
    base_model.compile(loss='categorical_crossentropy', optimizer=config_dict['Optimizer'], metrics=['accuracy'])
    print(base_model.summary())
    return (base_model)









































































