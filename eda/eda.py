import keras
import os
import matplotlib.pyplot as plt
from keras.preprocessing import image
import warnings
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
warnings.filterwarnings(action='ignore')

data_path='dataset'

filepath=[] 
for dirname, _, filenames in os.walk(data_path):
    for filename in filenames:
        filepath.append(os.path.join(dirname, filename))
clases=os.listdir(data_path)

for cls in clases:
    if(cls=='cloudy'):
         cloudy_images=os.listdir(os.path.join(data_path,cls))
    elif(cls =='foggy'):
         foggy_images=os.listdir(os.path.join(data_path,cls))
    elif(cls=='rainy'):
         rainy_images=os.listdir(os.path.join(data_path,cls))
    elif(cls=='shine'):
         shine_images=os.listdir(os.path.join(data_path,cls))
    else:
         sunrise_images=os.listdir(os.path.join(data_path,cls))

print("*"*30)


print("Total cloudy images is",len(cloudy_images))
print("Total foggy  images is",len(foggy_images))
print("Total rainy images is",len(rainy_images))
print("Total shine images is",len(shine_images))
print("Total sunrise images is",len(sunrise_images))

no_of_classes={'cloudy_images':len(cloudy_images),'foggy_images':len(foggy_images),
'rainy':len(rainy_images),'shine':len(shine_images),'sunrise':len(sunrise_images)}

print("*"*30)


fig, ax = plt.subplots( nrows=1, ncols=1,figsize=(30,40))
ax.bar(no_of_classes.keys(), no_of_classes.values(), width = .20)
plt.savefig('images/img1.png')
plt.title("Number of Images by Class");
plt.xlabel('Class Name')
plt.ylabel('# Images')

'''
##################################  Cloudy Images ###################################################
cloudy_sample = []
for file in cloudy_images[:10]: 
    image = cv2.imread(os.path.join(data_path,'cloudy',file))
    image_array = cv2.resize(image ,dsize=(128,128))
    cloudy_sample.append(list(image_array))
    
plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(cloudy_sample),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(cloudy_sample[i])   
    plt.axis('off')
    plt.title("Cloudy")
    plt.savefig('images/Cloudy.png')

############################################ foggy  images #############################################

foggy_sample = []
for file in foggy_images[:10]: 
    image = cv2.imread(os.path.join(data_path,'foggy',file))
    image_array = cv2.resize(image ,dsize=(128,128))
    foggy_sample.append(list(image_array))
    
plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(foggy_sample),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(foggy_sample[i])   
    plt.axis('off')
    plt.title("foggy")
    plt.savefig('images/foggy.png')

############################################ rainy  images #############################################

rainy_sample = []
for file in rainy_images[:10]: 
    image = cv2.imread(os.path.join(data_path,'rainy',file))
    image_array = cv2.resize(image ,dsize=(128,128))
    rainy_sample .append(list(image_array))
    
plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(rainy_sample),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(rainy_sample[i])   
    plt.axis('off')
    plt.title("rainy")
    plt.savefig('images/rainy.png')

############################################ shine  images #############################################
shine_sample = []
for file in shine_images[:10]: 
    image = cv2.imread(os.path.join(data_path,'shine',file))
    image_array = cv2.resize(image ,dsize=(128,128))
    shine_sample .append(list(image_array))
    
plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(shine_sample),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(shine_sample[i])   
    plt.axis('off')
    plt.title("shine")
    plt.savefig('images/shine.png')

############################################ sunrise  images #############################################

sunrise_sample = []
for file in sunrise_images[:10]: 
    image = cv2.imread(os.path.join(data_path,'sunrise',file))
    image_array = cv2.resize(image ,dsize=(128,128))
    sunrise_sample .append(list(image_array))
    
plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(sunrise_sample),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(sunrise_sample[i])   
    plt.axis('off')
    plt.title("shine")
    plt.savefig('images/sunrise.png')

'''
########################### Average Size of the images of Cloudy  #################################
width=[]
height=[]
for file in cloudy_images: 
    image = cv2.imread(os.path.join(data_path,'cloudy',file))
    width.append(image.shape[0])
    height.append(image.shape[1])
avg_width=np.average(width)
avg_height=np.average(height)
print("The avearge width & height of cloudy images is",(avg_width.round(),avg_height.round()))    
    
########################### Average Size of the images of foggy  #################################

width=[]
height=[]
for file in foggy_images: 
    image = cv2.imread(os.path.join(data_path,'foggy',file))
    width.append(image.shape[0])
    height.append(image.shape[1])
avg_width=np.average(width)
avg_height=np.average(height)
print("The avearge width & height of foggy images is",(avg_width.round(),avg_height.round()))    
    
########################### Average Size of the images of shine #################################
width=[]
height=[]
for file in shine_images: 
    image = cv2.imread(os.path.join(data_path,'shine',file))
    if image is not None:
       width.append(image.shape[0])
       height.append(image.shape[1])
avg_width=np.average(width)
avg_height=np.average(height)
print("The avearge width & height of shine images is",(avg_width.round(),avg_height.round())) 

########################### Average Size of the images of rainy #################################
width=[]
height=[]
for file in rainy_images: 
    image = cv2.imread(os.path.join(data_path,'rainy',file))
    if image is not None:
       width.append(image.shape[0])
       height.append(image.shape[1])
avg_width=np.average(width)
avg_height=np.average(height)
print("The avearge width & height of rainy images is",(avg_width.round(),avg_height.round())) 

########################### Average Size of the images of Sunrise #################################

width=[]
height=[]
for file in sunrise_images: 
    image = cv2.imread(os.path.join(data_path,'sunrise',file))
    if image is not None:
       width.append(image.shape[0])
       height.append(image.shape[1])
avg_width=np.average(width)
avg_height=np.average(height)
print("The avearge width & height of sunrise images is",(avg_width.round(),avg_height.round())) 

########################### Distriutions of the images of Cloudy #################################
cloud_dist=[]
for file in cloudy_images: 
    image = cv2.imread(os.path.join(data_path,'cloudy',file))
    cloud_dist.append(image)
fig, ax = plt.subplots( nrows=1, ncols=1,figsize=(30,40))
plt.hist(np.array(cloud_dist).ravel())
plt.savefig('images/cloud_dist.jpg')

########################### Distriutions of the images of foggy #################################

foggy_dist=[]
for file in foggy_images: 
    image = cv2.imread(os.path.join(data_path,'foggy',file))
    foggy_dist.append(image)
fig, ax = plt.subplots( nrows=1, ncols=1,figsize=(30,40))
plt.hist(np.array(foggy_dist).ravel())
plt.savefig('images/foggy_dist.jpg')

########################### Distriutions of the images of rainy #################################

rainy_dist=[]
for file in rainy_images: 
    image = cv2.imread(os.path.join(data_path,'rainy',file))
    if image is not None:
       rainy_dist.append(image)
fig, ax = plt.subplots( nrows=1, ncols=1,figsize=(30,40))
plt.hist(np.array(rainy_dist).ravel())
plt.savefig('images/rainy_dist.jpg')

########################### Distriutions of the images of shine #################################

shine_dist=[]
for file in shine_images: 
    image = cv2.imread(os.path.join(data_path,'shine',file))
    if image is not None:
       shine_dist.append(image)
fig, ax = plt.subplots( nrows=1, ncols=1,figsize=(30,40))
plt.hist(np.array(shine_dist).ravel())
plt.savefig('images/shine_dist.jpg')

########################### Distriutions of the images of sunrise #################################

sunrise_dist=[]
for file in sunrise_images: 
    image = cv2.imread(os.path.join(data_path,'sunrise',file))
    if image is not None:
       sunrise_dist.append(image)
fig, ax = plt.subplots( nrows=1, ncols=1,figsize=(30,40))
plt.hist(np.array(sunrise_dist).ravel())
plt.savefig('images/sunrise_dist.jpg')












