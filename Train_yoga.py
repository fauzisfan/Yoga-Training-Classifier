# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 18:21:11 2021

@author: knum
"""

# keras imports for the dataset and building our neural network
import numpy as np
import time
import pandas as pd
import tensorflow as tf
from keras import regularizers, models, layers
from scipy.stats import spearmanr
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.utils import to_categorical
import joblib
from sklearn.model_selection import train_test_split

def preparets(x, y, num_keypoints):
    assert 0 < num_keypoints < x.shape[1]
    x_seq       = np.atleast_3d(np.array([x[:,start*num_keypoints:(start*num_keypoints+num_keypoints)] for start in range(0, int(x.shape[1]/num_keypoints))]))
    y_seq       = np.atleast_1d(np.array([y[start*num_keypoints] for start in range(0, int(y.shape[0]/num_keypoints))]))
    return x_seq, y_seq

num_keypoints = 17
num_content = 2

def getData(filename, lab, tar):
    # read csv for standing pose
    try:
    	df_tracefile = pd.read_csv(filename)
    	frame = np.array(df_tracefile[['frame']], dtype=np.int)
    	pose = np.array(df_tracefile[['pose_score']], dtype=np.float32)
    	body_part = np.array(df_tracefile[['part_score']], dtype=np.float32)
    	x_coord = np.array(df_tracefile[['x_coord']], dtype=np.float32)
    	y_coord = np.array(df_tracefile[['y_coord']], dtype=np.float32)
    	label = np.array(df_tracefile[['label']], dtype=np.str)
    except:
    	raise
    
    '''Pre-Processing Data'''
    data = np.array([pose, body_part,x_coord,y_coord])
    
    x_seq, y_target = preparets(data[:,:,0], label[:,0], num_keypoints)
    _, frame = preparets(data[:,:,0], frame[:,0], num_keypoints)
    
    #For coba1.csv data
    # std = np.array(list(range(1, 109)))
    # bow = np.array(list(range(181, 270)))#181, 270
    # right = np.array(list(range(330, 415)))
    # left = np.array(list(range(544, 593)))
    
    lis = np.array(list(range(0, len(label))))
    
    lab_target = np.zeros(len(y_target), dtype=np.float32)
    
    for i in range(len(frame)):
        #For stand label
        if (len(np.where(frame[i]==lis)[0])>0):
            y_target[i] = label[i][0]
            lab_target[i] = tar
    
    return y_target, lab_target, x_seq

y_target, lab_target, x_seq = getData(filename='coba_none.csv', lab='none', tar='0')
y_target21, lab_target21, x_seq21 = getData(filename='coba_std1.csv', lab='std', tar='1')
y_target22, lab_target22, x_seq22 = getData(filename='coba_std2.csv', lab='std', tar='1')
y_target23, lab_target23, x_seq23 = getData(filename='coba_std3.csv', lab='std', tar='1')
y_target31, lab_target31, x_seq31 = getData(filename='coba_str1.csv', lab='str', tar='2')
y_target32, lab_target32, x_seq32 = getData(filename='coba_str2.csv', lab='str', tar='2')
y_target33, lab_target33, x_seq33 = getData(filename='coba_str3.csv', lab='str', tar='2')
y_target41, lab_target41, x_seq41 = getData(filename='coba_rgt1.csv', lab='rgt', tar='3')
y_target42, lab_target42, x_seq42 = getData(filename='coba_rgt2.csv', lab='rgt', tar='3')
y_target43, lab_target43, x_seq43 = getData(filename='coba_rgt3.csv', lab='rgt', tar='3')
y_target51, lab_target51, x_seq51 = getData(filename='coba_lft1.csv', lab='lft', tar='4')
y_target52, lab_target52, x_seq52 = getData(filename='coba_lft2.csv', lab='lft', tar='4')
y_target53, lab_target53, x_seq53 = getData(filename='coba_lft3.csv', lab='lft', tar='4')

# Combine all the data
lab_target_fin = to_categorical(np.concatenate((lab_target,
                                                lab_target21, lab_target22, lab_target23,
                                                lab_target31, lab_target32, lab_target33,
                                                lab_target41, lab_target42, lab_target43,
                                                lab_target51, lab_target52, lab_target53), axis=0))
x_seq_fin = np.concatenate((x_seq,
                            x_seq21, x_seq22, x_seq23,
                            x_seq31, x_seq32, x_seq33,
                            x_seq41, x_seq42, x_seq43,
                            x_seq51, x_seq52, x_seq53), axis=0)
X_input = x_seq_fin[:,4-num_content::,:].reshape(len(x_seq_fin),num_keypoints*(4-num_content))
y_target_fin = np.concatenate((y_target,
                               y_target21, y_target22, y_target23,
                               y_target31, y_target32, y_target33,
                               y_target41, y_target42, y_target43,
                               y_target51, y_target52, y_target53), axis=0)

normalizer = preprocessing.StandardScaler()
tempNorm = normalizer.fit(X_input)

scaler_file = "yoga_scaller.save"
joblib.dump(tempNorm, scaler_file) 

X_input = tempNorm.transform(X_input)

X_train, X_test, lab_train, lab_test = train_test_split(X_input, lab_target_fin, test_size=0.7, random_state=42)
_, _, y_train, y_test = train_test_split(X_input, y_target_fin, test_size=0.7, random_state=42)

#Implement Neural Network Here!
'''Neural Network'''
TRAINED_MODEL_NAME = "./model/yoga_net"

# # Reset the whole tensorflow graph
tf.get_default_graph()

model = Sequential()
model.add(Dense(50, activation='relu', input_dim=num_keypoints*(4-num_content)))
model.add(Dense(20, activation='relu'))
# model.add(Dense(10, activation='relu'))
model.add(Dense(len(lab_target_fin[0]), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

history = model.fit(X_train, lab_train, epochs=50)

# # summarize history for accuracy
# plt.figure()
# plt.plot(history.history['acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['Accuracy'], loc='upper left')
# plt.show()

# # summarize history for loss
# plt.figure()
# plt.plot(history.history['loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['loss'], loc='upper left')
# plt.show()

pred_train= model.predict(X_train)
scores = model.evaluate(X_train, lab_train, verbose=0)
print('Accuracy on training data: {}% \nError on training data: {}'.format(scores[1]*100, 1 - scores[1]))   

# serialize model to JSON
model_json = model.to_json()
with open('./model/'+'yoga_net.json', "w") as json_file:
    json_file.write(model_json)
    
# serialize weights to HDF5
model.save_weights(TRAINED_MODEL_NAME)
print("Saved model to disk")

# load the saved model
with open('./model/'+'yoga_net.json', 'r') as arch_file:
    loaded_model = model_from_json(arch_file.read())

# load weights into new model
loaded_model.load_weights(TRAINED_MODEL_NAME)
print("\nLoaded model from disk")

# Compile the model
loaded_model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
print("Model Compiled")

pred_test = np.empty((0,len(lab_test[0])),dtype=np.float)

for i in range(len(X_test)):
    pred_test = np.concatenate((pred_test, loaded_model.predict(X_test[i].reshape(1,-1))), axis=0)
    
# pred_test = loaded_model.predict(X_train)

y_pred = np.array([],dtype=np.str)

for i in range(len(pred_test)):
    val = np.where(pred_test[i]==max(pred_test[i]))
    # print(val)
    if (val[0]==1):
        y_pred = np.concatenate((y_pred, np.array(['std'])), axis = 0)
    elif(val[0]==2):
        y_pred = np.concatenate((y_pred, np.array(['str'])), axis = 0)
    elif(val[0]==3):
        y_pred = np.concatenate((y_pred, np.array(['rgt'])), axis = 0)
    elif(val[0]==4):
        y_pred = np.concatenate((y_pred, np.array(['lft'])), axis = 0)
    else:
        y_pred = np.concatenate((y_pred, np.array(['none'])), axis = 0)

scores = np.sum(y_pred==y_test)/len(y_pred)
print('Accuracy on testing data: {}% \nError on testing data: {}'.format(scores*100, 1 - scores))  

'''Parameters Coorelation'''
# #spearmanr method
# for i in range(num_keypoints):
#     corr, _ = spearmanr(x_seq[:,1,i], lab_target)
#     print('Spearmans correlation: %.3f' % corr)

# #seaborn method
# sns.set_theme(style="white")

# # Generate a large random dataset
# d = pd.DataFrame(data=x_seq21[:,:,0].T,
#                   columns=[
#                       'frame',
#                       'pose_score',
#                       'part_score',
#                       'x_coord',
#                       'y_coord',
#                       ])

# # Compute the correlation matrix
# corr = d.corr()

# # Generate a mask for the upper triangle
# mask = np.triu(np.ones_like(corr, dtype=bool))

# # Set up the matplotlib figure
# f, ax = plt.subplots(figsize=(11, 9))

# # Generate a custom diverging colormap
# cmap = sns.diverging_palette(230, 20, as_cmap=True)

# # Draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
#             square=True, linewidths=.5, cbar_kws={"shrink": .5})
