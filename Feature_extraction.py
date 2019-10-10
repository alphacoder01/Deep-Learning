#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torchvision import datasets
import PIL
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops
from skimage.filters import gabor
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


trainDset = datasets.CIFAR10('./cidar10/',train = True, download=True)
testDset = datasets.CIFAR10('./cifar10',train=False,download = True)


# In[9]:


print('No. of train samples : '+ str(len(trainDset)))
print('No. of test samples : '+ str(len(testDset)))


# In[10]:


img = trainDset[0][0]
img_gray = img.convert('L')
img_arr = np.array(img_gray)
plt.imshow(img)


# ## Local Binary Patterns

# In[13]:


feat_lbp = local_binary_pattern(img_arr,8,1,'uniform')#radius =1,
feat_lbp = np.uint8((feat_lbp/feat_lbp.max())*255) #converting to uint8
lbp_img = PIL.Image.fromarray(feat_lbp)#conversion from array to lbp image
plt.imshow(lbp_img,cmap='gray')


# In[14]:


lbp_hist,_ = np.histogram(feat_lbp,8)
lbp_hist = np.array(lbp_hist,dtype = float)
lbp_prob = np.divide(lbp_hist,np.sum(lbp_hist))
lbp_energy = np.sum(lbp_prob**2)
lbp_entropy = -np.sum(np.multiply(lbp_prob,np.log2(lbp_prob)))
print('LBP energy :'+str(lbp_energy))
print('LBP entropy :'+str(lbp_entropy))


# ## Co-occurance Matrix

# In[17]:


gCoMat = greycomatrix(img_arr, [2], [0], 256, symmetric=True,normed = True)
contrast = greycoprops(gCoMat, prop='contrast')
dissimilarity = greycoprops(gCoMat,prop = 'dissimilarity')
homogeneity = greycoprops(gCoMat, prop = 'homogeneity')
energy = greycoprops(gCoMat, prop= 'energy')
correlation = greycoprops(gCoMat, prop = 'correlation')
print('Contrast : '+str(contrast))
print('Dissimilarity : '+str(dissimilarity))
print('homogeneity : '+str(homogeneity))
print('Energy : '+str(energy))
print('Correlation : '+str(correlation))


# ## Gabor filter

# In[18]:


gaborFilter_real , gaborFilter_imag = gabor(img_arr,frequency=0.6)
gaborFilter = (gaborFilter_real**2 + gaborFilter_imag**2)//2

fig,ax = plt.subplots(1,3)
ax[0].imshow(gaborFilter_real,cmap = 'gray')
ax[1].imshow(gaborFilter_imag,cmap = 'gray')
ax[2].imshow(gaborFilter,cmap = 'gray')


# In[19]:


gabor_hist,_ = np.histogram(gaborFilter,8)
gabor_hist = np.array(gabor_hist,dtype = float)
gabor_prob = np.divide(gabor_hist,np.sum(gabor_hist))
gabor_energy = np.sum(gabor_prob**2)
gabor_entropy = -np.sum(np.multiply(gabor_prob,np.log2(gabor_prob)))
print('Gabor energy : '+str(gabor_energy))
print('Gabor entropy : '+str(gabor_entropy))


# ## Extracting features from the train dataset

# In[21]:


label = []
featLength = 2+5+2
trainFeats = np.zeros((len(trainDset),featLength))
for tr in range(len(trainDset)):
    print(str(tr+1) + '/' +str(len(trainDset)))
    img = trainDset[tr][0] #taking one image at a time
    img_gray = img.convert('L') #convert to grayscale
    img_arr = np.array(img_gray.getdata()).reshape(img.size[1],img.size[0])
    
    feat_lbp = local_binary_pattern(img_arr,5,2,'uniform').reshape(img.size[1],img.size[0])
    lbp_hist,_ = np.histogram(feat_lbp,8)
    lbp_hist = np.array(lbp_hist,dtype = float)
    lbp_prob = np.divide(lbp_hist,np.sum(lbp_hist))
    lnp_energy = np.nansum(lbp_prob**2)
    lbp_entropy = -np.nansum(np.multiply(lbp_prob,np.log2(lbp_prob)))
    
    gCoMat = greycomatrix(img_arr, [2], [0] , 256, symmetric=True, normed=True)
    contrast = greycoprops(gCoMat, prop='contrast')
    dissimilarity = greycoprops(gCoMat,prop = 'dissimilarity')
    homogeneity = greycoprops(gCoMat, prop = 'homogeneity')
    energy = greycoprops(gCoMat, prop= 'energy')
    correlation = greycoprops(gCoMat, prop = 'correlation')
    feat_glcm = np.array([contrast[0][0], dissimilarity[0][0], homogeneity[0][0], energy[0][0], correlation[0][0]])
    
    gaborFilter_real , gaborFilter_imag = gabor(img_arr,frequency=0.6)
    gaborFilter = (gaborFilter_real**2 + gaborFilter_imag**2)//2
    gabor_hist,_ = np.histogram(gaborFilter,8)
    gabor_hist = np.array(gabor_hist,dtype = float)
    gabor_prob = np.divide(gabor_hist,np.sum(gabor_hist))
    gabor_energy = np.sum(gabor_prob**2)
    gabor_entropy = -np.sum(np.multiply(gabor_prob,np.log2(gabor_prob)))
    
    concat_feat = np.concatenate(([lbp_energy,lbp_entropy],feat_glcm,[gabor_energy,gabor_entropy]))
    trainFeats[tr,:] = concat_feat
    
    label.append(trainDset[tr][1])

trainLabel = np.array(label)


# ## Extracting features from the test dataset

# In[24]:


label = []
featLength = 2+5+2
testFeats = np.zeros((len(testDset),featLength))
for ts in range(len(testDset)):
    print(str(ts+1) + '/' +str(len(testDset)))
    img = testDset[ts][0] #taking one image at a time
    img_gray = img.convert('L') #convert to grayscale
    img_arr = np.array(img_gray.getdata()).reshape(img.size[1],img.size[0])
    
    feat_lbp = local_binary_pattern(img_arr,5,2,'uniform').reshape(img.size[1],img.size[0])
    lbp_hist,_ = np.histogram(feat_lbp,8)
    lbp_hist = np.array(lbp_hist,dtype = float)
    lbp_prob = np.divide(lbp_hist,np.sum(lbp_hist))
    lnp_energy = np.nansum(lbp_prob**2)
    lbp_entropy = -np.nansum(np.multiply(lbp_prob,np.log2(lbp_prob)))
    
    gCoMat = greycomatrix(img_arr, [2], [0] , 256, symmetric=True, normed=True)
    contrast = greycoprops(gCoMat, prop='contrast')
    dissimilarity = greycoprops(gCoMat,prop = 'dissimilarity')
    homogeneity = greycoprops(gCoMat, prop = 'homogeneity')
    energy = greycoprops(gCoMat, prop= 'energy')
    correlation = greycoprops(gCoMat, prop = 'correlation')
    feat_glcm = np.array([contrast[0][0], dissimilarity[0][0], homogeneity[0][0], energy[0][0], correlation[0][0]])
    
    gaborFilter_real , gaborFilter_imag = gabor(img_arr,frequency=0.6)
    gaborFilter = (gaborFilter_real**2 + gaborFilter_imag**2)//2
    gabor_hist,_ = np.histogram(gaborFilter,8)
    gabor_hist = np.array(gabor_hist,dtype = float)
    gabor_prob = np.divide(gabor_hist,np.sum(gabor_hist))
    gabor_energy = np.sum(gabor_prob**2)
    gabor_entropy = -np.sum(np.multiply(gabor_prob,np.log2(gabor_prob)))
    
    concat_feat = np.concatenate(([lbp_energy,lbp_entropy],feat_glcm,[gabor_energy,gabor_entropy]))
    testFeats[ts,:] = concat_feat
    
    label.append(testDset[ts][1])

testLabel = np.array(label)


# ## Feature Normalization

# In[25]:


trMaxs = np.amax(trainFeats, axis = 0)  #Finding max along each column
trMins = np.amin(trainFeats, axis = 0)  #Finding min along each column
trMaxs_rep = np.tile(trMaxs,(50000,1))   #Repeating the maximum value along the column
trMins_rep = np.tile(trMins , (50000,1)) 
trainFeatsNorm = np.divide(trainFeats-trMins_rep,trMaxs_rep)

#normalozong the test features
tsMaxs_rep = np.tile(trMaxs,(10000,1))
tsMins_rep = np.tile(trMins,(10000,1))
testFeatsNorm = np.divide(testFeats - tsMins_rep,tsMaxs_rep)


# In[28]:


import pandas as pd
import pickle


# Saving feature matrices to diisk

# In[30]:


with open('trainFeats.pckl','wb') as f:
    pickle.dump(trainFeatsNorm,f)
with open('trainLabel.pckl' ,'wb') as f:
    pickle.dump(trainLabel,f)

with open('testFeats.pckl','wb') as f:
    pickle.dump(testFeatsNorm,f)
with open('testLabel.pckl','wb') as f:
    pickle.dump(testLabel,f)

print('Files saved to disk!')
    


# In[31]:


x = pd.read_pickle('./trainFeats.pckl')


# In[32]:


x


# In[ ]:




