#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


from sklearn.datasets import fetch_olivetti_faces
data = fetch_olivetti_faces()


# In[4]:


data.keys()


# In[7]:


print("Data Shape:", data.data.shape)
print("Target Shape:", data.target.shape)
print("There are {} unique persons in the dataset".format(len(np.unique(data.target))))
print("Size of each image is {}x{}".format(data.images.shape[1],data.images.shape[1]))


# In[12]:


def print_faces(images, target, top_n):
    top_n = min(top_n, len(images))
 # Set up figure size based on the number of images
    grid_size = int(np.ceil(np.sqrt(top_n)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.2, wspace=0.2)
    for i, ax in enumerate(axes.ravel()):
        if i < top_n:
            ax.imshow(images[i], cmap='bone')
            ax.axis('off')
            ax.text(2, 12, str(target[i]), fontsize=9, color='red')
            ax.text(2, 55, f"face: {i}", fontsize=9, color='blue')
        else:
            ax.axis('off')
            plt.show()
print_faces(data.images,data.target,400)


# In[16]:


def display_unique_faces(pics):
    fig = plt.figure(figsize=(24, 10)) # Set figure size
    columns, rows = 10, 4 # Define grid dimensions
 # Loop through grid positions and plot each image
    for i in range(1, columns * rows + 1):
        img_index = 10 * i - 1 # Calculate the image index
        if img_index < pics.shape[0]: # Check for valid image index
            img = pics[img_index, :, :]
            ax = fig.add_subplot(rows, columns, i)
            ax.imshow(img, cmap='gray')
            ax.set_title(f"Person {i}", fontsize=14)
            ax.axis('off')
        plt.suptitle("There are 40 distinct persons in the dataset", fontsize=24)
plt.show()
display_unique_faces(data.images)


# In[17]:


from sklearn.model_selection import train_test_split
X = data.data
Y = data.target
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, 
random_state=46)


# In[18]:


print("x_train: ",x_train.shape)
print("x_test: ",x_test.shape)


# In[19]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score


# In[20]:


nb = GaussianNB()
nb.fit(x_train, y_train)


# In[21]:


y_pred = nb.predict(x_test)


# In[22]:


nb_accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)


# In[24]:


cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print(f"Naive Bayes Accuracy: {nb_accuracy}%")


# In[26]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# In[27]:


nb = MultinomialNB()
nb.fit(x_train, y_train)


# In[28]:


y_pred = nb.predict(x_test)


# In[29]:


accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
print(f"Multinomial Naive Bayes Accuracy: {accuracy}%")


# In[30]:


misclassified_idx = np.where(y_pred != y_test)[0]
num_misclassified = len(misclassified_idx)


# In[31]:


print(f"Number of misclassified images: {num_misclassified}")
print(f"Total images in test set: {len(y_test)}")
print(f"Accuracy: {round((1 - num_misclassified / len(y_test)) * 100, 2)}%")


# In[33]:


n_misclassified_to_show = min(num_misclassified, 5) 
plt.figure(figsize=(10, 5))
for i in range(n_misclassified_to_show):
    idx = misclassified_idx[i]
    plt.subplot(1, n_misclassified_to_show, i + 1)
    plt.imshow(x_test[idx].reshape(64, 64), cmap='gray')
    plt.title(f"True: {y_test[idx]}, Pred: {y_pred[idx]}")
    plt.axis('off')
plt.show()


# In[34]:


from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score


# In[35]:


y_test_bin = label_binarize(y_test, classes=np.unique(y_test))


# In[36]:


y_pred_prob = nb.predict_proba(x_test)


# In[37]:


for i in range(y_test_bin.shape[1]):
    roc_auc = roc_auc_score(y_test_bin[:, i], y_pred_prob[:, i])
    print(f"Class {i} AUC: {roc_auc:.2f}")


# In[ ]:




