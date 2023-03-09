# importing the libraries
import pandas as pd
import numpy as np

# for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt
import pickle

# for creating validation set
from sklearn.model_selection import train_test_split
import os
# for evaluating the model
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD


def read_data_from_pickle(filename):
    with open(filename, 'rb') as handle:
        print(filename)
        return pickle.load(handle)

def save_data( filename, data):
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)



# # loading dataset
# train = pd.read_csv('train_LbELtWX/train.csv')
# test = pd.read_csv('test_ScVgIM0/test.csv')
#
# sample_submission = pd.read_csv('sample_submission_I5njJSF.csv')

if not os.path.isfile("lidar_data.pkl"):

    data = read_data_from_pickle('../../image_data/collision_data_2.pkl')
    data = data.loc[:10]
    data['lidar_image'] = data.loc[:, 'filename']


    data['lidar_image'] = data['lidar_image'].apply(lambda x: read_data_from_pickle(f"../../image_data/lidar/{x}.pkl") )
    # for index, row in data.iterrows():
    #     path = f"../../image_data/lidar/{row['filename']}.pkl"
    #     row['image'] = read_data_from_pickle(path)
    #
    #     print(index)

    print('Saving File ....')
    save_data('lidar_data.pkl',data)

else:
    print('Reading data ....')
    data = read_data_from_pickle(f"lidar_data.pkl")
    print('Data Read')

data.head()

# # loading training images
# train_img = []
# for img_name in tqdm(train['id']):
#     # defining the image path
#     image_path = 'train_LbELtWX/train/' + str(img_name) + '.png'
#     # reading the image
#     img = imread(image_path, as_gray=True)
#     # normalizing the pixel values
#     img /= 255.0
#     # converting the type of pixel to float 32
#     img = img.astype('float32')
#     # appending the image into the list
#     train_img.append(img)

# converting the list to numpy array
train_x = np.array(data['lidar_image'].values.tolist())
# defining the target
train_y = data['done_collision'].values
print(train_x.shape)


# # visualizing images
# i = 0
# plt.figure(figsize=(10,10))
# plt.subplot(221), plt.imshow(train_x[i], cmap='gray')
# plt.subplot(222), plt.imshow(train_x[i+25], cmap='gray')
# plt.subplot(223), plt.imshow(train_x[i+50], cmap='gray')
# plt.subplot(224), plt.imshow(train_x[i+75], cmap='gray')

# create validation set
train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size = 0.3, shuffle=True, stratify=train_y)
(train_x.shape, train_y.shape), (test_x.shape, test_y.shape)


# converting training images into torch format
train_x = train_x.reshape(len(train_x), 1, 240, 320)
train_x  = torch.from_numpy(train_x)

# converting the target into torch format
train_y = train_y.astype(int);
train_y = torch.from_numpy(train_y)

# shape of training data
print(train_x.shape)
print(train_y.shape)

# converting validation images into torch format
test_x = test_x.reshape(len(test_x), 1, 240, 320)
test_x  = torch.from_numpy(test_x)

# converting the target into torch format
test_y = test_y.astype(int)
test_y = torch.from_numpy(test_y)

# shape of validation data
print(test_x.shape)
print(test_y.shape)

from Model.Model import Net
# defining the model
model = Net()
# defining the optimizer
optimizer = Adam(model.parameters(), lr=0.07)
# defining the loss function
criterion = CrossEntropyLoss()
# checking if GPU is available
model = model.cuda()
criterion = criterion.cuda()


print(model)


def train(epoch):
    model.train()
    tr_loss = 0
    # getting the training set
    x_train, y_train = Variable(train_x), Variable(train_y)
    x_train = x_train.cuda()

    y_train = y_train.cuda()
    y_train = y_train.float()
    # # getting the validation set
    # x_val, y_val = Variable(val_x), Variable(val_y)
    # converting the data into GPU format


        # x_val = x_val.cuda()
        # y_val = y_val.cuda()

    # clearing the Gradients of the model parameters
    optimizer.zero_grad()

    # prediction for training and validation set
    output_train = model(x_train)
    # output_val = model(x_val)

    # computing the training and validation loss
    loss_train = criterion(output_train, y_train)
    # loss_val = criterion(output_val, y_val)
    train_losses.append(loss_train)
    # val_losses.append(loss_val)

    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()
    print(f"Epoch:{epoch}")
    if epoch % 2 == 0:
        # printing the validation loss
        print('Epoch : ', epoch + 1, '\t', 'loss :', tr_loss)


# defining the number of epochs
n_epochs = 25
# empty list to store training losses
train_losses = []
# # empty list to store validation losses
# val_losses = []
# training the model
for epoch in range(n_epochs):
    train(epoch)


# plotting the training and validation loss
plt.plot(train_losses, label='Training loss')
# plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.show()

# prediction for training set
with torch.no_grad():
    output = model(train_x.cuda())

softmax = torch.exp(output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)

# accuracy on training set
print(f"Accuracy Score training set {accuracy_score(train_y, predictions)}")



# TESTING SET


# generating predictions for test set
with torch.no_grad():
    output = model(test_x.cuda())

softmax = torch.exp(output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)

# replacing the label with prediction
sample_submission['label'] = predictions
sample_submission.head()

# saving the file
sample_submission.to_csv('submission.csv', index=False)


# # prediction for validation set
# with torch.no_grad():
#     output = model(val_x.cuda())
#
# softmax = torch.exp(output).cpu()
# prob = list(softmax.numpy())
# predictions = np.argmax(prob, axis=1)
#
# # accuracy on validation set
# accuracy_score(val_y, predictions)