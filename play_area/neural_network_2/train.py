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
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from tqdm import tqdm

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import torchvision.transforms as transforms
from play_area.neural_network_2.CustomImageDataset import CustomImageDataset
from Model.Model import Net

def read_data_from_pickle(filename):
    with open(filename, 'rb') as handle:
        print(filename)
        return pickle.load(handle)

def save_data( filename, data):
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def evaluate(note, actual, predictions):
    return f"""
    {note}
    accuracy_score: {accuracy_score(actual, predictions)}
    precision_score: {precision_score(actual, predictions)}
    recall_score: {recall_score(actual, predictions)}
    f1_score: {f1_score(actual, predictions)}
    """

def save_to_file(filename, data):
    with open(filename,'w') as f:
        f.write(data)



no_of_lidar_images = 100
lidar_data_filename = f"lidar_data_{no_of_lidar_images}.pkl"
n_epochs = 2
lr = 0.0005
batch_size = 2

if not os.path.isfile(lidar_data_filename):

    data = read_data_from_pickle('../../image_data/collision_data_2.pkl')
    data = data.loc[:no_of_lidar_images]
    data['lidar_image'] = data.loc[:, 'filename']


    data['lidar_image'] = data['lidar_image'].apply(lambda x: read_data_from_pickle(f"../../image_data/lidar/{x}.pkl") )
    # for index, row in data.iterrows():
    #     path = f"../../image_data/lidar/{row['filename']}.pkl"
    #     row['image'] = read_data_from_pickle(path)
    #
    #     print(index)

    print('Saving File ....')
    save_data(lidar_data_filename,data)

else:
    print('Reading data ....')
    data = read_data_from_pickle(lidar_data_filename)
    print('Data Read')




for index, row in data.iterrows():
    row['filename'] ="03072023_212450443245"

data_x = data['filename']
data_y = data['done_collision'].astype(int)


X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3, shuffle=True, stratify=data['done_collision'])

training_data = pd.concat([X_train,y_train],axis=1)
testing_data = pd.concat([X_test,y_test],axis=1)

training_dataset = CustomImageDataset(annotations_file=training_data,img_dir="../../image_data/lidar/")
testing_dataset = CustomImageDataset(annotations_file=testing_data,img_dir="../../image_data/lidar/")

trainLoader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size,shuffle=True, num_workers=2)
testLoader = torch.utils.data.DataLoader(testing_dataset, batch_size=1,shuffle=True, num_workers=2)

# dataiter = iter(trainLoader)
# images, labels = next(dataiter)
#
#
# plt.figure()
# xy_res = np.array(images).shape
# plt.imshow(images, cmap="PiYG_r")
# # cmap = "binary" "PiYG_r" "PiYG_r" "bone" "bone_r" "RdYlGn_r"
# plt.clim(-0.4, 1.4)
# plt.gca().set_xticks(np.arange(-.5, xy_res[1], 1), minor=True)
# plt.gca().set_yticks(np.arange(-.5, xy_res[0], 1), minor=True)
# plt.grid(True, which="minor", color="w", linewidth=0.6, alpha=0.5)
# # plt.gca().invert_yaxis()
# plt.show()



# defining the model
model = Net()
# defining the optimizer
optimizer = Adam(model.parameters(), lr=lr)
# defining the loss function
criterion = CrossEntropyLoss()
# checking if GPU is available
model = model.cuda()
criterion = criterion.cuda()


print(model)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
for epoch in range(n_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainLoader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].type(torch.cuda.FloatTensor).to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        running_loss = 0.0

print('Finished Training')

test_predictions = []
true_test_values = []
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testLoader:
        images, labels = data[0].type(torch.cuda.FloatTensor).to(device), data[1].to(device)
        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        softmax = torch.exp(outputs.data).cpu()
        prob = list(softmax.numpy())
        pred = np.argmax(prob, axis=1)

        test_predictions.append(pred)
        true_test_values.append(labels.cpu().numpy())


testing_pred_output = evaluate("Training", true_test_values, test_predictions)
save_to_file(f"{lidar_data_filename}_{n_epochs}_{lr}_{batch_size}",testing_pred_output)
print(training_pred_output)