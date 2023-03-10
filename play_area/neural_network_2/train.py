# importing the libraries
import pandas as pd
import numpy as np

# for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt
import pickle

# for creating validation set
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os
# for evaluating the model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import torchvision.transforms as transforms
from CustomImageDataset import CustomImageDataset
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
lidar_data_filename = f"../../image_data/collision_data.pkl"
N_EPOCHS = 300
LR = 0.0005
BATCH_SIZE = 64
data = read_data_from_pickle(lidar_data_filename)

not_done_indices = data.index[data['done_collision'] == False].tolist()
not_done_indices = not_done_indices[:3*int(len(not_done_indices)/4)]

data = data.drop(not_done_indices)

data_x = data['filename']
data_y = data['done_collision'].astype(int)



X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3, shuffle=True, stratify=data['done_collision'])


print(f"Y_train Counts \n{y_train.value_counts()}")
print(f"Y_test Counts \n{y_test.value_counts()}")


training_data = pd.concat([X_train,y_train],axis=1)
testing_data = pd.concat([X_test,y_test],axis=1)

training_dataset = CustomImageDataset(annotations_file=training_data,img_dir="../../image_data/lidar/")
testing_dataset = CustomImageDataset(annotations_file=testing_data,img_dir="../../image_data/lidar/")

trainLoader = torch.utils.data.DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
testLoader = torch.utils.data.DataLoader(testing_dataset, batch_size=1,shuffle=True, num_workers=2)

# dataiter = iter(trainLoader)
# images, labels = next(dataiter)
#
# plt.figure()
# xy_res = np.array(images[0][0]).shape
# plt.imshow(images[0][0], cmap="PiYG_r")
# # cmap = "binary" "PiYG_r" "PiYG_r" "bone" "bone_r" "RdYlGn_r"
# plt.clim(-0.4, 1.4)
# plt.gca().set_xticks(np.arange(-.5, xy_res[1], 1), minor=True)
# plt.gca().set_yticks(np.arange(-.5, xy_res[0], 1), minor=True)
# plt.grid(True, which="minor", color="w", linewidth=0.6, alpha=0.5)
# # plt.gca().invert_yaxis()
# plt.show()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


# defining the model
model = Net()
# defining the optimizer
optimizer = Adam(model.parameters(), lr=LR)
# defining the loss function
criterion = CrossEntropyLoss()
# checking if GPU is available
model = model.to(device)
criterion = criterion.to(device)


print(model)

totalLoss = []
for epoch in range(N_EPOCHS):  # loop over the dataset multiple times

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
        totalLoss.append(running_loss)
        if i % 10== 9:  # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
            running_loss = 0.0


print('Finished Training')

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(totalLoss,label="train_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Per batch")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(f"{lidar_data_filename}_{N_EPOCHS}_{LR}_{BATCH_SIZE}.png")
# serialize the model to disk
torch.save(model, f"{lidar_data_filename}_{N_EPOCHS}_{LR}_{BATCH_SIZE}.model")


test_predictions = []
true_test_values = []
softmax_list = []

def softmax(z):
    '''Return the softmax output of a vector.'''
    exp_z = np.exp(z)
    sum = exp_z.sum()
    softmax_z = np.round(exp_z/sum,3)
    return softmax_z

# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testLoader:
        images, labels = data[0].type(torch.cuda.FloatTensor).to(device), data[1].to(device)
        # calculate outputs by running images through the network
        outputs = model(images)

        softmax_list.append(softmax(outputs.data.cpu().numpy()))
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)

        test_predictions.append(predicted.cpu().numpy())
        true_test_values.append(labels.cpu().numpy())

print(f'Softmax  {softmax_list}')
print(f"Confusion Matrix {confusion_matrix(true_test_values,test_predictions)}")
testing_pred_output = evaluate("Testing", true_test_values, test_predictions)
save_to_file(f"{lidar_data_filename}_{N_EPOCHS}_{LR}_{BATCH_SIZE}", testing_pred_output + '\n\n' + str(softmax_list))
print(testing_pred_output)