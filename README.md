# 1D-CNN-Pytorch-timer-series-classifier
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import time
from sklearn.metrics import confusion_matrix

# setting seed
def setup_seed(seed):
    torch.manual_seed(seed)  # setting seed for cpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # setting seed for gpu
        torch.cuda.manual_seed_all(seed)  # setting seed for multi gpu
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmard = False

setup_seed(666)

# Generating low frequency waveforms
low_freqs = np.random.normal(1, 0.1, size=500)  # Random low frequencies
phases = np.random.uniform(0, 2 * np.pi, size=500)  # Regular time sampling
tt = np.linspace(0, 10, 1000)
lf_signals = np.sin(np.outer(low_freqs, tt) + phases.reshape(-1,1))  
# Should give an array of shape (500,1000) ― 500 different low-frequency waveforms

# plt.plot(tt, lf_signals[1, :], color='red')
# plt.plot(tt, lf_signals[50, :], color='green')
# plt.plot(tt, lf_signals[99, :], color='blue')
# plt.title("numpy.sin()")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()

# Generating high frequency waveforms
high_freqs = np.random.normal(10, 0.1, size=500)  # Random higher frequencies
phases = np.random.uniform(0, 2 * np.pi, size=500)
tt = np.linspace(0, 10, 1000)
hf_signals = np.sin(np.outer(high_freqs, tt) + phases.reshape(-1, 1))  
# 500 high-frequency waveforms

# plt.plot(tt, hf_signals[1, :], color='red')
# plt.plot(tt, hf_signals[50, :], color='green')
# plt.plot(tt, hf_signals[99, :], color='blue')
# plt.title("numpy.sin()")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()

# Combine the low_ and high_ frequency waveforms into one dataset
X_data = np.vstack([lf_signals, hf_signals])
Y_data = np.hstack([np.zeros(500), np.ones(500)])  # 0 = ‘low’, 1=‘high’

# Building a 1D CNN model
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1000, out_channels=500, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=500, out_channels=50, kernel_size=1)

        self.fc1 = nn.Linear(in_features=50, out_features=40)
        self.fc2 = nn.Linear(in_features=40, out_features=30)
        self.out = nn.Linear(in_features=30, out_features=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.flatten(1)  # flatten the tensor starting at dimension 1

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x

cnn_cuda = Network()
if torch.cuda.is_available():
    cnn_cuda=cnn_cuda.cuda()

if __name__ == '__main__':
    cnn = Network()
    input = torch.ones(1000, 1000, 1)
    output = cnn(input)
    # print(output.shape)

# Loss function
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# Optimizer
learning_rate = 0.0001
optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate, momentum=0.9)


# dataset splitting
# Splitting the dataset into training set and testing set
train_x, test_x, train_y, test_y = train_test_split(X_data, Y_data)

train_x = torch.from_numpy(train_x).type(torch.float32).unsqueeze(-1)
train_y = torch.from_numpy(train_y).type(torch.int64)
test_x = torch.from_numpy(test_x).type(torch.float32).unsqueeze(-1)
test_y = torch.from_numpy(test_y).type(torch.int64)

batch = 50

train_ds = TensorDataset(train_x, train_y)
train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True)

test_ds = TensorDataset(test_x, test_y)
test_dl = DataLoader(test_ds, batch_size=batch)

train_data_size = len(train_ds)
test_data_size = len(test_ds)



# parameters for training model
# totaol No. of traing
total_train_step = 0

# totaol No. of testing
total_test_step = 0

epoch = 200

writer = SummaryWriter("../logs-train")
start_time =time.time()
for i in range(epoch):
    print("-------The {} round of training-------".format(i + 1))

    # training
    cnn.train()
    for data in train_dl:
        train_x, train_y = data
        if torch.cuda.is_available():
            train_x = train_x.cuda()
            train_y = train_y.cuda()

        outputs = cnn(train_x)
        loss = loss_fn(outputs, train_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 50 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("The number of training:{}, Loss:{}".format(total_train_step, 
                                                              loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # testing
    cnn.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dl:
            test_x, test_y = data
            if torch.cuda.is_available():
                test_x = test_x.cuda()
                test_y = test_y.cuda()

            outputs = cnn(test_x)
            loss = loss_fn(outputs, test_y)

            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == test_y).sum()
            total_accuracy = total_accuracy + accuracy

    print("The overall loss of testing:{}".format(total_test_loss))
    print("The overall accuracy of testing:{}".format(total_accuracy/test_data_size))

#     writer.add_scalar("test_loss",total_test_loss,total_test_step)
#     writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_test_step)
#     total_test_step = total_test_step+1

#     torch.save(cnn,"cnn{}.pth".format(i))
#     print("Model has been saved")

writer.close()
