
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, ZeroPad2d,Flatten

class Net(Module):
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            ZeroPad2d(padding=(5,6,5,5)),
            Conv2d(1, 16, kernel_size=(12,16), stride=(7,9)),
            ReLU(),
        )

        self.some_other_layer = Sequential(
            ZeroPad2d(padding=(1,1,1,2)),
            Conv2d(16, 32, kernel_size=(6, 6), stride=(4,4)),
            ReLU(),


        )

        self.some_other_layer_2 = Sequential(
            Conv2d(32, 256, kernel_size=(9, 9), stride=(1, 1)),
            ReLU(),

        )

        self.some_other_layer_3 = Flatten(start_dim=1, end_dim=-1)





    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.some_other_layer(x)
        x = self.some_other_layer_2(x)
        x = self.some_other_layer_3(x)
        return x