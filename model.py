from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d, Dropout

class Print(Module):
    def __init__(self, name):
        super(Print, self).__init__()
        self.name = name

    def forward(self, x):
        print("shape of "+self.name+": ", x.shape)
        return x

class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()
        self.cnn_layers = Sequential(
            Conv2d(3, 32, kernel_size=3, stride=2),
            ReLU(),
            BatchNorm2d(32),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(32, 32, kernel_size=3, stride=2),
            ReLU(),
            BatchNorm2d(32),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(32, 64, kernel_size=3, stride=2),
            ReLU(),
            BatchNorm2d(64),
            MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = Sequential(
            Linear(64*2*2, 64),
            Dropout(),
            Linear(64, 4),
            # Dropout()
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x