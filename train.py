from model import Net
import dataset
from parameters import *

# Pytorch Libraries and modules
import torch
from torch.optim import RMSprop
from torch.nn import CrossEntropyLoss

def train():
    # setup the device for running
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net()
    model = model.to(device)

    lossFile = open("loss.txt", "a+")

    train_loader = dataset.get_train_loader(batch_size, augmented_training_data)
    num_train_batches = len(train_loader)
    val_loader = dataset.get_val_loader(batch_size)
    num_val_batches = len(val_loader)   

    loss_list = []
    criterion = CrossEntropyLoss().to(device)
    optimizer = RMSprop(model.parameters(), lr=learning_rate)
    model.train()

    epoch = 1
    while epoch <= num_epochs:
        for param_group in optimizer.param_groups:
            print('Current learning rate: ' + str(param_group['lr']))
        model.train()
        for batch_num, (inputs, labels) in enumerate(train_loader, 1):
            # print("train batch_num: ", batch_num)
            inputs = inputs.view(inputs.size(0), inputs.size(3),inputs.size(1),inputs.size(2))
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs).to(device) 
            optimizer.zero_grad()

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            if batch_num % output_period == 0:
                print('[%d:%.2f] loss: %.3f' % (
                    epoch, batch_num*1.0/num_train_batches,
                    loss.item()
                    ))      
        epoch += 1

        print("evaluating model!")
        model.eval()
        with torch.no_grad():
            for batch_num, (inputs, labels) in enumerate(val_loader, 1):
                inputs = inputs.view(inputs.size(0), inputs.size(3),inputs.size(1),inputs.size(2))
                inputs, labels = inputs.to(device), labels.to(device)
                # print("val batch_num: ", batch_num)

                outputs = model(inputs).to(device) 
                loss = criterion(outputs, labels)

                if batch_num % output_period == 0:
                    print('[%d:%.2f] error: %.3f' % (
                            epoch-1, batch_num*1.0/num_val_batches,
                            loss.item()
                            ))
                    loss_list.append(loss.item())
    # save after every epoch
    print(loss_list)
    torch.save(model.state_dict(), path)
    lossFile.close()

if __name__ == '__main__':
    print('Starting training')
    train()
    print('Training terminated')