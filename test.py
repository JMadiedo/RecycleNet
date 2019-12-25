from model import Net
from parameters import *
import dataset

# Pytorch Libraries and modules
import torch

def test():
    # setup the device for running
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net()
    model.load_state_dict(torch.load(path+ "_"+str(learning_rate)))
    model = model.to(device)

    test_loader = dataset.get_test_loader(batch_size)
    num_test_batches = len(test_loader)

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch_num, (inputs, labels) in enumerate(test_loader, 1):
            inputs = inputs.view(inputs.size(0), inputs.size(3),inputs.size(1),inputs.size(2))
            inputs, labels = inputs.to(device), labels.to(device)
            print("test batch_num: ", batch_num)

            outputs = model(inputs).to(device) 
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = (100 * correct / total)

            if batch_num % test_output_period == 0:
                print('[%d:%.2f] accuracy: %d %%' % (
                    1, batch_num*1.0/num_test_batches,
                    accuracy
                    ))

if __name__ == '__main__':
    print('Starting testing')
    test()
    print('Testing terminated')