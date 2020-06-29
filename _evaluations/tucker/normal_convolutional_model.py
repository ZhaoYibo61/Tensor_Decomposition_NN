from layers.Conv2dTucker import Conv2dTucker
import torch

class DecomposedConvolutionalNetwork(torch.nn.Module):

    def __init__(self, decomposition_rank=3, num_classes=2):
        super(DecomposedConvolutionalNetwork, self).__init__()
        self.layer1 = torch.nn.Sequential(
            Conv2dTucker(in_channels=3, out_channels=32, kernel_size=3, decomp_rank=decomposition_rank),
            torch.nn.ReLU()#,
            # torch.nn.MaxPool2d(kernel_size=3)
        )
        self.layer2 = torch.nn.Sequential(
            Conv2dTucker(in_channels=32, out_channels=64, kernel_size=5, decomp_rank=decomposition_rank),
            torch.nn.ReLU()#,
            # torch.nn.MaxPool2d(kernel_size=5)
        )
        self.fully_connect = torch.nn.Linear(in_features=64, out_features=num_classes)


    def forward(self, input_tensor):
        output = self.layer1(input_tensor)
        output = self.layer2(output)
        print(output.shape)
        # exit()
        output = output.view(-1, 1)
        output = self.fully_connect(output)
        return output


'''=================================================================================================================='''
'''=================================================================================================================='''
'''=================================================================================================================='''

from torchvision import transforms, datasets
import os
from torch.utils import data

'''Dataset Setup'''
def example_dataset_setup(crop=224, mean_input=[0.5, 0.5, 0.5], stand_dev=[0.5, 0.5, 0.5], data_path='../hymenoptera_data', batch_size=1, num_workers=4):
    data_transform = transforms.Compose([
        transforms.Resize(size=crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_input, std=stand_dev)
    ])
    training_folder = datasets.ImageFolder(root=os.path.join(data_path, 'train'), transform=data_transform)
    testing_folder = datasets.ImageFolder(root=os.path.join(data_path, 'val'), transform=data_transform)

    training_dataset = data.DataLoader(dataset=training_folder, batch_size=batch_size, num_workers=num_workers)
    testing_dataset = data.DataLoader(dataset=testing_folder, batch_size=batch_size, num_workers=num_workers)
    return training_dataset, testing_dataset


'''Example run'''
def example_run_001(epochs=10, console_print=True):
    training_data, testing_data = example_dataset_setup()
    # compute_power = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    model = DecomposedConvolutionalNetwork()
    # model = model.to(compute_power)
    loss_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters())

    '''================================================TRAINING======================================================'''
    total_step = len(training_data)
    all_loss = []
    all_accuracy = []
    for epoch in range(epochs):
        for current_step, (images, labels) in enumerate(training_data):
            # Forward pass
            current_output = model(images)
            training_loss = loss_criterion(current_output, labels)

            # Bookeeping
            all_loss.append(training_loss.item())

            # Backpropagation  & Optimization
            optimizer.zero_grad()
            training_loss.backward()
            optimizer.step()

            # Bookeeping
            total = labels.size(0)
            x, predicted = torch.max(current_output.data, dim=1)
            correct = (predicted == labels).sum().item()
            all_accuracy.append(correct / total)

            if console_print:
                if (current_step + 1) % 100 == 0:
                    print('Epoch {}/{}, Step {}/{}, Training Loss: {:.4f}, Training Accuracy: {:.2f}%'.format(
                        epoch + 1, epochs, current_step + 1, total_step, training_loss.item(), (correct / total) * 100
                    ))

    '''================================================TESTING======================================================='''
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in testing_data:
            output = model(images)
            x, predicted = torch.max(output.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy: {}%'.format((correct/total)*100))


'''==================================================RUN MODELS======================================================'''
example_run_001()