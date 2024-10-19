import torch
import time

from utils import data_loader

data_path = './DataSets'
model_path = './models'
batch_size = 512
device = torch.device("cpu")

print('Loading test set...')
test_set = data_loader.load_test_set(data_path)
test_data = torch.utils.data.DataLoader(dataset = test_set, batch_size = batch_size, shuffle = False)
print('Using ', device)
print('Loading saved model...')
model = torch.load(model_path + '/MyCNN_MNIST.pkl')
print('Testing...')

num_correct = 0
start_time = time.time()
for images, labels in test_data:
    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)
    pred = torch.max(outputs, 1)[1]
    num_correct += (pred == labels).sum().item()
end_time = time.time()
using_time = end_time - start_time
print('Accuracy: {:.6f}%'.format(100 * num_correct / len(test_set)))
print('Using time: ',using_time)