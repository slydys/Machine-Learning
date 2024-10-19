import math
import torch
import os
import time

from utils import data_loader
from models import CNN

data_path = './DataSets'
model_path = './models'
device = torch.device("mps")
batch_size = 512
epoch = 10

print('Loading train set...')
train_set = data_loader.load_train_set(data_path)
train_data = torch.utils.data.DataLoader(dataset = train_set, batch_size = batch_size, shuffle = True)
batch_num = math.ceil(len(train_set) / batch_size)
print('Using ', device)

start_time = time.time()

model = CNN.MyCNN().to(device)
cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

print('-----------------\n'
      'num of epoch:{} \n'
      'batch size: {}\n'
      'num of batch: {}'.format(epoch, batch_size, batch_num))
print('-----------------\n')
print('Start training...')

for epoch in range(epoch):
    print('Training epoch {}/{}'.format(epoch + 1, epoch))
    num_correct = 0
    val_loss = 0
    for batch_idx, (images, labels) in enumerate(train_data):
        num_correct_batch = 0
        val_loss_batch = 0
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        pred = torch.max(outputs, 1)[1]
        optimizer.zero_grad()
        loss = cost(outputs, labels)
        loss.backward()
        optimizer.step()
        val_loss_batch += loss.data
        val_loss += val_loss_batch
        num_correct_batch += (pred == labels).sum().item()
        num_correct += num_correct_batch
        print('Batch {}/{}, Loss: {:.6f}, Accuracy: {:.6f}%'.format(batch_idx + 1, batch_num, val_loss_batch / batch_size,  100 * num_correct_batch /batch_size))
    print('Epoch {}: Loss: {:.6f}, Accuracy: {:.6f}%\n'.format(epoch + 1, val_loss / len(train_set), 100 * num_correct / len(train_set)))

    print('Saving the model...')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(model, model_path + '/MyCNN_MNIST.pkl')

end_time = time.time()
using_time = end_time - start_time
print('Using time: ', using_time)



