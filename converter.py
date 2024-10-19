from utils import convert

train_set = './DataSets/MNIST/processed/training.pt'
test_set = './DataSets/MNIST/processed/test.pt'
save_path = './Images/train'
num_train = 5
num_test = 5

convert.toImages(train_set, save_path, num_train)