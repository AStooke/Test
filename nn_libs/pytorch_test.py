
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data
import time

EPOCHS = 10
BATCHES = 100
BATCH = 128
DATA = BATCH * BATCHES


class CNN(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(9 * 9 * 64, 512)  # annoying to write 7 * 7 * 64
        self.fc2 = nn.Linear(512, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 9 * 9 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

    def num_flat_features(self, x):
        return int(np.prod(x.size()[1:]))


def load_data(img_shape, output_size):

    print("Generating synthetic data")
    x = np.random.randn(DATA, *img_shape).astype("float32")
    y = np.random.randint(low=0, high=output_size - 1, size=DATA).astype("int32")

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    return data.TensorDataset(x, y)


def main():
    img_shape = (3, 84, 84)
    output_size = 10

    cnn = CNN(output_size)
    cnn.cuda()
    optimizer = optim.SGD(cnn.parameters(), lr=1e-3)

    dataset = load_data(img_shape, output_size)
    train_loader = data.DataLoader(dataset=dataset, batch_size=BATCH, shuffle=False)

    for x, _ in train_loader:
        x = x.cuda()
        x = Variable(x)
        output = cnn(x)
        pred = output.data.max(1, keepdim=True)

    t_0 = time.time()
    for _ in range(EPOCHS):
        for x, _ in train_loader:
            x = x.cuda()
            x = Variable(x)
            output = cnn(x)
            pred = output.data.max(1, keepdim=True)
    t_1 = time.time()
    print("Ran inference on {} batches in {:.3f} s".format(BATCHES * EPOCHS, t_1 - t_0))

    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        x = Variable(x)
        y = Variable(y)
        optimizer.zero_grad()
        output = cnn(x)
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()

    t_0 = time.time()
    for _ in range(EPOCHS):
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            x = Variable(x)
            y = Variable(y)
            optimizer.zero_grad()
            output = cnn(x)
            loss = F.nll_loss(output, y)
            loss.backward()
            optimizer.step()
    t_1 = time.time()
    print("Ran training on {} batches in {:.3f} s".format(BATCHES * EPOCHS, t_1 - t_0))


if __name__ == "__main__":
    # torch.backends.cudnn.benchmark = True
    main()

