import torch.nn as nn
import torch.nn.functional as F
from avalam import Board
        

class DQN(nn.Module):

    def __init__(self, h, w, outputs,device):
        super(DQN, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(1, 9 * 9, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(9 * 9)
        self.conv2 = nn.Conv2d(9 * 9, 64, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=2, stride=1)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w,5),3),2)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h,5),3),2)
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)


    def forward(self, x):
        x = x.to(self.device)
        x = F.elu(self.bn1(self.conv1(x)))
        x = F.elu(self.bn2(self.conv2(x)))
        x = F.elu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class BestMove():
    def __init__(self) -> None:
        self.move = None

    def update(self, act) -> None:
        self.move = act
