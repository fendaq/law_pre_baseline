import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

learning_rate = 0.1
batch_size = 8
embedding_size = 100 + 50 #100word2vec，50 POS tag
hidden_size = 75



class encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(encoder, self).__init__()

        self.dataSize = input_size
        self.hiddenSize = hidden_size
        self.outputSize = output_size
        self.GRU = nn.GRU(input_size, hidden_size, bidirectional = True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, output_size))

    def forward(self, x):
        out, _ = self.GRU(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out



class model(nn.Module):
    pass


