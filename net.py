import torch.nn as nn
import torch.nn.functional as F
import torch


class Discriminator(nn.Module):
    def __init__(self, input_size=784, hidden_dim=32, output_size=1):
        super(Discriminator, self).__init__()
        
        self.discriminator = nn.Sequential(
            nn.Linear(input_size, hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, output_size)
        )

    def forward(self, x):
        # x = x.flatten()
        x = x.view(-1, 784)
        return self.discriminator(x)


class Generator(nn.Module):
    def __init__(self, input_size=100, hidden_dim=32, output_size=784):
        super(Generator, self).__init__()

        self.generator = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim * 4, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.generator(x)


def real_loss(D_out, smooth=False):
    batch_size = D_out.size(0)
    if smooth:
        labels = torch.ones(batch_size) * 0.9
    else:
        labels = torch.ones(batch_size)

    if torch.cuda.is_available():
        labels = labels.cuda()

    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)
    return loss

def fake_loss(D_out):
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size)

    if torch.cuda.is_available():
        labels = labels.cuda()
        
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)
    return loss


if __name__ == '__main__':
    generator = Generator()
    discriminator = Discriminator()
    
    print(generator)
    print(discriminator)