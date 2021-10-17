#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    #initialize weights of fully connected layer
    if type(m) == nn.Linear:
        nn.init.orthogonal_(m.weight, gain=1)
        m.bias.data.zero_()
    elif type(m) == nn.BatchNorm1d:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.Conv1d:
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in')
        m.bias.data.zero_()

class Encoder(nn.Module):
    def __init__(self, num_inputs):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(

            nn.BatchNorm1d(num_inputs),
            nn.Linear(num_inputs, num_inputs),
            nn.Dropout(0.5),
            nn.Tanhshrink(),
            nn.Linear(num_inputs, num_inputs))

        self.encoder.apply(init_weights)
    def forward(self, x):
        x = self.encoder(x)
        return x

# Decoder_a
class Decoder_a(nn.Module):
    def __init__(self, num_inputs):
        super(Decoder_a, self).__init__()
        self.decoder = nn.Sequential(

            nn.Linear(num_inputs, num_inputs),
            nn.Dropout(0.5),
            nn.Tanhshrink(),
            nn.Linear(num_inputs, num_inputs))

        self.decoder.apply(init_weights)
    def forward(self, x):
        x = self.decoder(x)
        return x

# Decoder_b
class Decoder_b(nn.Module):
    def __init__(self, num_inputs):
        super(Decoder_b, self).__init__()
        self.decoder = nn.Sequential(

            nn.Linear(num_inputs, num_inputs),
            nn.Dropout(0.5),
            nn.Tanhshrink(),
            nn.Linear(num_inputs, num_inputs))

        self.decoder.apply(init_weights)
    def forward(self, x):
        x = self.decoder(x)
        return x

# Decoder_c
class Decoder_c(nn.Module):
    def __init__(self, num_inputs):
        super(Decoder_c, self).__init__()
        self.decoder = nn.Sequential(

            nn.Linear(num_inputs, num_inputs),
            nn.Dropout(0.5),
            nn.Tanhshrink(),
            nn.Linear(num_inputs, num_inputs))

        self.decoder.apply(init_weights)
    def forward(self, x):
        x = self.decoder(x)
        return x

# Decoder_d
class Decoder_d(nn.Module):
    def __init__(self, num_inputs):
        super(Decoder_d, self).__init__()
        self.decoder = nn.Sequential(

            nn.Linear(num_inputs, num_inputs),
            nn.Dropout(0.5),
            nn.Tanhshrink(),
            nn.Linear(num_inputs, num_inputs))

        self.decoder.apply(init_weights)
    def forward(self, x):
        x = self.decoder(x)
        return x

#classifier combine with autoencoder
class Discriminator(nn.Module):
    def __init__(self, num_inputs):
        super(Discriminator, self).__init__()

        self.bn = nn.BatchNorm1d(num_inputs)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.fc1 = nn.Linear(num_inputs, 100)
        self.fc2 = nn.Linear(100, num_inputs)

        self.fc3 = nn.Linear(num_inputs, 50)
        self.fc4 = nn.Linear(50, 16)
        self.fc5 = nn.Linear(16, 1)
        self.last = nn.Sigmoid()       

    def forward(self, z):
        z = self.bn(z)

        residual = z
        z = self.relu(z)
        z = self.fc1(z)
        z = self.relu(z)
        z = self.fc2(z)
        z += residual

        z = self.relu(z)
        z = self.fc3(z)
        z = self.relu(z)
        z = self.fc4(z)
        z = self.relu(z)
        z = self.fc5(z)
        z = self.last(z)         
        return z

