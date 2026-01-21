import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt

## define a function for the generator
def make_generator_network(
        input_size=20,
        num_hidden_layers=1,
        num_hidden_units=100,
        num_ouput_units=784):
    model = nn.Sequential()
    for i in range(num_hidden_layers):
        model.add_module(f'fc_g{i}',
                         nn.Linear(input_size,num_hidden_units))
        model.add_module(f'relu_g{i}',nn.LeakyReLU())
        input_size=num_hidden_units
    model.add_module(f'fc_g{num_hidden_layers}',
                     nn.Linear(num_hidden_units,num_ouput_units))
    model.add_module(f"tanh_g", nn.Tanh())
    return model

## define a function for the discriminator:
def make_discriminator_network(
        input_size,
        num_hidden_layers=1,
        num_hidden_units=100,
        num_output_units=1):
    model = nn.Sequential()
    for i in range(num_hidden_layers):
        model.add_module(f'fc_d{i}', nn.Linear(input_size,num_hidden_units, bias=False))
        model.add_module(f'relu_d{i}', nn.LeakyReLU())
        model.add_module('dropout', nn.Dropout(p=0.5))
        input_size = num_hidden_units
    model.add_module(f'fc_d{num_hidden_layers}', nn.Linear(num_hidden_units,num_output_units))
    model.add_module('sigmoid', nn.Sigmoid())
    return model

image_size = (28,28)
z_size = 20
gen_hidden_layers = 1
gen_hidden_size = 100
disc_hidden_layers = 1
disc_hidden_size = 100
torch.manual_seed(1)
gen_model = make_generator_network(
                                   input_size=z_size,
                                   num_hidden_layers=gen_hidden_layers,
                                   num_hidden_units=gen_hidden_size,
                                   num_ouput_units= np.prod(image_size)
                                   )
print(gen_model)

disc_model = make_discriminator_network(
                                        input_size=np.prod(image_size),
                                        num_hidden_layers=disc_hidden_layers,
                                        num_hidden_units= disc_hidden_size,
                                        )
print(disc_model)