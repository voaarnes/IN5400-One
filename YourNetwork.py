import torch
import torch.nn as nn
from RainforestDataset import get_classes_list


class TwoNetworks(nn.Module):
    '''
    This class takes two pretrained networks,
    concatenates the high-level features before feeding these into
    a linear layer.

    functions: forward
    '''
    def __init__(self, pretrained_net1, pretrained_net2):
        super(TwoNetworks, self).__init__()

        _, num_classes = get_classes_list()

        # TODO select all parts of the two pretrained networks, except for
        # the last linear layer.
        self.fully_conv1 = torch.nn.Sequential(*(list(pretrained_net1.children())[:-1]))

        pretrained_net2.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=(3, 3), bias=False)
        self.fully_conv2 = torch.nn.Sequential(*(list(pretrained_net2.children())[:-1]))


        # TODO create a linear layer that has in_channels equal to
        # the number of in_features from both networks summed together.
        self.linear = nn.Linear(1024, num_classes)
        self.activation = nn.Sigmoid()



    def forward(self, inputs1, inputs2):
        # TODO feed the inputs through the fully convolutional parts
        # of the two networks that you initialised above, and then
        # concatenate the features before the linear layer.
        # And return the result.
        out1 = self.fully_conv1(inputs1)
        out2 = self.fully_conv2(inputs2)

        combined = torch.cat((out1.view(out1.size(0), -1), out2.view(out2.size(0), -1)), dim=1)
        output = self.linear(combined)

        return self.activation(output)


class SingleNetwork(nn.Module):
    '''
    This class takes one pretrained network,
    the first conv layer can be modified to take an extra channel.

    functions: forward
    '''

    def __init__(self, pretrained_net, weight_init=None):
        super(SingleNetwork, self).__init__()
        _, num_classes = get_classes_list()

        #print(pretrained_net.model.weight.data.size())

        if weight_init is not None:
            # TODO Here we want an additional channel in the weights tensor, specifically in the first
            # conv2d layer so that there are weights for the infrared channel in the input aswell.
            current_weights = []



            if weight_init == "kaiminghe":
                #pass
                irinput = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=(3, 3), bias=False)
                x = torch.nn.init.kaiming_uniform_(irinput.weight)                                  # Initialize weights with kaiming_uniform
                y = pretrained_net.conv1.weight
                current_weights = torch.cat((x, y),1)                                               # concatenate IR Weights into premade RGB


            # TODO Create a new conv2d layer, and set the weights to be
            # what you created above. You will need to pass the weights to
            # torch.nn.Parameter() so that the weights are considered
            # a model parameter.
            # eg. first_conv_layer.weight = torch.nn.Parameter(your_new_weights)

            pretrained_net.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=(3, 3), bias=False)
            pretrained_net.conv1.weight = torch.nn.Parameter(current_weights)


        # TODO Overwrite the last linear layer.
        pretrained_net.fc = nn.Linear(in_features=512, out_features=num_classes)
        self.activation = nn.Sigmoid()
        self.net = pretrained_net

    def forward(self, inputs):
        x = self.net(inputs)
        return self.activation(x)
