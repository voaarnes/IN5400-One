import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
import os
import PIL.Image
import pandas as pd

SEED = 37


def get_classes_list():
    classes = ['clear', 'cloudy', 'haze', 'partly_cloudy',
               'agriculture', 'artisinal_mine', 'bare_ground', 'blooming',
               'blow_down', 'conventional_mine', 'cultivation', 'habitation',
               'primary', 'road', 'selective_logging', 'slash_burn', 'water']
    classes.sort()                                                                  #Added sort because it neds to corespond with the MultiLabelBinarizer's automatic sorter
    return classes, len(classes)


class ChannelSelect(torch.nn.Module):
    """This class is to be used in transforms.Compose when you want to use selected channels. e.g only RGB.
    It works only for a tensor, not PIL object.
    Args:
        channels (list or int): The channels you want to select from the original image (4-channel).

    Returns: img
    """
    def __init__(self, channels=[0, 1, 2]):
        super().__init__()
        self.channels = channels

    def forward(self, img):
        """
        Args:
            img (Tensor): Image
        Returns:
            Tensor: Selected channels from the image.
        """
        return img[self.channels, ...]

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RainforestDataset(Dataset):
    def __init__(self, root_dir, trvaltest, transform):

        self.imgNames = []
        self.bLabels = []
        self.imgFiles = []
        self.ending = '.tif'

        classes, num_classes = get_classes_list()
        mlb = MultiLabelBinarizer()
        # TODO Binarise your multi-labels from the string. HINT: There is a useful sklearn function to
        # help you binarise from strings.
        img_labels = pd.read_csv(root_dir+'train_v2.csv')
        self.imgNames = img_labels["image_name"]
        self.bLabels = mlb.fit_transform(img_labels["tags"].str.split(" "))


        # Load PATHS of first 1000 images
        count = 0
        for iName in self.imgNames:
            if(count>1000): break;
            name = os.path.join(root_dir+'train-tif-v2/', iName + self.ending)
            print(name)
            self.imgFiles.append(name)
            count+=1



        # TODO Perform a test train split. It's recommended to use sklearn's train_test_split with the following
        # parameters: test_size=0.33 and random_state=0 - since these were the parameters used
        # when calculating the image statistics you are using for data normalisation.

        #for debugging you can use a test_size=0.66 - this trains then faster


        # OR optionally you could do the test train split of your filenames and labels once, save them, and
        # from then onwards just load them from file.


    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        # TODO get the label and filename, and load the image from file.
        img = "temp"
        labels = self.bLabels[idx]

        sample = {'image': img,
                  'label': labels,
                  'filename': self.img_filenames[idx]}
        return sample


###################################################################################
root_dir = "C:/data/rainforest/"
rdata = RainforestDataset(root_dir, 0, None)
