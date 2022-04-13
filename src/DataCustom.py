from torch.utils.data.dataset import Dataset
import torchvision
import torchvision.transforms as transform
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


class MyCustomDataSet:
    def __init__(self,args):
        self.max_len = args.max_len
        self.max_number = args.max_number
        self.test_size = args.test_size
        self.no_of_timesteps = args.no_of_timesteps

        
    def load_data(self):
        X = []
        y = []
        comeback_df = pd.read_csv("data/comeback.txt")
        moveleft_df = pd.read_csv("data/moveleft.txt")
        
        dataset = comeback_df.iloc[:,1:].values
        n_sample = len(dataset)
        for i in range(self.no_of_timesteps,n_sample):
            X.append(dataset[i-self.no_of_timesteps:i,:])
            y.append(1)

        dataset = moveleft_df.iloc[:,1:].values
        n_sample = len(dataset)
        for i in range(self.no_of_timesteps, n_sample):
            X.append(dataset[i-self.no_of_timesteps:i,:])
            y.append(0)


        X, y = np.array(X), np.array(y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y,test_size = self.test_size)
