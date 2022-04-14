from ast import arg
from inspect import ArgSpec
from pickletools import optimize
from re import X
from tkinter import Y
import numpy as np
from src.model import ModelClassifier
from src.parser import parameter_parser

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

from src import MyCustomDataSet


from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class DatasetMaper(Dataset):
    '''
    Handles batches of dataset
    '''

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class Execute:
    '''
    Class for execution. Initializes the preprocessing as well as the model
    '''

    def __init__(self, args):
        self.__init_data__(args)
        self.args = args
        self.batch_size = args.batch_size
        self.model = ModelClassifier(args)

    def __init_data__(self, args):

        self.DataCustom = MyCustomDataSet(args)
        self.DataCustom.load_data()

        raw_x_train = self.DataCustom.X_train
        raw_x_test = self.DataCustom.X_test

        self.y_train = self.DataCustom.y_train
        self.y_test = self.DataCustom.y_test

        self.x_train = raw_x_train
        self.x_test = raw_x_test
        # self.x_train = self.x_train.reshape(self.x_train.shape[0], (self.x_train.shape[1]*self.x_train.shape[2]))
        # self.x_test = self.x_test.reshape(self.x_test.shape[0], (self.x_test.shape[1]*self.x_test.shape[2]))
        print(self.x_train.shape)
        print(self.x_test.shape) 

        print(self.y_train.shape)
        print(self.y_test.shape)
        # print(self.x_train[0])

    def train(self):

        training_set = DatasetMaper(self.x_train, self.y_train)
        test_set = DatasetMaper(self.x_test, self.y_test)

        self.loader_training = DataLoader(
            training_set, batch_size=self.batch_size)
        self.loader_test = DataLoader(test_set)

        optimizer = optim.RMSprop(
            self.model.parameters(), lr=args.learning_rate)
        for epoch in range(args.epochs):

            predictions = []

            self.model.train()

            for x_batch, y_batch in self.loader_training:

                x = x_batch.type(torch.LongTensor)
                y = y_batch.type(torch.FloatTensor)

                y_pred = self.model(x)
                y_pred = y_pred.squeeze(1)
                # print(y_pred.shape)
                loss = F.binary_cross_entropy(y_pred, y)

                optimizer.zero_grad()

                loss.backward()

                predictions += list(y_pred.squeeze().detach().numpy())

            test_predictions = self.evaluation()

            train_accuracy = self.calculate_accuray(self.y_train, predictions)
            test_accuracy = self.calculate_accuray(
                self.y_test, test_predictions)

            print("Epoch: %d, loss: %.5f, Train accuracy: %.5f, Test accuracy: %.5f" % (
                epoch+1, loss.item(), train_accuracy, test_accuracy))

    def evaluation(self):

        predictions = []
        self.model.eval()
        with torch.no_grad():
            for x_batch, y_batch in self.loader_test:
                x = x_batch.type(torch.LongTensor)
                y = y_batch.type(torch.FloatTensor)

                y_pred = self.model(x)
                predictions += list(y_pred.detach().numpy())

        return predictions

    @staticmethod
    def calculate_accuray(grand_truth, predictions):
        true_positives = 0
        true_negatives = 0

        for true, pred in zip(grand_truth, predictions):
            if (pred > 0.5) and (true == 1):
                true_positives += 1
            elif (pred < 0.5) and (true == 0):
                true_negatives += 1
            else:
                pass

        return (true_positives+true_negatives) / len(grand_truth)


if __name__ == "__main__":
    args = parameter_parser()

    execute = Execute(args)
    if torch.cuda.is_available():
        execute.train()
    else:
        print("cuda is not available")