# You need to learn a function with n inputs.
# For given number of inputs, we will generate random function.
# Your task is to learn it
import time
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ..utils import solutionmanager as sm
from ..utils import gridsearch as gs
import numpy

class SolutionModel(nn.Module):
    def __init__(self, input_size, output_size, solution):
        super(SolutionModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = solution.hidden_size
        
        self.linear1 = nn.Linear(input_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = nn.Linear(self.hidden_size, output_size)
        self.activator = solution.activator
        self.bilinear = nn.Bilinear(self.hidden_size, self.hidden_size, output_size)
        self.bilinear0 = nn.Bilinear(self.hidden_size, self.hidden_size, self.hidden_size)

        self.LeakyReLU_coef0 = solution.LeakyReLU_coef0
        self.LeakyReLU_coef1 = solution.LeakyReLU_coef1
        self.BN = nn.BatchNorm1d(self.hidden_size, eps = 1e-05, track_running_stats=False)

    def forward(self, x):
        x = self.linear1(x)
        
        x = self.BN(x)
        y = torch.nn.LeakyReLU(self.LeakyReLU_coef0)(x)

        x = self.bilinear0(x, y)
        y = torch.nn.LeakyReLU(self.LeakyReLU_coef1)(x)

        x = self.linear3(y)
        x = torch.sigmoid(x)
        return x

    def calc_error(self, output, target):
        loss = torch.nn.BCELoss()
        return loss(output, target)

    def calc_predict(self, output):
        return output.round()

class Solution():
    def __init__(self):
        self.learning_rate = 0.045
        self.hidden_size = 20
        self.activator = torch.sigmoid
        self.LeakyReLU_coef0 = 0.074
        self.LeakyReLU_coef1 = 0.035
        # Grid search settings, see grid_search_tutorial
        self.learning_rate_grid = numpy.arange(0.02,0.04,0.005)
        self.LeakyReLU_coef0_grid = numpy.arange(0.07,0.09,0.002)
        self.LeakyReLU_coef1_grid = numpy.arange(0.02,0.04,0.002)
        self.LeakyReLU_coef2_grid = numpy.arange(0.07,0.09,0.002)
        self.hidden_size_grid = range(18,22)

        # grid search will initialize this field
        self.grid_search = None
        # grid search will initialize this field
        self.iter = 0
        # This fields indicate how many times to run with same arguments
        self.iter_number = 1

    # Return trained model
    def train_model(self, train_data, train_target, context):
        #print (train_data)
        #print (train_target)
        model = SolutionModel(train_data.size(1), train_target.size(1), self)

        

        optimizer = optim.Rprop(model.parameters(), lr=self.learning_rate)
        
        while True:
            # Report step, so we know how many steps
            context.increase_step()
            # model.parameters()...gradient set to zero
            optimizer.zero_grad()
            # evaluate model => model.forward(data)
            output = model(train_data)
            # if x < 0.5 predict 0 else predict 1
            predict = model.calc_predict(output)
            # Number of correct predictions
            correct = predict.eq(train_target.view_as(predict)).long().sum().item()
            # Total number of needed predictions
            total = predict.view(-1).size(0)
            # No more time left or learned everything, stop training
            time_left = context.get_timer().get_time_left()
            if time_left < 0.1 or correct == total:
                break
            # calculate error
            error = model.calc_error(output, train_target)
            #print(error)
            # calculate deriviative of model.forward() and put it in model.parameters()...gradient
            error.backward()
            # print progress of the learning
            self.print_stats(context.step, error, correct, total)

            
            # update model: model.parameters() -= lr * gradient
            optimizer.step()
        return model

    def print_stats(self, step, error, correct, total):
        if step % 1000 == 0:
            print("Step = {} Correct = {}/{} Error = {}".format(step, correct, total, error.item()))


###
###
### Don't change code after this line
###
###
class Limits:
    def __init__(self):
        self.time_limit = 2.0
        self.size_limit = 10000
        self.test_limit = 1.0

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10

    def create_data(self, input_size, seed):
        random.seed(seed)
        data_size = 1 << input_size
        data = torch.FloatTensor(data_size, input_size)
        target = torch.FloatTensor(data_size)
        for i in range(data_size):
            for j in range(input_size):
                input_bit = (i>>j)&1
                data[i,j] = float(input_bit)
            target[i] = float(random.randint(0, 1))
        return (data, target.view(-1, 1))

    def create_case_data(self, case):
        input_size = min(3+case, 7)
        data, target = self.create_data(input_size, case)
        return sm.CaseData(case, Limits(), (data, target), (data, target)).set_description("{} inputs".format(input_size))


class Config:
    def __init__(self):
        self.max_samples = 10000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()

run_grid_search = False
# Uncomment next line if you want to run grid search
#run_grid_search = True
if run_grid_search:
    gs.GridSearch().run(Config(), case_number=16, random_order=False, verbose=False)
else:
    # If you want to run specific case, put number here
    sm.SolutionManager().run(Config(), case_number=-1)