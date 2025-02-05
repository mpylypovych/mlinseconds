# There are random function from 8 inputs and X random inputs added.
# We split data in 2 parts, on first part you will train and on second
# part we will test
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
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


class SolutionModel(nn.Module):
    def __init__(self, input_size, output_size, solution):
        super(SolutionModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = solution.hidden_size
        
        self.first_size = 16

        self.linear0 = nn.Linear(self.input_size, self.first_size)
        self.linear01 = nn.Linear( self.first_size, self.hidden_size)

        self.linear1 = nn.Linear(self.input_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear2_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = nn.Linear(self.hidden_size, output_size)
        self.activator = solution.activator

        self.LeakyReLU_coef0 = solution.LeakyReLU_coef0
        self.LeakyReLU_coef1 = solution.LeakyReLU_coef1
        self.BN0 = nn.BatchNorm1d(self.first_size, eps = 1e-05, track_running_stats=False)
        self.BN = nn.BatchNorm1d(self.hidden_size, eps = 1e-05, track_running_stats=False)

        self.droput_coef = 0.00005
        self.in_bias = torch.rand(1, self.hidden_size)
        
        self._out_bias = nn.Parameter(self.in_bias)

    def forward(self, x):

        #x = nn.Dropout(0.001)(x)
        #x = nn.Dropout(self.droput_coef)(x)
        x = self.linear0(x)
        #x = x + self._out_bias
        x = torch.nn.LeakyReLU(self.LeakyReLU_coef0)(x)
        x = self.BN0(x)
        #x = x + self._out_bias
        # x = nn.Dropout(self.droput_coef)(x)

        x = self.linear01(x)
        #x = x + self._out_bias
        x = torch.nn.LeakyReLU(self.LeakyReLU_coef0)(x)
        x = self.BN(x)
        #x = x + self._out_bias
        # x = nn.Dropout(self.droput_coef)(x)

        x = self.linear2(x)
        #x = x + self._out_bias
        x = torch.nn.LeakyReLU(self.LeakyReLU_coef0)(x)
        x = self.BN(x)
        x = x + self._out_bias

        # №x = nn.Dropout(self.droput_coef)(x)

        #x = self.linear2_1(x)
        #x = x + self._out_bias
        #x = torch.nn.LeakyReLU(self.LeakyReLU_coef0)(x)
        #x = self.BN(x)
        #x = x + self._out_bias
        #x = nn.Dropout(self.droput_coef)(x)
        
        x = self.linear3(x)

        x = nn.Sigmoid()(x)
        return x

    def calc_error(self, output, target):
        # This is loss function
        loss = torch.nn.BCELoss()
        return loss(output, target)

    def calc_predict(self, output):
        # Simple round output to predict value
        return output.round()

class Solution():
    def __init__(self):
        # Control speed of learning
        self.learning_rate = 0.0103
        # Control number of hidden neurons
        self.hidden_size = 50
        self.hidden_size2 = 40
        self.activator = torch.sigmoid
        self.LeakyReLU_coef0 = 0.0074
        self.LeakyReLU_coef1 = 0.035
        self.grid_search = None
        self.iter = 0
        self.iter_number = 1

    # Return trained model
    def train_model(self, train_data, train_target, context):
 
        # Model represent our neural network
        model = SolutionModel(train_data.size(1), train_target.size(1), self)
        # Optimizer used for training neural network
        sm.SolutionManager.print_hint("Hint[2]: Learning rate is too small", context.step)

        optimizer = optim.RMSprop(model.parameters(), lr=self.learning_rate)

        batch_size = 1024
        batch_count = 8192 // batch_size
        
        while True:
            confidence_sum = 0.0
            for i in range(batch_count):
                train_batch = train_data[batch_size * i : batch_size * (i + 1) ]
                target_batch = train_target[batch_size * i : batch_size * (i + 1) ]
                # Report step, so we know how many steps
                context.increase_step()
                # model.parameters()...gradient set to zero
                optimizer.zero_grad()
                # evaluate model => model.forward(data)
                output = model(train_batch)
                # if x < 0.5 predict 0 else predict 1
                predict = model.calc_predict(output)

                lol = output - target_batch
                confidence_sum += abs(lol).sum()

                # calculate error
                error = model.calc_error(output, target_batch)
                # calculate deriviative of model.forward() and put it in model.parameters()...gradient
                error.backward()

                # update model: model.parameters() -= lr * gradient
                optimizer.step()

            time_left = context.get_timer().get_time_left()

            if time_left < 0.1 or confidence_sum / 8192.0 < 0.0075:
                break
        return model


###
###
### Don't change code after this line
###
###
class Limits:
    def __init__(self):
        self.time_limit = 2.0
        self.size_limit = 1000000
        self.test_limit = 1.0

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10

    def create_data(self, data_size, input_size, random_input_size, seed):
        torch.manual_seed(seed)
        function_size = 1 << input_size
        function_input = torch.ByteTensor(function_size, input_size)
        for i in range(function_input.size(0)):
            fun_ind = i
            for j in range(function_input.size(1)):
                input_bit = fun_ind&1
                fun_ind = fun_ind >> 1
                function_input[i][j] = input_bit
        function_output = torch.ByteTensor(function_size).random_(0, 2)

        if data_size % function_size != 0:
            raise "Data gen error"

        data_input = torch.ByteTensor(data_size, input_size).view(-1, function_size, input_size)
        target = torch.ByteTensor(data_size).view(-1, function_size)
        for i in range(data_input.size(0)):
            data_input[i] = function_input
            target[i] = function_output
        data_input = data_input.view(data_size, input_size)
        target = target.view(data_size)
        if random_input_size > 0:
            data_random = torch.ByteTensor(data_size, random_input_size).random_(0, 2)
            data = torch.cat([data_input, data_random], dim=1)
        else:
            data = data_input
        perm = torch.randperm(data.size(1))
        data = data[:,perm]
        perm = torch.randperm(data.size(0))
        data = data[perm]
        target = target[perm]
        return (data.float(), target.view(-1, 1).float())

    def create_case_data(self, case):
        data_size = 256*32
        input_size = 8
        random_input_size = min(32, (case-1)*4)

        data, target = self.create_data(2*data_size, input_size, random_input_size, case)
        return sm.CaseData(case, Limits(), (data[:data_size], target[:data_size]), (data[data_size:], target[data_size:])).set_description("{} inputs and {} random inputs".format(input_size, random_input_size))

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
    gs.GridSearch().run(Config(), case_number=1, random_order=False, verbose=False)
else:
    # If you want to run specific case, put number here
    sm.SolutionManager().run(Config(), case_number=-1)

  