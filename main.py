# This is a sample Python script.
import typing
import optuna
import torch.nn.functional as F
import torch.nn as nn
import tqdm
from tqdm import trange
import torchvision.transforms as transforms
import pytorch_lightning as pl
import torch
import numpy as np
import tensorflow as tf
import copy
from functools import partial
from DG2 import ism, rho_metrics, dsm
from dataclasses import dataclass, field
import typing as type
import matplotlib.pyplot as plt
import torch.utils.data as data_utils
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import GPUtil
from sparselearning.funcs.init_scheme import erdos_renyi_init
from sparselearning.core import Masking
import optuna
import sklearn.base as base
import torch.nn.utils.prune as prune
from yellowbrick.regressor import ResidualsPlot, PredictionError
from keras.wrappers.scikit_learn import KerasRegressor
import sklearn.metrics as metrics
import pandas as pd
from scipy import stats
from evograd import expectation
from evograd.distributions import Normal
import wandb
from PIL import Image
from scipy import spatial

data_X = [[0.55, 0.789, 0.697, 0.69873], [0.133654, 0.36524, 0.48563, 0.36589]]
data_Y = [1, 0]


# I need this wrapper to use yellowbrick functionality with Pytorch
################################### CLASSSES###############################

class NetWrapper(base.BaseEstimator):
    """
    Wrap our model as a BaseEstimator
    """
    _estimator_type = "regressor"

    # Tell yellowbrick this is a regressor

    def __init__(self, model):
        # save a reference to the model
        self.model = model
        self.classes_ = None
        self.device = next(model.parameters()).device
        self.output_shape = None

    def fit(self, X, y):
        # save the list of classes
        self.classes_ = list(set(i for i in y))
        self.output_shape = y.shape[1]

    def score(self, X, y, **kwargs):
        v = self.model(torch.tensor(X, dtype=torch.float, device=self.device)).detach().numpy()
        return metrics.r2_score(y_pred=v, y_true=y.reshape(v.shape))

    def predict_proba(self, X):
        """
        Define predict_proba or decision_function

        Compute predictions from model.
        Transform input into a Tensor, compute the prediction,
        transform the prediction back into a numpy array
        """
        v = self.model(torch.tensor(X, dtype=torch.float, device=self.device)).detach().numpy()
        print("v:", v.shape)
        if self.output_shape == 1:
            return np.squeeze(v)
        return v

    def predict(self, X):
        v = self.model(torch.tensor(X, dtype=torch.float, device=self.device)).detach().numpy()
        print("v:", v.shape)
        if self.output_shape == 1:
            return np.squeeze(v)
        return v.reshape((-1))


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.w11 = nn.Parameter(torch.rand(1))
        # self.register_parameter("w11",self.w11)
        self.w13 = nn.Parameter(torch.rand(1))
        # self.register_parameter("w13",self.w13)
        self.w22 = nn.Parameter(torch.rand(1))
        # self.register_parameter("w22",self.w22)
        self.w24 = nn.Parameter(torch.rand(1))
        # self.register_parameter("w24",self.w24)
        self.w31 = nn.Parameter(torch.rand(1))
        # self.register_parameter("w31",self.w31)
        self.w34 = nn.Parameter(torch.rand(1))
        # self.register_parameter("w34",self.w34)
        self.b1 = nn.Parameter(torch.rand(1))
        # self.register_parameter("b1",self.b1)
        self.b2 = nn.Parameter(torch.rand(1))
        # self.register_parameter("b2",self.b2)
        # self.b3 = nn.Parameter(torch.rand(1))
        # self.register_parameter("b3",self.b3)

    def forward(self, x):
        if len(x.shape) == 2:
            z1 = self.w11 * x[:, 0] + self.w13 * x[:, 2]
            z2 = self.w22 * x[:, 1] + self.w24 * x[:, 3]
            # z3 = self.w31 * x[:, 0] + self.w34 * x[:, 3]
            a1 = F.relu(z1)
            a2 = F.relu(z2)
            # a3 = F.relu(z3)
            y = self.b1 * a1 + self.b2 * a2  # + self.b3 * a3
            return y
        else:
            z1 = self.w11 * x[0] + self.w13 * x[2]
            z2 = self.w22 * x[1] + self.w24 * x[3]
            # z3 = self.w31 * x[0] + self.w34 * x[3]
            a1 = F.relu(z1)
            a2 = F.relu(z2)
            # a3 = F.relu(z3)
            y = self.b1 * a1 + self.b2 * a2  # + self.b3 * a3
            return y


@dataclass
class TestFunction():
    matrix: np.ndarray = field(init=False)

    def __post_init__(self):
        self.matrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])

    def forward(self):
        def function(x):
            # return 1.137 * (x[0]) * (x[2]) + 0.1037 * (x[0]) * (x[3]) + 0.678 * (x[1]) * (x[3])
            return 1.137 * (x[0]) * (x[2]) + 0.678 * (x[1]) * (x[3])

        return function

    def get_interaction_matrix(self):
        return self.matrix

    def generate_noisy_data(self, samples: int, range: type.Union[type.List[int], type.Tuple[int]]):
        xvalues = np.random.uniform(range[0], range[1], (samples, 4))
        x_volts = np.array(list(map(self.forward(), xvalues)))
        x_watts = x_volts * x_volts / len(x_volts)
        # Set a target SNR
        target_snr_db = 20
        # Calculate signal power and convert to dB
        sig_avg_watts = np.mean(x_watts)
        sig_avg_db = 10 * np.log10(sig_avg_watts)
        # Calculate noise according to [2] then convert to watts
        noise_avg_db = sig_avg_db - target_snr_db
        noise_avg_watts = 10 ** (noise_avg_db / 10)
        # Generate an sample of white noise
        mean_noise = 0
        noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(x_watts))
        result = x_volts + noise_volts

        return xvalues, result

    def generate_data(self, samples: int, range: type.Union[type.List[int], type.Tuple[int]]):
        xvalues = np.random.uniform(range[0], range[1], (samples, 4))
        result = np.array(list(map(self.forward(), xvalues)))
        return xvalues, result


class NeuralNet(pl.LightningModule):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, learning_rate: float, L1_reg: int = 0,
                 L2_reg: int = 0):
        super(NeuralNet, self).__init__()
        assert L1_reg * L2_reg == 0, f"There can only be one type of regularization at the moment. L1: {L1_reg} , " \
                                     f"L2: {L2_reg}"
        self.reg_L1 = L1_reg
        self.reg_L2 = L2_reg
        self.lr = learning_rate
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.mse_loss(logits, y, reduction="mean")
        if self.reg_L1:
            all_params = torch.nn.utils.parameters_to_vector(self.parameters())
            loss += self.reg_L1 * torch.sum(torch.abs(all_params))

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.reg_L2)
        return optimizer


"""
Define an nn.Module class for a simple residual block with equal dimensions
"""


class ResBlock(nn.Module):
    """
    Iniialize a residual block with two convolutions followed by batchnorm layers
    """

    def __init__(self, in_size: int, hidden_size: int, out_size: int):
        super().__init__()

        self.fc1 = nn.Linear(in_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_size)

    def linear_block(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    """
    Combine output with the original input
    """

    def forward(self, x): return x + self.linear_block(x)  # skip connection


################################ FUNCTIONS ################################

def define_model(layer_neurons: type.List[int], input_size: int, CLASSES: int = 1):
    in_features = input_size
    layers = []
    for out_features in layer_neurons:
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        # p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
        # layers.append(nn.Dropout(p))

        in_features = out_features
    layers.append(nn.Linear(in_features, CLASSES))
    return nn.Sequential(*layers)


def define_res_model(input_size: int, CLASSES: int = 1):
    layers = [
        nn.Linear(input_size, 114),
        ResBlock(114, hidden_size=48, out_size=114),
        nn.Linear(114, CLASSES)
    ]
    return nn.Sequential(*layers)


def objective(trial: optuna.trial.Trial) -> float:
    return evaluation_score


def hyper_parameter_optimization():
    study = optuna.create_study()
    study.optimize(objective, n_trials=500)


# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def loss_one_epoch(model: nn.Module, data_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer,
                   loss: type.Callable,
                   matrix_regularizer: bool = False,
                   train: bool = True, L1_lambda: float = 0,
                   use_wandb: bool = False,
                   use_absolute: bool = False
                   ):
    # loss = nn.CrossEntropyLoss(reduction="sum")
    model.cuda()
    cumulative_sum = 0
    total_items = 0
    batch_size = 1

    for data, label in data_loader:
        data = data.cuda()
        label = label.cuda()
        value = compute_loss_batch(batch=(data, label), model=model, loss_object=loss, train=train)
        # value = torch.zeros(1,device=torch.device("cuda"))
        if train:
            # Todo: Do I want the Biases also be included?
            sum_of_absolute_values = torch.sum(torch.abs(parameters_to_vector(model.parameters())))
            sum_of_absolute_values.mul_(L1_lambda)
            value.add_(sum_of_absolute_values)
            if use_wandb:
                wandb.log({"train objective loss": value})
            if matrix_regularizer:
                int_m, _, _, _ = ism(model, dim=dim, lb=LB, ub=UB, is_NN=True, use_grad=True)
                int_m = replace_nan(int_m)
                reg = 0
                if use_absolute:
                    reg = 100 * torch.abs(int_m).sum()
                else:
                    reg = 100 * torch.linalg.matrix_norm(int_m)

                value.add_(reg)
                if use_wandb:
                    wandb.log({"scaled_frobenius_norm": reg})
                    # wandb.log({"train combined loss": value})

            optimizer.zero_grad()

            value.backward()

            optimizer.step()
        else:
            if matrix_regularizer is not None:
                with torch.no_grad():
                    if matrix_regularizer:
                        int_m, _, _, _ = ism(model, dim=dim, lb=LB, ub=UB, is_NN=True, use_grad=True)
                        int_m = replace_nan(int_m)
                        reg = 100 * torch.abs(int_m).sum()
                    value.add_(reg)
                    if use_wandb:
                        wandb.log({"scaled_frobenius_norm": reg})
                        wandb.log({"train combined loss": value})
                        wandb.log({"train objective loss": value})
        cumulative_sum += value.item()
        total_items += len(data)
        if batch_size == 1:
            batch_size = len(data)

    return cumulative_sum * batch_size / total_items


def compute_loss_batch(batch, model, loss_object, train=True):
    value = 0
    if not train:
        model.eval()
        with torch.no_grad():
            data, label = batch
            prediction = model(data)
            value = loss_object(prediction, label.view((-1, 1)))
    else:
        data, label = batch
        prediction = model(data)
        value = loss_object(prediction, label.view((-1, 1)))
    return value


def forward_no_residual(x, weights):
    layers = len(weights)
    for i in range(layers):
        x = weights[i] @ np.array(x).reshape(-1, 1)
        x = sigmoid(x)
    return x


def inspect_layer_SVD(model: nn.Module, layer_name: str = None):
    weight_matrix = None
    for name, w in model.named_parameters():
        if layer_name + ".weight" == name:
            weight_matrix = w.cpu().detach().numpy()
            break

    U_, S_, V_ = np.linalg.svd(weight_matrix, full_matrices=False)
    ## Re-scale U and V with S
    U = U_ @ np.diag(S_)
    V = V_ @ np.diag(S_)
    # For U
    # these variable interact we know beforehand
    UcosineX1X3 = np.arccos(np.dot(U[:, 0], U[:, 2]))
    UcosineX1X4 = np.arccos(np.dot(U[:, 0], U[:, 3]))
    UcosineX2X4 = np.arccos(np.dot(U[:, 1], U[:, 3]))
    # These variables do not interact
    UcosineX2X1 = np.arccos(np.dot(U[:, 1], U[:, 0]))
    UcosineX3X4 = np.arccos(np.dot(U[:, 2], U[:, 3]))
    UcosineX2X3 = np.arccos(np.dot(U[:, 1], U[:, 2]))
    # For V
    # these variable interact we know beforehand
    VcosineX1X3 = np.arccos(np.dot(V[:, 0], V[:, 2]))
    VcosineX1X4 = np.arccos(np.dot(V[:, 0], V[:, 3]))
    VcosineX2X4 = np.arccos(np.dot(V[:, 1], V[:, 3]))
    # These variables do not interact
    VcosineX2X1 = np.arccos(np.dot(V[:, 1], V[:, 0]))
    VcosineX3X4 = np.arccos(np.dot(V[:, 2], V[:, 3]))
    VcosineX2X3 = np.arccos(np.dot(V[:, 1], V[:, 2]))
    print("Results for U:\n")
    print("------\"Interact\"-------|------Not interact-----|")
    print("|                      |                       |")
    print(f"| X1*X3 | X1*X4 | X2*X4| X2*X1  | X3*X4 | X2*X3 |")
    print(f"| {UcosineX1X3:0.3f} | {UcosineX1X4:0.3f} | {UcosineX2X4:0.3f}| {UcosineX2X4:0.3f}  | {UcosineX2X1:0.3f} "
          f"| {UcosineX3X4:0.3f} | {UcosineX2X3:0.3f} |")

    print("Results for V:\n")
    print("------\"Interact\"-------|------Not interact-----|")
    print("|                      |                       |")
    print(f"| X1*X3 | X1*X4 | X2*X4| X2*X1  | X3*X4 | X2*X3 |")
    print(f"| {VcosineX1X3:0.3f} | {VcosineX1X4:0.3f} | {VcosineX2X4:0.3f}| {VcosineX2X4:0.3f}  | {VcosineX2X1:0.3f} "
          f"| {VcosineX3X4:0.3f} | {VcosineX2X3:0.3f} |")
    pass


@torch.no_grad()
def replace_nan(tensor: torch.TensorType):
    assert tensor.dim() == 2, f"Tensor is not 2 dimensional, got {tensor.dim()} instead"
    # max_i , max_j = tuple(tensor.shape)
    tensor.data[torch.isnan(tensor.data)] = 0
    return tensor


@torch.no_grad()
def simulate_single(weights: torch.Tensor):
    new_model = define_res_model(dim)
    vector_to_parameters(weights, new_model.parameters())
    new_model.cuda()
    loss_object = nn.MSELoss(reduction="mean")
    # pbar = trange(10, unit="carrots")
    train_loader, (train_features, train_targets), test_loader, (test_features, test_targets) = get_train_val_data(
        testF,
        batch_size=BATCH_SIZE,
        N_samples=N_samples, LB=LB, UB=UB)
    int_m, _, _, _ = ism(new_model, dim=dim, lb=LB, ub=UB, is_NN=True, use_grad=False)
    int_m = replace_nan(int_m)
    l = loss_one_epoch(new_model, data_loader=train_loader, optimizer=None, matrix_regularizer=int_m,
                       loss=loss_object, train=False,
                       L1_lambda=0)
    return -l

    # # I'm going to train


def simulate(batch_weights):
    rewards = []
    i = 0
    n = len(batch_weights)

    pbar = tqdm.tqdm(n, dynamic_ncols=True)
    for weights in batch_weights:
        rewards.append(simulate_single(weights))
        pbar.set_description(f"Element {i} out of {n} processed")
        pbar.update(1)
        i += 1
    return torch.tensor(rewards)


def run_ES_with_regularization():
    global EPOCHS, BATCH_SIZE, N_samples, dim, LB, UB, testF
    testF = TestFunction()
    HIDDEN_NEURONS = 10
    CLASSES = 1
    EPOCHS = 10
    BATCH_SIZE = 128
    LEARNING_RATE = 0.002
    L1_reg = 1.5
    L2_reg = 0
    N_samples = 10000
    dim = 4
    LB = 1
    UB = 10
    # W & B initialization
    wandb.init(
        project="nn_interaction",
        entity="luis_alfredo",
        tags=["interaction analysis"],
        reinit=True
    )

    generations = 100
    model = define_res_model(dim)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # loss_object = nn.MSELoss(reduction="mean")
    # pbar = trange(EPOCHS, unit="carrots")
    # # I'm going to train
    model.eval()
    # model.cuda()
    ########################## training with evograd     ###############################################################
    # mu = torch.randn(4, requires_grad=True)  # population mean
    n = len(parameters_to_vector(model.parameters()))
    mu = torch.rand(n, requires_grad=True)
    npop = 50  # population size
    std = 0.05  # noise standard deviation
    alpha = 0.03  # learning rate
    p = Normal(mu, std)
    # env = gym.make("CartPole-v0")

    for t in range(generations):
        sample = p.sample(npop)
        fitnesses = simulate(sample)
        scaled_fitnesses = (fitnesses - fitnesses.mean()) / fitnesses.std()
        # scaled_fitnesses.to(torch.device("cuda"))
        mean = expectation(scaled_fitnesses, sample, p=p)
        # mean.to(torch.device("cuda"))
        mean.backward()

        with torch.no_grad():
            mu += alpha * mu.grad
            mu.grad.zero_()

        print("step: {}, mean fitness: {:0.5}".format(t, float(fitnesses.mean())))
    torch.save(mu, "population_mean_regularized")

    # now I evaluate the model of the mean

    train_loader, (train_features, train_targets), test_loader, (test_features, test_targets) = get_train_val_data(
        testF,
        batch_size=BATCH_SIZE,
        N_samples=N_samples, LB=LB, UB=UB)

    best_mean = torch.load("population_mean_regularized")
    vector_to_parameters(best_mean, model.parameters())
    model.to(test_features.device)
    model.eval()
    y_hat = model(test_features)
    loss_object = nn.MSELoss()
    print(f"Loss over the entire test dataset after {generations} training generations \n"
          f" {loss_object(y_hat, test_targets).item():0.5f}")

    model.cuda()
    lambda_matrix, fhat_archive, f_archive, fp1 = ism(model, dim=dim, lb=LB, ub=UB, is_NN=True, use_grad=False)
    # reg_matrix = replace_nan(lambda_matrix)
    # reg_value = torch.linalg.matrix_norm(reg_matrix)
    non_seps, seps, theta, epsilon = dsm(fhat_archive.detach().cpu().numpy(),
                                         lambda_matrix.detach().cpu().numpy(),
                                         f_archive.detach().cpu().numpy(),
                                         fp1.detach().cpu().numpy(),
                                         dim)
    torch.set_printoptions(linewidth=200)
    model.to(test_features.device)
    print(f"The estimated lamda matrix of neural network after {generations} of training:\n{lambda_matrix}")
    print(f"The estimated interaction matrix of neural network after {generations} of training:\n {theta.astype(int)}")
    print(f"The real interaction matrix is:\n{testF.get_interaction_matrix()}")
    rho_results = rho_metrics(theta, testF.get_interaction_matrix())
    print(f"Rho results: rho 1: {rho_results[0]:0.3f} ,rho 2: {rho_results[1]:0.3f},rho 3:{rho_results[2]:0.3f}")
    print(f"Ratio of MSE_Test/MSE_Train:\n\t "
          f"{loss_object(y_hat, test_targets).item() / loss_object(model(train_features), train_targets).item():0.5f}")
    net = NetWrapper(model)

    visualizer = ResidualsPlot(net, hist=False, qqplot=True)

    visualizer.fit(train_features.detach().numpy(), train_targets.detach().numpy())  # Fit the training data to the
    # visualizer
    visualizer.score(test_features.detach().numpy(),
                     test_targets.detach().numpy())  # Evaluate the model on the test data

    visualizer.show()

    ### Now prediction error
    visualizer = PredictionError(net)
    visualizer.fit(train_features.detach().numpy(), train_targets.detach().numpy())  # Fit the training data to the
    # visualizer
    visualizer.score(test_features.detach().numpy(),
                     test_targets.detach().numpy())  # Evaluate the model on the test data
    visualizer.show()
    t, p_value = stats.kstest((y_hat - test_targets).detach().numpy(), 'norm')
    print(f"\n the Kolmogov-Smirnov test for the residuals is: "
          f"statistic: {t},p-value: {p_value}")


def test_model(model: nn.Module, loss_object, test_features: torch.Tensor, test_targets: torch.Tensor):
    model.eval()
    model.to(test_features.device)
    y_hat = model(test_features)
    print(f"Loss over the entire test {loss_object(y_hat, test_targets).item():0.5f}")

    lambda_matrix, fhat_archive, f_archive, fp1 = ism(model, dim=dim, lb=LB, ub=UB, is_NN=True, use_grad=False)

    reg_matrix = replace_nan(lambda_matrix)
    reg_value = torch.linalg.matrix_norm(reg_matrix)
    non_seps, seps, theta, epsilon = dsm(fhat_archive.detach().numpy(),
                                         lambda_matrix.detach().numpy(),
                                         f_archive.detach().numpy(),
                                         fp1.detach().numpy(),
                                         dim)
    torch.set_printoptions(linewidth=200)
    print(f"The estimated lamda matrix of neural network after {EPOCHS} of training:\n{lambda_matrix}")
    print(f"The estimated interaction matrix of neural network after {EPOCHS} of training:\n {theta.astype(int)}")
    print(f"The real interaction matrix is:\n{testF.get_interaction_matrix()}")
    rho_results = rho_metrics(theta, testF.get_interaction_matrix())
    print(f"Rho results: rho 1: {rho_results[0]:0.3f} ,rho 2: {rho_results[1]:0.3f},rho 3:{rho_results[2]:0.3f}")
    print(f"Ratio of MSE_Test/MSE_Train:\n\t "
          f"{loss_object(y_hat, test_targets).item() / loss_object(model(train_features), train_targets).item():0.5f}")
    net = NetWrapper(model)

    visualizer = ResidualsPlot(net, hist=False, qqplot=True)

    visualizer.fit(train_features.detach().numpy(), train_targets.detach().numpy())  # Fit the training data to the
    # visualizer
    visualizer.score(test_features.detach().numpy(),
                     test_targets.detach().numpy())  # Evaluate the model on the test data
    visualizer.show()

    ### Now prediction error
    visualizer = PredictionError(net)
    visualizer.fit(train_features.detach().numpy(), train_targets.detach().numpy())  # Fit the training data to the
    # visualizer
    visualizer.score(test_features.detach().numpy(),
                     test_targets.detach().numpy())  # Evaluate the model on the test data
    visualizer.show()
    t, p_value = stats.kstest((y_hat - test_targets).detach().numpy(), 'norm')
    print(f"\n the Kolmogov-Smirnov test for the residuals is: "
          f"statistic: {t},p-value: {p_value}")


def run_interaction_experiment(use_noisy_data: bool = True, use_frobenious: bool = False,
                               use_wandb: bool = False,
                               use_absolute_value: bool = False,
                               res_model: bool = False
                               ):
    layers = []
    lr = 0.02

    if use_noisy_data:
        layers = [114, 25, 47]
        lr = 0.02079
        print("EXPERIMENT with noise")

    else:
        layers = [60, 41, 128, 125, 114]
        lr = 0.09986
        print("EXPERIMENT without noise")

    np.set_printoptions(precision=3)
    global EPOCHS, BATCH_SIZE, N_samples, dim, LB, UB, testF
    testF = TestFunction()
    HIDDEN_NEURONS = 10
    CLASSES = 1
    EPOCHS = 100
    BATCH_SIZE = 128
    LEARNING_RATE = 0.002
    L1_reg = 0
    L2_reg = 0.005
    N_samples = 10000
    dim = 4
    LB = 1
    UB = 10

    if res_model:
        model = define_res_model(dim)
    else:
        model = define_model(layer_neurons=layers, input_size=dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=L2_reg)
    loss_object = nn.MSELoss(reduction="mean")
    pbar = trange(EPOCHS, unit="carrots")
    train_loader, (train_features, train_targets), test_loader, (test_features, test_targets) = get_train_val_data(
        testF,
        noisy=use_noisy_data,
        batch_size=BATCH_SIZE,
        N_samples=N_samples, LB=LB, UB=UB)

    # W & B initialization
    if use_wandb:
        subname = "non regularised"
        if use_frobenious:
            type_frobenius = "L1" if use_absolute_value else "L2"
            subname = f"regularized_{type_frobenius}"
        name = f"{subname}_diff_{float(np.random.rand(1)) * 1000:3.0f}"
        wandb.init(
            project="nn_interaction",
            entity="luis_alfredo",
            name=name,
            tags=["interaction analysis"],
            reinit=True,
            save_code=True
        )
        wandb.watch(model)
    # # I'm going to train
    model.train()
    model.cuda()
    int_m = None
    for e in pbar:
        l = loss_one_epoch(model, data_loader=train_loader, matrix_regularizer=use_frobenious, optimizer=optimizer,
                           loss=loss_object, train=True,
                           L1_lambda=L1_reg, use_wandb=use_wandb)
        pbar.set_description(f"loss:{l:0.3f}, , epoch: {e}")

    # inspect_layer_SVD(model, "0")
    modules = [m for (name, m) in model.named_modules() if isinstance(m, nn.Linear)]
    number_of_modules = len(modules)
    parameters_to_prune = list(zip(modules, ["weight"] * number_of_modules))
    sparsity = 0.9

    # prune.global_unstructured(
    #     parameters_to_prune,
    #     pruning_method=prune.L1Unstructured,
    #     amount=sparsity
    # )
    # sum_non_zero = sum([b.sum() for b in model.buffers()])
    # total_elems = sum([b.nelement() for b in model.buffers()])
    # print(
    #     "Global sparsity: {:.2f}%".format(sum_non_zero / total_elems)
    # )
    # make permanent the pruning acording to https://pytorch.org/tutorials/intermediate/pruning_tutorial.html#remove-pruning-re-parametrization
    # for m in modules:
    #     prune.remove(m, 'weight')

    # print(f"Pruned network with level of sparsity: {sparsity}")

    model.eval()
    model.to(test_features.device)
    y_hat = model(test_features)
    print(f"Loss over the entire test dataset after {EPOCHS} training epochs with L1={L1_reg} and L2={L2_reg}:\n"
          f" {loss_object(y_hat, test_targets).item():0.5f}")
    model.cuda()
    lambda_matrix, fhat_archive, f_archive, fp1 = ism(model, dim=dim, lb=LB, ub=UB, is_NN=True, use_grad=False)
    # reg_matrix = replace_nan(lambda_matrix)
    # reg_value = torch.linalg.matrix_norm(reg_matrix)
    non_seps, seps, theta, epsilon = dsm(fhat_archive.detach().cpu().numpy(),
                                         lambda_matrix.detach().cpu().numpy(),
                                         f_archive.detach().cpu().numpy(),
                                         fp1.detach().cpu().numpy(),
                                         dim)
    torch.set_printoptions(linewidth=200)
    model.to(train_features.device)
    if use_wandb:
        thing = replace_nan(lambda_matrix).detach().cpu().numpy()
        dataFrame = pd.DataFrame(thing)

        table = wandb.Table(dataframe=dataFrame)
        wandb.log({"final_lambda_matrix": table})

    print(f"The estimated lamda matrix of neural network after {EPOCHS} of training:\n{lambda_matrix}")
    print(f"The estimated interaction matrix of neural network after {EPOCHS} of training:\n {theta.astype(int)}")
    print(f"The real interaction matrix is:\n{testF.get_interaction_matrix()}")
    rho_results = rho_metrics(theta, testF.get_interaction_matrix())
    print(f"Rho results: rho 1: {rho_results[0]:0.3f} ,rho 2: {rho_results[1]:0.3f},rho 3:{rho_results[2]:0.3f}")
    print(f"Ratio of MSE_Test/MSE_Train:\n\t "
          f"{loss_object(y_hat, test_targets).item() / loss_object(model(train_features), train_targets).item():0.5f}")
    net = NetWrapper(model)

    msg = ""
    if use_noisy_data:
        msg += "_noisy_data"
    if use_frobenious:
        thing = "L1" if use_absolute_value else "L2"
        msg += f"_frobenius_reg_{thing}"

    plt.figure()
    visualizer = ResidualsPlot(net, hist=False, qqplot=True)

    visualizer.fit(train_features.detach().numpy(), train_targets.detach().numpy())  # Fit the training data to the
    # visualizer
    visualizer.score(test_features.detach().numpy(),
                     test_targets.detach().numpy())  # Evaluate the model on the test data
    visualizer.show(f"images/residual_plot_{msg}.png")
    plt.close()
    if use_wandb:
        image = Image.open(f"images/residual_plot_{msg}.png")
        wandb.log({"residual plot": wandb.Image(image)})

    ### Now prediction error
    plt.figure()
    visualizer = PredictionError(net)
    visualizer.fit(train_features.detach().numpy(), train_targets.detach().numpy())  # Fit the training data to the
    # visualizer
    visualizer.score(test_features.detach().numpy(),
                     test_targets.detach().numpy())  # Evaluate the model on the test data
    visualizer.show(f"images/prediction_error_{msg}.png")
    plt.close()
    if use_wandb:
        image = Image.open(f"images/prediction_error_{msg}.png")
        wandb.log({"prediction error": wandb.Image(image)})
    t, p_value = stats.kstest((y_hat - test_targets).detach().numpy(), 'norm')
    if use_wandb:
        conf = {
            "noisy": use_noisy_data,
            "use_frobenius": use_frobenious,
            "lr": lr,
            "max_epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "l2": L2_reg,
            "l1": L1_reg,
            "n samples": N_samples,
            "input dimension": dim,
            "upper bound": UB,
            "lower bound": LB
        }
        wandb.config.update(conf)
        wandb.join()
        wandb.finish()
    print(f"\n the Kolmogov-Smirnov test for the residuals is: "
          f"statistic: {t},p-value: {p_value}")

    # print("\nNow with my Hand-crafted Net")
    # crafted_net = SimpleNet()
    #
    # crafted_net.train()
    # optimizer = torch.optim.Adam(crafted_net.parameters(), lr=lr)
    # for e in pbar:
    #     l = loss_one_epoch(crafted_net, data_loader=train_loader, optimizer=optimizer, loss=loss_object, train=True,
    #                        L1_lambda=0)
    #     pbar.set_description(f"loss:{l:0.3f} , epoch: {e} ")
    # crafted_net.eval()
    # crafted_net.to(test_features.device)
    # y_hat = crafted_net(test_features)
    # print(f"Loss over the entire test dataset after {EPOCHS} training epochs with L1={0} and L2={L2_reg}:\n"
    #       f" {loss_object(y_hat, test_targets).item():0.5f}")
    #
    # lambda_matrix, fhat_archive, f_archive, fp1 = ism(crafted_net, dim=dim, lb=LB, ub=UB, is_NN=True)
    # non_seps, seps, theta, epsilon = dsm(fhat_archive.detach().numpy(),
    #                                      lambda_matrix.detach().numpy(),
    #                                      f_archive.detach().numpy(),
    #                                      fp1.detach().numpy(),
    #                                      dim)
    # torch.set_printoptions(linewidth=200)
    # print(f"The estimated lamda matrix of neural network after {EPOCHS} of training:\n {lambda_matrix}")
    # print(f"The estimated interaction matrix of neural network after {EPOCHS} of training:\n {theta.astype(int)}")
    # print(f"The real interaction matrix is:\n{testF.get_interaction_matrix()}")
    # rho_results = rho_metrics(theta, testF.get_interaction_matrix())
    # print(f"Rho results: rho 1: {rho_results[0]:0.3f} ,rho 2: {rho_results[1]:0.3f},rho 3:{rho_results[2]:0.3f}")
    # print(f"Ratio of MSE_Test/MSE_Train:\n\t "
    #       f"{loss_object(y_hat, test_targets).item() / loss_object(crafted_net(train_features), train_targets).item():0.5f}")
    #
    # net = NetWrapper(crafted_net)
    #
    # visualizer = ResidualsPlot(net, hist=False, qqplot=True)
    #
    # visualizer.fit(train_features.detach().numpy(), train_targets.detach().numpy())  # Fit the training data to the
    # # visualizer
    # visualizer.score(test_features.detach().numpy(),
    #                  test_targets.detach().numpy())  # Evaluate the model on the test data
    # visualizer.show()
    #
    # ### Now prediction error
    # visualizer = PredictionError(net)
    # visualizer.fit(train_features.detach().numpy(), train_targets.detach().numpy())  # Fit the training data to the
    # # visualizer
    # visualizer.score(test_features.detach().numpy(),
    #                  test_targets.detach().numpy())  # Evaluate the model on the test data
    # visualizer.show()
    # t, p_value = stats.kstest((y_hat - test_targets).detach().numpy(), 'norm')
    # print(f"\n The Kolmogov-Smirnov test for the residuals is: "
    #       f"statistic: {t}, p-value: {p_value:0.5f}")

    network_type = "ResNet" if res_model else "FC"
    torch.save(model.state_dict(), f"models/{network_type}_network{msg}")
    # torch.save(crafted_net.state_dict(), "models/hand_craft_net")


def create_heatmap(matrix: typing.Union[torch.Tensor, np.array], title: str = "", save_path: str = "images/") -> str:
    N = matrix.shape[0]
    M = matrix.shape[1]
    fig, ax = plt.subplots()
    im = ax.imshow(matrix)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(M))

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #          rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(N):
        for j in range(M):
            text = ax.text(j, i, matrix[i, j],
                           ha="center", va="center", color="w")

    ax.set_title(title)
    fig.tight_layout()
    path = save_path + title.replace(" ", "_") + ".png"
    plt.savefig(path)
    return path


# This function assumes that there are 3 models, non regularised, regularised with L2 norm in the interaction matrix
# and regularised with L1 norm in the interaction matrix. This function compares all of the above models
# It also asumes that they are all in the models/ folder
def compare_models(network_type: str = "FC", use_wandb: bool = False):
    assert network_type == "FC" or network_type == "ResNet", "Network type not recognized it needs to be either FC or ResNet"
    global L2_reg_model, L1_reg_model
    from sklearn.manifold import MDS, TSNE
    testF = TestFunction()
    HIDDEN_NEURONS = 10
    CLASSES = 1
    EPOCHS = 100
    BATCH_SIZE = 128
    LEARNING_RATE = 0.002
    L1_reg = 0
    L2_reg = 0.005
    N_samples = 1000
    dim = 4
    LB = 1
    UB = 10
    layers = [114, 25, 47]
    if use_wandb:
        wandb.init(
            project="nn_interaction",
            entity="luis_alfredo",
            name=f"{network_type} model analysis",
            tags=["interaction analysis"],
            reinit=True,
        )

    if network_type == "FC":
        non_reg_model = define_model(layers, input_size=dim)
        L2_reg_model = define_model(layers, input_size=dim)
        L1_reg_model = define_model(layers, input_size=dim)
        non_reg_model.load_state_dict(torch.load("models/FC_network_noisy_data"))
        L1_reg_model.load_state_dict(torch.load("models/FC_network_noisy_data_frobenius_reg_L1"))
        L2_reg_model.load_state_dict(torch.load("models/FC_network_noisy_data_frobenius_reg_L2"))
    elif network_type == "ResNet":
        non_reg_model = define_res_model(dim)
        L2_reg_model = define_res_model(dim)
        L1_reg_model = define_res_model(dim)
        non_reg_model.load_state_dict(torch.load("models/ResNet_network_noisy_data"))
        L1_reg_model.load_state_dict(torch.load("models/ResNet_network_noisy_data_frobenius_reg_L1"))
        L2_reg_model.load_state_dict(torch.load("models/ResNet_network_noisy_data_frobenius_reg_L2"))

    train_loader, (train_features, train_targets), test_loader, (test_features, test_targets) = get_train_val_data(
        testF,
        noisy=True,
        batch_size=128,
        N_samples=N_samples,
        LB=LB,
        UB=UB)
    L1_reg_model.cuda()
    L2_reg_model.cuda()
    lambda_matrix_L1, _, _, _ = ism(L1_reg_model, dim=dim, lb=LB, ub=UB, is_NN=True,
                                    use_grad=False)
    lambda_matrix_L2, _, _, _ = ism(L2_reg_model, dim=dim, lb=LB, ub=UB, is_NN=True,
                                    use_grad=False)
    if use_wandb:
        columns = ["Type of regularisation", "Lambda matrix"]
        table = wandb.Table(columns=columns)
        thing1 = replace_nan(lambda_matrix_L1).detach().cpu().numpy()
        thing2 = replace_nan(lambda_matrix_L2).detach().cpu().numpy()
        path1 = create_heatmap(thing1, "Lambda Matrix L1")
        path2 = create_heatmap(thing2, "Lambda Matrix L2")
        image1 = Image.open(path1)
        image2 = Image.open(path2)
        table.add_data("L1", wandb.Image(image1))
        table.add_data("L2", wandb.Image(image2))
        wandb.log({"Lambda matrices": table})

    non_reg_model.to(train_features.device)
    L1_reg_model.to(train_features.device)
    L2_reg_model.to(train_features.device)

    L1_reg_model_wrapper = NetWrapper(L1_reg_model)
    L2_reg_model_wrapper = NetWrapper(L2_reg_model)
    non_reg_model_wrapper = NetWrapper(non_reg_model)

    # Check how well each models adapts to the test data

    # How well  non reg does
    plt.figure()
    visualizer = ResidualsPlot(non_reg_model_wrapper, hist=False, qqplot=True)
    visualizer.fit(train_features.detach().numpy(), train_targets.detach().numpy())  # Fit the training data to the
    # visualizer
    visualizer.score(test_features.detach().numpy(),
                     test_targets.detach().numpy())  # Evaluate the model on the test data

    title = f"Residuals of non-regularised model withe type {network_type}"

    visualizer.finalize()
    plt.title(title, fontsize=12)
    plt.savefig(f"images/{network_type}_residuals_noReg_data.png")
    plt.close()
    # Now the prediction error
    plt.figure()
    visualizer = PredictionError(non_reg_model_wrapper)
    visualizer.fit(train_features.detach().numpy(), train_targets.detach().numpy())  # Fit the training data to the
    # visualizer
    visualizer.score(test_features.detach().numpy(),
                     test_targets.detach().numpy())  # Evaluate the model on the test data

    title = f"Prediction error of non-regularised model with type {network_type}"

    visualizer.finalize()
    plt.title(title, fontsize=12)
    plt.savefig(f"images/{network_type}_pred_noReg_data.png")
    plt.close()

    if use_wandb:
        image1 = Image.open(f"images/{network_type}_residuals_noReg_data.png")
        image2 = Image.open(f"images/{network_type}_pred_noReg_data.png")
        wandb.log({f"{network_type}_residuals_noReg_data.png": wandb.Image(image1)})
        wandb.log({f"{network_type}_pred_noReg_data.png": wandb.Image(image2)})

    # How well  L1 reg does
    plt.figure()
    visualizer = ResidualsPlot(L1_reg_model_wrapper, hist=False, qqplot=True)
    visualizer.fit(train_features.detach().numpy(), train_targets.detach().numpy())  # Fit the training data to the
    # visualizer
    visualizer.score(test_features.detach().numpy(),
                     test_targets.detach().numpy())  # Evaluate the model on the test data

    title = f"Residuals of L1 model with type {network_type}"
    visualizer.finalize()
    plt.title(title, fontsize=12)
    plt.savefig(f"images/{network_type}_residuals_L1reg_data.png")
    plt.close()

    # Now the prediction error
    plt.figure()
    visualizer = PredictionError(L1_reg_model_wrapper)
    visualizer.fit(train_features.detach().numpy(), train_targets.detach().numpy())  # Fit the training data to the
    # visualizer
    visualizer.score(test_features.detach().numpy(),
                     test_targets.detach().numpy())  # Evaluate the model on the test data

    title = f"Prediction error of L1 model with type {network_type}"

    visualizer.finalize()
    plt.title(title, fontsize=12)
    plt.savefig(f"images/{network_type}_pred_L1Reg_data.png")
    plt.close()

    if use_wandb:
        image1 = Image.open(f"images/{network_type}_residuals_L1Reg_data.png")
        image2 = Image.open(f"images/{network_type}_pred_L1Reg_data.png")
        wandb.log({f"{network_type}_residuals_L1Reg_data.png": wandb.Image(image1)})
        wandb.log({f"{network_type}_pred_L1Reg_data.png": wandb.Image(image2)})
    # How well  L2 reg does
    plt.figure()
    visualizer = ResidualsPlot(L2_reg_model_wrapper, hist=False, qqplot=True)
    visualizer.fit(train_features.detach().numpy(), train_targets.detach().numpy())  # Fit the training data to the
    # visualizer
    visualizer.score(test_features.detach().numpy(),
                     test_targets.detach().numpy())  # Evaluate the model on the test data

    title = f"Residuals of L2 model with type {network_type}"
    visualizer.finalize()
    plt.title(title, fontsize=12)
    plt.savefig(f"images/{network_type}_residuals_L2reg_data.png")
    plt.close()
    # Now the prediction error
    plt.figure()
    visualizer = PredictionError(L2_reg_model_wrapper)
    visualizer.fit(train_features.detach().numpy(), train_targets.detach().numpy())  # Fit the training data to the
    # visualizer
    visualizer.score(test_features.detach().numpy(),
                     test_targets.detach().numpy())  # Evaluate the model on the test data

    title = f"Prediction error of L2 model with type {network_type}"

    visualizer.finalize()
    plt.title(title, fontsize=12)
    plt.savefig(f"images/{network_type}_pred_L2Reg_data.png")
    plt.close()

    if use_wandb:
        image1 = Image.open(f"images/{network_type}_residuals_L2Reg_data.png")
        image2 = Image.open(f"images/{network_type}_pred_L2Reg_data.png")
        wandb.log({f"{network_type}_residuals_L2Reg_data.png": wandb.Image(image1)})
        wandb.log({f"{network_type}_pred_L2Reg_data.png": wandb.Image(image2)})

    # Comparison of models against eachother
    y_non_regularized = non_reg_model(train_features).detach().numpy()
    y_L1_regularized = L1_reg_model(train_features).detach().numpy()
    y_L2_regularized = L2_reg_model(train_features).detach().numpy()

    # L1 vs No Reg
    plt.figure()
    prediction_error_vis_L1 = PredictionError(L1_reg_model_wrapper)
    prediction_error_vis_L1.fit(train_features.detach().numpy(), y_non_regularized)
    prediction_error_vis_L1.score(test_features.detach().numpy(),
                                  test_targets.detach().numpy())
    title = f"Agreement between  L1 regularized and non-regularised on {network_type} models"

    prediction_error_vis_L1.finalize()
    plt.title(title, fontsize=12)
    plt.savefig(f"images/{network_type}_L1_noReg.png")
    plt.close()
    if use_wandb:
        image = Image.open(f"images/{network_type}_L1_noReg.png")
        wandb.log({f"{network_type}_L1_noReg": wandb.Image(image)})

    # L2 vs No Reg
    plt.figure()
    prediction_error_vis_L2 = PredictionError(L2_reg_model_wrapper)
    prediction_error_vis_L2.fit(train_features.detach().numpy(), y_non_regularized)
    prediction_error_vis_L2.score(test_features.detach().numpy(),
                                  test_targets.detach().numpy())
    title = f"agreement between L2 regularized and non-regularised on {network_type} models"
    prediction_error_vis_L2.finalize()
    plt.title(title, fontsize=15)
    plt.savefig(f"images/{network_type}_L2_noReg.png")
    plt.close()
    if use_wandb:
        image = Image.open(f"images/{network_type}_L2_noReg.png")
        wandb.log({f"{network_type}_L2_noReg": wandb.Image(image)})
    # L1 vs L2
    plt.figure()
    prediction_error_vis_L2_L1 = PredictionError(L1_reg_model_wrapper)
    prediction_error_vis_L2_L1.fit(train_features.detach().numpy(), y_L2_regularized)
    prediction_error_vis_L2_L1.score(test_features.detach().numpy(), L2_reg_model(train_features).detach().numpy())
    title = f"Agreement between L2 regularized and L1 on {network_type} models"
    prediction_error_vis_L2_L1.finalize()
    plt.title(title, fontsize=20)
    plt.savefig(f"images/{network_type}_L2_L1.png")
    plt.close()

    if use_wandb:
        image = Image.open(f"images/{network_type}_L2_L1.png")
        wandb.log({f"{network_type}_L2_L1": wandb.Image(image)})

    # Compare with MSD and T-SNE
    non_reg_vector = parameters_to_vector(non_reg_model.parameters()).detach().numpy()
    L1_vector = parameters_to_vector(L1_reg_model.parameters()).detach().numpy()
    L2_vector = parameters_to_vector(L2_reg_model.parameters()).detach().numpy()
    cosine_similarity_non_reg_L1 = spatial.distance.cosine(non_reg_vector, L1_vector)
    cosine_similarity_non_reg_L2 = spatial.distance.cosine(non_reg_vector, L1_vector)
    cosine_similarity_L2_L1 = spatial.distance.cosine(L2_vector, L1_vector)

    data = np.stack((non_reg_vector, L1_vector, L2_vector))
    if use_wandb:
        wandb.log({"cosine_similarity between non reg and L1": cosine_similarity_non_reg_L1})
        wandb.log({"cosine_similarity between non reg and L2": cosine_similarity_non_reg_L2})
        wandb.log({"cosine_similarity between non L1 and L2": cosine_similarity_L2_L1})
    mds = MDS(n_components=2)
    tsne = TSNE(n_components=2)
    new_data_MDS = mds.fit_transform(data)
    new_data_TSNE = tsne.fit_transform(data)
    markers = ["d", "v", "s"]  # , "*", "^", "d", "v", "s", "*", "^"]
    # create MDS plot
    plt.figure()
    for xp, yp, m in zip(new_data_MDS[:, 0], new_data_MDS[:, 1], markers):
        plt.scatter(xp, yp, marker=m, s=200)
    plt.title("MDS projection", fontsize=20)
    plt.legend(["No Reg", "L1 reg", "L2 reg"], fontsize=15)
    plt.savefig(f"images/{network_type}_MDS.png")
    plt.close()
    # Create TSNE plot
    plt.figure()
    for xp, yp, m in zip(new_data_TSNE[:, 0], new_data_TSNE[:, 1], markers):
        plt.scatter(xp, yp, marker=m, s=200)
    plt.title("TSNE projection", fontsize=20)
    plt.legend(["No Reg", "L1 reg", "L2 reg"], fontsize=15)
    plt.savefig(f"images/{network_type}_TSNE.png")
    plt.close()
    if use_wandb:
        image1 = Image.open(f"images/{network_type}_MDS.png")
        image2 = Image.open(f"images/{network_type}_TSNE.png")
        wandb.log({f"{network_type}_MDS": wandb.Image(image1)})
        wandb.log({f"{network_type}_TSNE": wandb.Image(image2)})
        wandb.join()
        wandb.finish()


def f(x, y, w: list):
    salida = 0
    for i in range(len(x)):
        y_pred_i = forward_no_residual(x[i], w)
        y_real = y[i]
        salida += -y_real * np.log(y_pred_i) - (1 - y_real) * (np.log(1 - y_pred_i))
    return salida


def det_function(x):
    return x[0] ** 2 + (x[1] - x[2]) ** 2 + (x[2] - x[3]) ** 2 + (x[4] - x[5]) ** 2


def calculate_epsilon(a, k):
    evaluation = []

    for ele in range(k):
        w1 = np.random.rand(3, 4)
        w2 = np.random.rand(1, 3)
        weights = [w1, w2]
        evaluation.append(abs(f(data_X, data_Y, weights)))

    return a * min(evaluation)


def get_train_val_data(function: TestFunction, noisy: bool = True, batch_size: int = 128, N_samples: int = 1000,
                       LB: int = 1,
                       UB: int = 10):
    if noisy:
        features, targets = function.generate_noisy_data(N_samples, [LB, UB])
        features = torch.tensor(features, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)
        train = data_utils.TensorDataset(features, targets)
        train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)
        # Generate the test data
        test_features, test_targets = function.generate_noisy_data(N_samples, [LB, UB])
        test_features = torch.tensor(test_features, dtype=torch.float32)
        test_targets = torch.tensor(test_targets, dtype=torch.float32)
        test = data_utils.TensorDataset(test_features, test_targets)
        test_loader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=True)
        return train_loader, (features, targets), test_loader, (test_features, test_targets)
    else:
        features, targets = function.generate_data(N_samples, [LB, UB])
        features = torch.tensor(features, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)
        train = data_utils.TensorDataset(features, targets)
        train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)
        # Generate the test dataccc
        test_features, test_targets = function.generate_data(N_samples, [LB, UB])
        test_features = torch.tensor(test_features, dtype=torch.float32)
        test_targets = torch.tensor(test_targets, dtype=torch.float32)
        test = data_utils.TensorDataset(test_features, test_targets)
        test_loader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=True)
        return train_loader, (features, targets), test_loader, (test_features, test_targets)


def interact(X1: set, X2: set, p1: np.ndarray, y1, epsilon, UB, LB, shapes, xremain, type="neuron"):
    if type == "neuron":
        p2 = np.array(p1)
        p2[list(X1)] = [UB] * len(X1)
        p2_temp = unravel(p2, shapes)
        y2 = f(data_X, data_Y, p2_temp)

        delta1 = y1 - y2

        p3 = np.array(p1)
        p4 = np.array(p2)
        p3[list(X2)] = (UB + LB) / 2 * np.ones(len(X2))
        p4[list(X2)] = (UB + LB) / 2 * np.ones(len(X2))
        p3_temp = unravel(p3, shapes)
        p4_temp = unravel(p4, shapes)
        delta2 = f(data_X, data_Y, p3_temp) - f(data_X, data_Y, p4_temp)
        if abs(delta1 - delta2) > epsilon:
            if len(X2) == 1:
                X1 = X1.union(X2)
            else:
                k = len(X2) // 2
                list_X2 = list(X2)
                X_2_1 = set(list_X2[0:k])
                X_2_2 = set(list_X2[k:])

                X_1_1, xremain = interact(X1, X_2_1, p1, y1, epsilon, UB, LB, shapes, xremain, type)
                X_1_2, xremain = interact(X1, X_2_2, p1, y1, epsilon, UB, LB, shapes, xremain, type)
                X1 = X1.union(X_1_1, X_1_2)
        else:
            xremain = xremain.union(X2)
        return X1, xremain
    elif type == "normal":
        p2 = np.array(p1)
        p2[list(X1)] = [UB] * len(X1)

        y2 = det_function(p2)

        delta1 = y1 - y2

        p3 = np.array(p1)
        p4 = np.array(p2)
        p3[list(X2)] = (UB + LB) / 2 * np.ones(len(X2))
        p4[list(X2)] = (UB + LB) / 2 * np.ones(len(X2))
        delta2 = det_function(p3) - det_function(p4)
        if abs(delta1 - delta2) > epsilon:
            if len(X2) == 1:
                X1 = X1.union(X2)
            else:
                k = len(X2) // 2
                list_X2 = list(X2)
                X_2_1 = set(list_X2[0:k])
                X_2_2 = set(list_X2[k:])

                X_1_1, xremain = interact(X1, X_2_1, p1, y1, epsilon, UB, LB, shapes, xremain, type)
                X_1_2, xremain = interact(X1, X_2_2, p1, y1, epsilon, UB, LB, shapes, xremain, type)
                X1 = X1.union(X_1_1, X_1_2)
        else:
            xremain = xremain.union(X2)
        return X1, xremain


def RDG():
    w1 = np.random.rand(3, 4) * 2 - 1
    w2 = np.random.rand(1, 3) * 2 - 1
    weights = [w1, w2]

    epsilon = calculate_epsilon(10e-12, 1000)
    X1 = {0}
    X2 = set([])
    # X2 = {1,2,3,4,5}
    # xremain = {0,1,2,3,4,5}
    for elem in range(1, custom_len([k.shape for k in weights])):
        X2.add(elem)
        xremain.add(elem)

    LOWER_BOUND = -1
    UPPER_BOUND = 1
    p1 = LOWER_BOUND * np.ones(len(xremain))
    p1_temp = unravel(p1, shapes)
    y1 = f(data_X, data_Y, p1_temp)
    # y1 = det_function(p1)

    seps = []
    groups = []

    while len(xremain) > 0:
        xremain = set([])
        sub1_a, xremain = interact(X1, X2, p1, y1, epsilon, UPPER_BOUND, LOWER_BOUND, [], xremain)
        if len(sub1_a) == len(X1):
            if len(X1) == 1:
                seps.append(X1)
            else:
                groups.append(X1)
            if len(xremain) > 1:
                first_element = xremain.pop()
                X1 = {first_element}
                X2 = copy.copy(xremain)
            else:
                seps.append(xremain.pop())
                break
        else:
            X1 = copy.copy(sub1_a)
            X2 = copy.copy(xremain)
            if len(xremain) == 0:
                groups.append(X1)
                break
    return seps, groups


def ravel(x: list):
    out = []
    for elemento in x:
        out.extend(list(elemento.ravel()))
    return out


def unravel(x: np.ndarray, shapes: list):
    i = 0
    salida = []
    len_shapes = sum([np.prod(y) for y in shapes])
    assert len(x) == len_shapes, " El tamaño de x es {} cuando debería ser {}".format(len(x), len_shapes)
    for shape in shapes:
        n = shape[0] * shape[1]
        layer = np.array(x[i:i + n])
        layer = layer.reshape(shape)
        salida.append(layer)
        i += n
    return salida


def custom_len(array):
    return sum([np.prod(y) for y in array])


def main():
    # Todo: modify this code to use hydra when the experiments get more convoluted and with more options

    # Initialization of config
    np.set_printoptions(precision=3)
    testF = TestFunction()
    HIDDEN_NEURONS: int = 10
    CLASSES: int = 1
    BATCH_SIZE: int = 10
    LEARNING_RATE: float = 0.002
    L1_reg: int = 0
    L2_reg: int = 0
    N_samples: int = 10000
    dim: int = 4
    LB: int = 1
    UB: int = 10

    print(f"The original matrix:\n {testF.get_interaction_matrix()}")
    lambda_matrix, fhat_archive, f_archive, fp1 = ism(f=testF.forward(), dim=dim, lb=LB, ub=UB)
    print(f"The estimated interaction matrix:\n {lambda_matrix}")
    non_seps, seps, theta, epsilon = dsm(fhat_archive,
                                         lambda_matrix,
                                         f_archive,
                                         fp1,
                                         dim)
    rho_results = rho_metrics(theta, testF.get_interaction_matrix())

    print(f"Metrics for direct function \n")
    print(f"The estimated theta matrix:\n {theta.astype(int)}")
    print(f"Rho results: rho 1: {rho_results[0]:0.3f} ,rho 2: {rho_results[1]:0.3f},rho 3:{rho_results[2]:0.3f}")

    net = NeuralNet(input_size=dim, hidden_size=HIDDEN_NEURONS, num_classes=CLASSES, L1_reg=L1_reg, L2_reg=L2_reg,
                    learning_rate=LEARNING_RATE)
    # Generate train data
    features, targets = testF.generate_noisy_data(N_samples, [LB, UB])
    features = torch.tensor(features, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)
    train = data_utils.TensorDataset(features, targets)
    train_loader = data_utils.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    # Generate the test data
    test_features, test_targets = testF.generate_noisy_data(N_samples, [LB, UB])
    test_features = torch.tensor(test_features, dtype=torch.float32)
    test_targets = torch.tensor(test_targets, dtype=torch.float32)
    # loss function and optimizer
    loss_object = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    # One epoch of training
    l = loss_one_epoch(net, data_loader=train_loader, optimizer=optimizer, loss=loss_object, train=True)
    net.eval()
    net.cpu()
    y_hat = net(test_features)
    print(f"Loss over the entire test dataset after one training epoch with L1={L1_reg} and L2={L2_reg}:\n"
          f" {loss_object(y_hat, test_targets).item():0.5f}")
    theta_NN, _, _, _ = ism(net, dim=4, lb=LB, ub=UB, is_NN=True)
    print(f"The estimated interaction matrix of neural network after one epoch of training:\n {theta_NN}")

    # 10 epochs of training
    epochs = 10
    del net
    net = NeuralNet(input_size=dim, hidden_size=HIDDEN_NEURONS, num_classes=CLASSES, L1_reg=L1_reg, L2_reg=L2_reg,
                    learning_rate=LEARNING_RATE)

    # optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    # pbar = trange(epochs, unit="carrots")
    # for e in pbar:
    #     l = loss_one_epoch(net, data_loader=train_loader, optimizer=optimizer, loss=loss_object, train=true)
    #     pbar.set_description(f"loss:{l:0.3f} , epoch: {e} ")
    trainer = pl.Trainer(
        gpus=GPUtil.getFirstAvailable(),
        max_epochs=epochs,
        progress_bar_refresh_rate=20
    )
    trainer.fit(model=net, train_dataloader=train_loader)
    net.cpu()
    net.eval()
    y_hat = net(test_features)
    print(
        f"Loss over the entire test dataset after {epochs} training epochs with L1={L1_reg} and L2={L2_reg} trained with ADAM:"
        f" {loss_object(y_hat, test_targets).item():0.5f}")
    lambda_matrix, fhat_archive, f_archive, fp1 = ism(net, dim=dim, lb=LB, ub=UB, is_NN=True)
    non_seps, seps, theta, epsilon = dsm(fhat_archive.detach().numpy(),
                                         lambda_matrix.detach().numpy(),
                                         f_archive.detach().numpy(),
                                         fp1.detach().numpy(),
                                         dim)
    print(f"The estimated interaction matrix of neural network after {epochs} epochs of training:\n {theta_NN}")
    rho_results = rho_metrics(theta, testF.get_interaction_matrix())
    print(f"The estimated theta matrix:\n {theta.astype(int)}")
    print(f"Metrics for Neural Network \n")
    print(f"Rho results: rho 1: {rho_results[0]:0.3f} ,rho 2: {rho_results[1]:0.3f},rho 3:{rho_results[2]:0.3f}")


def get_nearest_point(reference: torch.Tensor, batch: torch.Tensor, metric: typing.Callable):
    if metric is None:
        metric = partial(F.mse_loss, reduction="sum")
    x, y = batch
    distances = torch.tensor(list(map(metric, batch[0], reference.repeat((len(batch[0]), 1)))), dtype=torch.float)

    nearest = torch.argmin(distances)
    x_nearest = x[nearest].clone()
    y_nearest = y[nearest].clone()
    return x_nearest, y_nearest


def batch_ism(batch:torch.Tensor,f: nn.Module, dim: int, ub: int, lb: int, use_grad: bool =
False):

    metric = partial(F.mse_loss, reduction="sum")

    x, y = batch
    x.cuda()
    y.cuda()

    max_x,index_max_x = torch.max(x,axis=0)
    min_x,index_min_x = torch.min(x,axis=0)


    if use_grad:
        # pdb.set_trace()
        f_archive = torch.zeros((dim, dim)) * torch.nan
        fhat_archive = torch.zeros((dim, 1), device=torch.device("cuda")) * torch.nan
        delta1 = torch.zeros((dim, dim)) * torch.nan
        delta2 = torch.zeros((dim, dim)) * torch.nan
        lambda_matrix = torch.zeros((dim, dim)) * torch.nan
        # Things for the neurlan network
        f_archive_NN = torch.zeros((dim, dim)) * torch.nan
        fhat_archive_NN = torch.zeros((dim, 1), device=torch.device("cuda")) * torch.nan
        delta1_NN = torch.zeros((dim, dim)) * torch.nan
        delta2_NN = torch.zeros((dim, dim)) * torch.nan
        lambda_matrix_NN = torch.zeros((dim, dim)) * torch.nan

        p1 = min_x

        p1_NN, fp1 = get_nearest_point(reference=min_x,batch=batch,metric=metric)

        p1_NN.to(torch.device("cuda"))
        # The output of the  neural network is being keot in parallel
        fp1_NN = f(p1_NN)
        counter = 0
        prev = 0
        prog = 0

        for i in range(dim - 1):
            if not torch.isnan(fhat_archive[i]):
                fp2 = fhat_archive[i]
                fp2_NN = fhat_archive_NN[i]
            else:
                p2 = copy.deepcopy(p1)
                p2[i] = max_x[i]
                p2.to(torch.device("cuda"))
                p2_NN, fp2 = get_nearest_point(reference=p2,batch=batch,metric=metric)
                fp2_NN = f(p2_NN)

                fhat_archive[i] = fp2
                fhat_archive_NN[i] = fp2_NN

            for j in range(i + 1, dim):
                counter += 1
            prev = prog
            prog = torch.tensor([counter // (dim * (dim - 1)) * 2 * 100])

            if prog % 5 == 0 and prog != prev:
                print("Progress: {}%".format(prog))
            if not torch.isnan(fhat_archive[j]):
                fp3 = fhat_archive[j]
            else:
                p3 = copy.deepcopy(p1)
                p3[j] = max_x[j]
                p3.to(torch.device("cuda"))
                p3_NN, fp3 = get_nearest_point(reference=p3,batch=batch,metric=metric)
                p3_NN.to(torch.device("cuda"))
                fp3_NN = f(p3_NN)

                fhat_archive[j] = fp3
                fhat_archive_NN[j] =fp3_NN

            p4 = copy.deepcopy(p1)
            p4[i] = max_x[j]
            p4[j] = max_x[j]
            p4.to(torch.device("cuda"))
            p4_NN, fp4 = get_nearest_point(reference=p4,batch=batch,metric=metric)
            p4_NN.to(torch.device("cuda"))
            fp3_NN = f(p4_NN)

            f_archive[i, j] = fp4
            f_archive[j, i] = fp4
            d1 = fp2 - fp1
            d2 = fp4 - fp3
            delta1[i, j] = d1
            delta2[i, j] = d2
            lambda_matrix[i, j] = torch.abs(d1 - d2)



            f_archive_NN[i, j] = fp4
            f_archive_NN[j, i] = fp4_NN
            d1 = fp2_NN - fp1_NN
            d2 = fp4_NN - fp3_NN
            delta1_NN[i, j] = d1
            delta2_NN[i, j] = d2
            # pdb.set_trace()
            lambda_matrix_NN[i,j]=torch.abs(d1-d2)
            # pdb.set_trace()
        return (lambda_matrix, fhat_archive, f_archive,
                fp1),(lambda_matrix_NN,fhat_archive_NN,f_archive_NN,fp1_NN)
    else:
        with torch.no_grad():
        # pdb.set_trace()

        f_archive = torch.zeros((dim, dim)) * torch.nan
        fhat_archive = torch.zeros((dim, 1), device=torch.device("cuda")) * torch.nan
        delta1 = torch.zeros((dim, dim)) * torch.nan
        delta2 = torch.zeros((dim, dim)) * torch.nan
        lambda_matrix = torch.zeros((dim, dim)) * torch.nan
        # Things for the neurlan network
        f_archive_NN = torch.zeros((dim, dim)) * torch.nan
        fhat_archive_NN = torch.zeros((dim, 1), device=torch.device("cuda")) * torch.nan
        delta1_NN = torch.zeros((dim, dim)) * torch.nan
        delta2_NN = torch.zeros((dim, dim)) * torch.nan
        lambda_matrix_NN = torch.zeros((dim, dim)) * torch.nan

        p1 = min_x

        p1_NN, fp1 = get_nearest_point(reference=min_x,batch=batch,metric=metric)

        p1_NN.to(torch.device("cuda"))
        # The output of the  neural network paralele bein computed
        fp1_NN = f(p1_NN)
        counter = 0
        prev = 0
        prog = 0

        for i in range(dim - 1):
            if not torch.isnan(fhat_archive[i]):
                fp2 = fhat_archive[i]
                fp2_NN = fhat_archive_NN[i]
            else:
                p2 = copy.deepcopy(p1)
                p2[i] = max_x[i]
                p2.to(torch.device("cuda"))
                p2_NN, fp2 = get_nearest_point(reference=p2,batch=batch,metric=metric)
                fp2_NN = f(p2_NN)

                fhat_archive[i] = fp2
                fhat_archive_NN[i] = fp2_NN

            for j in range(i + 1, dim):
                counter += 1
            prev = prog
            prog = torch.tensor([counter // (dim * (dim - 1)) * 2 * 100])

            if prog % 5 == 0 and prog != prev:
                print("Progress: {}%".format(prog))
            if not torch.isnan(fhat_archive[j]):
                fp3 = fhat_archive[j]
            else:
                p3 = copy.deepcopy(p1)
                p3[j] = max_x[j]
                p3.to(torch.device("cuda"))
                p3_NN, fp3 = get_nearest_point(reference=p3,batch=batch,metric=metric)
                p3_NN.to(torch.device("cuda"))
                fp3_NN = f(p3_NN)

                fhat_archive[j] = fp3
                fhat_archive_NN[j] =fp3_NN

            p4 = copy.deepcopy(p1)
            p4[i] = max_x[j]
            p4[j] = max_x[j]
            p4.to(torch.device("cuda"))
            p4_NN, fp4 = get_nearest_point(reference=p4,batch=batch,metric=metric)
            p4_NN.to(torch.device("cuda"))
            fp3_NN = f(p4_NN)

            f_archive[i, j] = fp4
            f_archive[j, i] = fp4
            d1 = fp2 - fp1
            d2 = fp4 - fp3
            delta1[i, j] = d1
            delta2[i, j] = d2
            lambda_matrix[i, j] = torch.abs(d1 - d2)



            f_archive_NN[i, j] = fp4
            f_archive_NN[j, i] = fp4_NN
            d1 = fp2_NN - fp1_NN
            d2 = fp4_NN - fp3_NN
            delta1_NN[i, j] = d1
            delta2_NN[i, j] = d2
            # pdb.set_trace()
            lambda_matrix_NN[i,j]=torch.abs(d1-d2)
            # pdb.set_trace()
        return (lambda_matrix, fhat_archive, f_archive,
                fp1),(lambda_matrix_NN,fhat_archive_NN,f_archive_NN,fp1_NN)

    


if __name__ == '__main__':
    # run_ES_with_regularization()
    run_interaction_experiment(use_frobenious=False, use_wandb=False, use_absolute_value=False, res_model=True)
    run_interaction_experiment(use_frobenious=True, use_wandb=False, use_absolute_value=False, res_model=True)
    run_interaction_experiment(use_frobenious=True, use_wandb=False, use_absolute_value=True, res_model=True)
    compare_models("ResNet", use_wandb=True)

    # run_interaction_experiment(use_frobenious=False, use_wandb=True)

    # print_hi('PyCharm')
    # w1 = np.random.rand(3, 4) * 4 +1
    # w2 = np.random.rand(1, 3) * 4 + 1
    # weights = [w1, w2]
    #
    # epsilon = calculate_epsilon(10e-12, 1000)
    # X1 = {0}
    # X2 = set([])
    # xremain={0}
    # for elem in range(1, custom_len([k.shape for k in weights])):
    #     X2.add(elem)
    #     xremain.add(elem)
    # prueba = [np.random.rand(3, 4) * 2 - 1, np.random.rand(3, 1) * 2 - 1]
    # prueba_ravel = ravel(prueba)
    # shapes = [(3, 4), (1, 3)]
    # original = unravel(prueba_ravel, shapes)
    # print(prueba)
    # print(original)
    # print(epsilon)
    # LOWER_BOUND = 1
    # UPPER_BOUND = 5
    # p1 = LOWER_BOUND*np.ones(len(xremain))
    # p1_temp = unravel(p1,shapes)
    # y1 = f(data_X,data_Y,p1_temp)
    # xremain=set([])
    # sub1_a,xremain = interact(X1,X2,p1,y1,epsilon,UPPER_BOUND,LOWER_BOUND,shapes,xremain)
    #
    # sep, nonsep = RGD()
    # print(sep)
    # print(nonsep)
    # cosa = 0
