import os
from typing import List
from typing import Optional
import hydra
import optuna
import torch.utils.data as data_utils
from optuna.integration import PyTorchLightningPruningCallback
from packaging import version
import pytorch_lightning as pl
# optuna imports
from optuna.trial import TrialState
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

SEED = 42
# Torch imports
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.utils.data as data_utilis
from torchvision import datasets
from torchvision import transforms
from main import TestFunction, NeuralNet

testF = TestFunction()
N_samples: int = 10000
dim: int = 4
LB: int = 1
UB: int = 10
DEVICE = torch.device("cuda")
BATCHSIZE = 128
BATCH_SIZE = 128
CLASSES = 1
DIR = os.getcwd()
EPOCHS = 10
LOG_INTERVAL = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10


# def dfine_model(n_layers)
def define_model_trial(trial: optuna.trial.Trial) -> nn.Module:
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 5)
    layers = []
    in_features = dim
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        # p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
        # layers.append(nn.Dropout(p))

        in_features = out_features
    layers.append(nn.Linear(in_features, CLASSES))
    return nn.Sequential(*layers)


def get_train_val_data(function: TestFunction, noisy: bool = True):
    if noisy:
        features, targets = function.generate_noisy_data(N_samples, [LB, UB])
        features = torch.tensor(features, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)
        train = data_utils.TensorDataset(features, targets)
        train_loader = data_utils.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
        # Generate the test data
        test_features, test_targets = function.generate_noisy_data(N_samples, [LB, UB])
        test_features = torch.tensor(test_features, dtype=torch.float32)
        test_targets = torch.tensor(test_targets, dtype=torch.float32)
        test = data_utils.TensorDataset(test_features, test_targets)
        test_loader = data_utils.DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)
        return train_loader, test_loader
    else:
        features, targets = function.generate_data(N_samples, [LB, UB])
        features = torch.tensor(features, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)
        train = data_utils.TensorDataset(features, targets)
        train_loader = data_utils.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
        # Generate the test data
        test_features, test_targets = function.generate_data(N_samples, [LB, UB])
        test_features = torch.tensor(test_features, dtype=torch.float32)
        test_targets = torch.tensor(test_targets, dtype=torch.float32)
        test = data_utils.TensorDataset(test_features, test_targets)
        test_loader = data_utils.DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)
        return train_loader, test_loader


class LightningNet(pl.LightningModule):
    def __init__(self, dropout: float, output_dims: List[int]):
        super().__init__()
        self.model = Net(dropout, output_dims)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.model(data.view(-1, 28 * 28))

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        data, target = batch
        output = self(data)
        return F.nll_loss(output, target)

    def validation_step(self, batch, batch_idx: int) -> None:
        data, target = batch
        output = self(data)
        pred = output.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(target.view_as(pred)).float().mean()
        self.log("val_acc", accuracy)
        self.log("hp_metric", accuracy, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.model.parameters())


def objective(trial: optuna.trial.Trial):
    # Generate the model.
    model = define_model_trial(trial).to(DEVICE)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])

    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the FashionMNIST dataset.

    train_loader, valid_loader = get_train_val_data(testF,noisy=False)

    # Training of the model.
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break

            data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = F.mse_loss(output.view_as(target), target)
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        epoch_val_loss = 0
        counter = 1
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                # Limiting validation data.
                if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                    break
                data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                pred = model(data)
                # Get the index of the max log-probability.
                epoch_val_loss += F.mse_loss(input=target.view_as(pred), target=pred, reduction="sum")
                counter += BATCHSIZE
        epoch_val_loss = epoch_val_loss / counter

        trial.report(epoch_val_loss, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return epoch_val_loss


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials= 100, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    print("Here begins the plotting fiesta")
    # Here begins the plotting fiesta
    fig1 = plot_optimization_history(study)
    fig2 = plot_intermediate_values(study)
    fig3 = plot_param_importances(study)
    fig1.show()
    fig2.show()
    fig3.show()
