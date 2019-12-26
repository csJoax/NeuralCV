import torch
import torch.nn as nn
from base.ncv_recorder import Logging as logging


class NcvModel(nn.Module):
    """
    NeuralCV model
    """

    def __init__(self, name, nn_module=None):
        super(NcvModel, self).__init__()
        self.name = name

    def save(self, name=None):
        if not name:
            name = self.name

        name = f'{name}.pth'
        torch.save(self.net.state_dict(), f'{name}.pth')
        logging.info(f'Checkpoint saved: {name}.')

    def load(self, path, device):
        self.load_state_dict(torch.load(path, map_location=device))
        logging.info(f'Model loaded from {path}.')
