""" Full assembly of the parts to form the complete network """
import numpy as np
from .unet_parts import *
from base.models.ncv_model import NcvModel
from base.logwriter import Logging as logging

__all__ = ["UNet"]


class UNet(NcvModel):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        logging.info(f'Network:\n'
                     f'\t{self.n_channels} input channels\n'
                     f'\t{self.n_classes} output channels (classes)\n'
                     f'\t{"Bilinear" if self.bilinear else "Dilated conv"} upscaling')

        self.downs = [Down()] * 4
        self.ups = [Up()] * 4
        self.concats = [Concat()] * 4

        base = 64  # channel num base
        self.inc = DoubleConv(n_channels, base)

        filters = base * np.array([1, 2, 4, 8, 16])
        in_filters, out_filters = filters[:-1], filters[1:]
        self.convs1 = [DoubleConv(in_num, out_num)
                       for in_num, out_num in zip(in_filters, out_filters)]

        filters = filters[::-1]
        in_filters, out_filters = filters[:-1], filters[1:]
        self.convs2 = [DoubleConv(in_num, out_num)
                       for in_num, out_num in zip(in_filters, out_filters)]
        self.outc = OutConv(base, n_classes)

    def forward(self, x):
        x = x1 = self.inc(x)

        x_downs = []
        for down, conv1 in zip(self.downs, self.convs1):
            x = down(x)
            x = conv1(x)
            x_downs.append(x)

        x_ups = []
        for up_op, x1, concat in zip(self.ups, x_downs[::-1], self.concats):
            x2 = up_op(x)
            x = concat(x1, x2)
            x_ups.append(x)

        logits = self.outc(x)
        return logits
