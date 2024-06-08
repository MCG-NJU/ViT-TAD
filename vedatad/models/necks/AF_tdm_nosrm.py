import torch.nn as nn

from vedacore.misc import build_from_module, registry


@registry.register_module('neck')
class AF_tdm_nosrm(nn.Module):
    """Spatial Reduction Module."""

    def __init__(self,num_layers,kernel_size,stride,padding):
        super(AF_tdm_nosrm, self).__init__()

        self.pool=[]
        self.num_layers=num_layers
        self.stride=stride
        self.kernel_size=kernel_size
        self.padding=padding

        for i in range(self.num_layers-1):
            if isinstance(self.stride, int):
                self.pool.append(nn.MaxPool1d(kernel_size=self.kernel_size,stride=self.stride, padding=self.padding))
            else:
                self.pool.append(nn.MaxPool3d(kernel_size=self.kernel_size,stride=self.stride, padding=self.padding))

    def init_weights(self):
        pass

    def forward(self, x): 
        x_result=[]
        temp=x
        x_result.append(temp)
        for i in range(self.num_layers-1):
            temp=self.pool[i](temp)
            x_result.append(temp)
        if isinstance(self.stride, int):
            return x_result
        else:
            x_result_avg=[]
            for i in range(len(x_result)):
                x_result_avg.append(self.srm(x_result[i]).squeeze(-1).squeeze(-1))
            return x_result_avg
