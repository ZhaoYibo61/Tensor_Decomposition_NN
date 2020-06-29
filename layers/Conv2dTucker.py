import tensorly
tensorly.set_backend(backend_name='pytorch')
from tensorly import decomposition
import torch
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair

class Conv2dTucker(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', decomp_rank=None):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2dTucker, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, _pair(0), groups, bias, padding_mode)

        self.layer_learnable_weights = torch.nn.ParameterDict()
        if decomp_rank is None:
            self.decomp_rank = kernel_size[0]
        else:
            self.decomp_rank = decomp_rank

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(self.decomp_rank))
        else:
            self.register_parameter('bias', None)

        self.generate_decomposition_tensors()
        self.weight.requires_grad = False


    def generate_decomposition_tensors(self):
        # rank_list is needed to decompose self.weight according to API
        rank_list = [self.decomp_rank] * len(self.weight.shape)

        # Handle MemoryError if decomp_rank is too large
        try:
            core, factors = tensorly.decomposition.tucker(self.weight, ranks=rank_list, init='svd')
        except MemoryError as memError:
            # Memory errors happens in cases such as Resnet's 512 sized layer
            core, factors = tensorly.decomposition.tucker(self.weight, ranks=rank_list, init='random')

        self.layer_learnable_weights['core'] = torch.nn.Parameter(core)
        self.layer_learnable_weights['output_channel_component'] = torch.nn.Parameter(factors[0])
        self.layer_learnable_weights['input_channel_component'] = torch.nn.Parameter(factors[1])


    def forward(self, input_tensor):
        # self.generate_decomposition_tensor() # Doing the decomp here makes runtime unbearably slow
        operation_1 = torch.tensordot(input_tensor, self.layer_learnable_weights['input_channel_component'], dims=([1],[0]))
        operation_2 = operation_1.permute(0, -1, 1, 2) # place the last dimension into the input channel location since the previous tensordot moved it
        operation_3 = torch.nn.functional.conv2d(operation_2, self.layer_learnable_weights['core'], self.bias, self.stride, self.padding, self.dilation, self.groups)
        operation_4 = torch.tensordot(operation_3, self.layer_learnable_weights['output_channel_component'], dims=([1],[1]))
        operation_5 = operation_4.permute(0, -1, 1, 2) # place the last dimension into the input channel location since the previous tensordot moved it
        return operation_5

















