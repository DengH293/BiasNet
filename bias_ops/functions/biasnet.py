import torch
from torch.autograd import Function
import biasops._C  # Import the compiled CUDA extension module


class BiasNetFunction(Function):
    @staticmethod
    def forward(ctx, f, p, rulebook, W1, b1, W2, b2):
        """
        Forward pass function.

        Args:
            f (Tensor): Input feature tensor, shape [N, C]
            p (Tensor): Position tensor, shape [N, 3]
            rulebook (Tensor): Rulebook tensor, shape [M, 2]
            W1 (Tensor): Weights of the first linear layer, shape [32, 3]
            W2 (Tensor): Weights of the second linear layer, shape [C, 32]

        Returns:
            Tuple[Tensor, Tensor]: (output, max_indices)
        """
        # Check if inputs are on CUDA
        if not f.is_cuda:
            raise RuntimeError("f must be a CUDA tensor")
        if not p.is_cuda:
            raise RuntimeError("p must be a CUDA tensor")
        if not rulebook.is_cuda:
            raise RuntimeError("rulebook must be a CUDA tensor")
        if not W1.is_cuda:
            raise RuntimeError("W1 must be a CUDA tensor")
        if not W2.is_cuda:
            raise RuntimeError("W2 must be a CUDA tensor")


        output, max_indices = biasops._C.bias_max_forward(f, p, rulebook, W1, b1, W2, b2)

        ctx.save_for_backward(f, p, rulebook, max_indices, W1, b1, W2, b2)
        return output, max_indices

    @staticmethod
    def backward(ctx, grad_output, grad_max_indices):
        """
        Backward pass function.

        Args:
            grad_output (Tensor): Upstream gradient, shape [num_output, C]
            grad_max_indices (Tensor): Gradient for max_indices, not needed

        Returns:
            Tuple[Tensor, None, None, Tensor, Tensor]:
                Gradients with respect to f, p, rulebook, W1, W2
        """
        # Retrieve saved variables
        f, p, rulebook, max_indices, W1, b1, W2, b2 = ctx.saved_tensors

        grad_f, grad_W1, grad_b1, grad_W2, grad_b2 = biasops._C.bias_max_backward(
            grad_output.contiguous(), max_indices, p, rulebook, W1, b1, W2, b2
        )


        return grad_f, None, None, grad_W1, grad_b1, grad_W2, grad_b2


class BiasNet(torch.nn.Module):
    def __init__(self, C, C_in=3):
        """
        Initialize the BiasNet module.

        Args:
            C (int): Number of output channels
        """
        super(BiasNet, self).__init__()
        if C <= 32 or C % 32 != 0:
            raise ValueError(f"Number of channels C={C} must be greater than 32 and a multiple of 32.")

        self.C = C
        

        # First linear layer (Linear(3, 32))
        self.W1 = torch.nn.Parameter(torch.empty(32, C_in, device='cuda'))
        torch.nn.init.kaiming_uniform_(self.W1, a=0, mode='fan_in', nonlinearity='relu')
        self.b1 = torch.nn.Parameter(torch.zeros(32, device='cuda'))

        # Second linear layer (Linear(32, C))
        self.W2 = torch.nn.Parameter(torch.empty(C, 32, device='cuda'))
        torch.nn.init.kaiming_uniform_(self.W2, a=0, mode='fan_in', nonlinearity='relu')
        self.b2 = torch.nn.Parameter(torch.zeros(C, device='cuda'))

    def forward(self, f, p, rulebook):
        """
        Forward pass function.

        Args:
            f (Tensor): Input feature tensor, shape [N, C]
            p (Tensor): Position tensor, shape [N, 3]
            rulebook (Tensor): Rulebook tensor, shape [M, 2]

        Returns:
            Tensor: Output feature tensor, shape [num_output, C]
        """
        output, _ = BiasNetFunction.apply(f, p, rulebook, self.W1, self.b1, self.W2, self.b2)
        return output
