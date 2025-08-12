import torch
import torch.nn as nn
from torch import Tensor
from functools import partial
from abc import abstractmethod
from torchdiffeq import odeint

from scflow.helpers import instantiate_from_config


_RTOL = 1e-5
_ATOL = 1e-5


def pad_vector_like_x(v, x):
    """
    Function to reshape the vector by the number of dimensions
    of x. E.g. x (bs, c, h, w), v (bs) -> v (bs, 1, 1, 1).
    """
    if isinstance(v, float):
        return v
    return v.reshape(-1, *([1] * (x.dim() - 1)))


class AbstractFlowMatching(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, t: Tensor, **kwargs):
        if t.numel() == 1:
            t = t.expand(x.size(0))
        _pred = self.net(x=x, t=t, **kwargs)
        return _pred

    def ode_fn(self, t, x, **kwargs):
        return self(x=x, t=t, **kwargs)
    
    def generate(self, x: Tensor, ode_kwargs=None, reverse=False, n_intermediates=0, **kwargs):
        """
        Args:
            x: Tensor, shape (bs, *dim), represents the source minibatch
            ode_kwargs: dict, additional arguments for the ode solver.
            reverse: bool, whether to reverse the direction of the flow. If
                True, we map from x1 -> x0 (target -> source), otherwise
                we map from x0 -> x1 (source -> target).
            n_intermediates: int, number of intermediate points to return.
            kwargs: additional arguments for the network (e.g. conditioning information).
        """
        # we use fixed step size for odeint to avoid numerical underflow and use 40 nfes (1/40 step size)
        default_ode_kwargs = dict(method="euler", rtol=_RTOL, atol=_ATOL, options=dict(step_size=1./40))
        # allow overriding default ode_kwargs
        default_ode_kwargs.update(ode_kwargs or dict())
        #print(default_ode_kwargs)
        # t specifies which intermediate times should the solver return
        # e.g. t = [0, 0.5, 1] means return the solution at t=0, t=0.5 and t=1
        # but it also specifies the number of steps for fixed step size methods
        t = torch.linspace(0, 1, n_intermediates + 2, device=x.device, dtype=x.dtype)
        t = 1 - t if reverse else t

        # allow conditioning information for model
        ode_fn = partial(self.ode_fn, **kwargs)

        ode_results = odeint(ode_fn, x, t, **default_ode_kwargs)

        if n_intermediates > 0:
            return ode_results
        return ode_results[-1]

    def encode(self, x: Tensor, ode_kwargs=None, n_intermediates=0, **kwargs):
        """ x1 -> x0 (target -> source) """
        return self.generate(x=x, ode_kwargs=ode_kwargs, reverse=True, n_intermediates=n_intermediates, **kwargs)

    def decode(self, z: Tensor, ode_kwargs=None, n_intermediates=0, **kwargs):
        """ x0 -> x1 (source -> target) """
        return self.generate(x=z, ode_kwargs=ode_kwargs, reverse=False, n_intermediates=n_intermediates, **kwargs)

    def encode_to_t(self, z: Tensor, t, **kwargs):
        # Default to euler method with fixed step size
        """ x1 -> x0 (target -> source) """
        ts = torch.linspace(1, 0, 51)[:-1].to(z.device)
        t = ts[torch.argmin((ts - t).abs())]
        ts = ts[ts > t]
        for t in ts:
            # t = t.expand(z.size(0))
            gradient = self(x=z, t=t, **kwargs)
            z = z - gradient * (1/50)
        return z
    
    def decode_from_t(self, z: Tensor, t, **kwargs):
        # Default to euler method with fixed step size
        """ x0 -> x1 (source -> target) """
        ts = torch.linspace(0, 1, 51)[:-1].to(z.device)
        # find the nearest index to t
        t = ts[torch.argmin((ts - t).abs())]
        ts = ts[ts >= t]
        for t in ts:
            # t = t.expand(z.size(0))
            gradient = self(x=z, t=t, **kwargs)
            z = z + gradient * (1/50)
        return z

    @abstractmethod
    def training_losses(self, x_target: Tensor, x_source: Tensor = None, **cond_kwargs):
        """
        x_target: target mini-batch (sampled from data distribution)
        x_source: source mini-batch (can be None for Flow Matching)
        cond_kwargs: additional arguments for the conditional flow network (e.g. context)
        """
        raise NotImplementedError("Abstract method")

    @abstractmethod
    def sample_xt(self, *args, **kwargs):
        raise NotImplementedError("Abstract method")

    @abstractmethod
    def compute_conditional_flow(self, *args, **kwargs):
        raise NotImplementedError("Abstract method")


class FlowMatching(AbstractFlowMatching):
    def __init__(self, net_cfg, sigma_min: float = 0.02):
        """
        Args:
            net_cfg: Config for a neural network that takes in x and t and
                outputs the vector field at that point in time and space
                with the same shape as x.
            sigma_min: a float representing the standard deviation of the
                Gaussian distribution around the mean of the probability
                path N(t * x1 + (1 - t) * x0, sigma_min). This is linearly
                interpolated for Flow Matching (FM).

        References:
            [2] Lipman et al. (2023). Flow Matching for Generative Modeling.
        """
        super().__init__()
        self.net = instantiate_from_config(net_cfg)
        self.sigma_min = sigma_min

    def sample_xt(self, x: Tensor, eps: Tensor, t):
        """
        Sample from N(t * x1, (1 - (1 - sigma_min) * t)^2),
        see (Eq. 22) [2].

        Args:
            x : Tensor, shape (bs, *dim), represents the target minibatch
            eps: Tensor, shape (bs, *dim), represents the noise
            t : FloatTensor, shape (bs,) represents the time, with values
        """
        t_ = pad_vector_like_x(t, x)
        xt = t_ * x + (1 - (1 - self.sigma_min) * t_) * eps

        return xt

    def compute_conditional_flow(self, x0: Tensor, x1: Tensor):
        """
        Compute the conditional vector field ut(x1|x0) = x1 - (1 - sigma_min) * x0,
        see Eq. (23) [2].

        Args:
            x0 : Tensor, shape (bs, *dim), represents the source minibatch (noise)
            x1 : Tensor, shape (bs, *dim), represents the target minibatch
        Returns:
            ut : conditional vector field ut(x1|x0) = x1 - (1 - sigma_min) * x0
        """
        return x1 - (1 - self.sigma_min) * x0

    def training_losses(self, x_target: Tensor, x_source: Tensor = None, **cond_kwargs):
        """ x_source = x0, x_target = x1 """
        if x_source is None:
            x_source = torch.randn_like(x_target)

        bs, dev, dtype = x_target.shape[0], x_target.device, x_target.dtype

        # Sample time t from uniform distribution U(0, 1)
        t = torch.rand(bs, device=dev, dtype=dtype)

        # sample xt and ut
        xt = self.sample_xt(x=x_target, eps=x_source, t=t)
        ut = self.compute_conditional_flow(x0=x_source, x1=x_target)
        vt = self.forward(x=xt, t=t, **cond_kwargs)

        # *range(1, ut.ndim) means take the mean over all dimensions except the batch dimension
        # it is done to be compatible with other losses which may take mean over B later
        return (vt - ut).square().mean(dim=[*range(1, ut.ndim)])
