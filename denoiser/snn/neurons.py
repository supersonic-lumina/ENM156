"""
Spiking neurons: step() for one time step, run() for a whole sequence.

A leaky integrate-and-fire (LIF) neuron keeps a membrane potential v:
  v = decay * v + input current
  spike (output 1, else 0) when v is above the threshold
  a spike takes the threshold off v again (reset by subtraction)
The spike is a step, which has no slope to train through, so the backward
pass uses a stand-in (surrogate) slope: that of a fast sigmoid around the
threshold.
"""
import math

import torch
import torch.nn as nn


class _Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, slope):
        ctx.save_for_backward(x)
        ctx.slope = slope
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad / (1 + ctx.slope * x.abs()) ** 2, None


def spike(x, slope):
    """1 where x > 0, else 0. Its gradient is that of x / (1 + slope * |x|):
    1 at the threshold, falling off on both sides, faster for a higher slope."""
    return _Spike.apply(x, slope)


# run() over a whole sequence with a hand-written backward pass (backpropagation
# through time). The result and gradients are those of step() in a loop, but
# autograd records one node per sequence instead of several per time step,
# which is most of the cost for small layers. The loops stay in Python, so
# each time step costs a few GPU kernel launches; to keep that number low,
# every step writes straight into time-major (T, B, n) buffers (no copies)
# and the backward pass precomputes what doesn't depend on the step before.


class _LeakyRun(torch.autograd.Function):
    """v_t = decay * v_(t-1) + current_t, from v = 0; output v_t."""

    @staticmethod
    def forward(ctx, currents, decay):
        out = currents.new_empty(currents.shape[1], currents.shape[0], currents.shape[2])
        out[0] = currents[:, 0]
        for t in range(1, currents.shape[1]):
            torch.addcmul(currents[:, t], decay, out[t - 1], out=out[t])
        out = out.transpose(0, 1)
        ctx.save_for_backward(out, decay)
        return out

    @staticmethod
    def backward(ctx, grad):
        out, decay = ctx.saved_tensors
        n_steps = grad.shape[1]
        g = grad.new_empty(n_steps, grad.shape[0], grad.shape[2])
        g[-1] = grad[:, -1]
        for t in reversed(range(n_steps - 1)):  # through v_t, now and in all later steps
            torch.addcmul(grad[:, t], decay, g[t + 1], out=g[t])
        g_current = g.transpose(0, 1)
        g_decay = None
        if ctx.needs_input_grad[1]:
            g_decay = (g_current[:, 1:] * out[:, :-1]).sum((0, 1))
        return g_current, g_decay


class _LIFRun(torch.autograd.Function):
    """LIF.step() over every time step from rest, optionally with recurrent
    weights (the spikes of the step before as extra current) -> spikes."""

    @staticmethod
    def forward(ctx, currents, decay, weight, threshold, slope):
        n_steps = currents.shape[1]
        shape = (n_steps, currents.shape[0], currents.shape[2])
        before = currents.new_empty(shape)  # membrane potentials before the reset
        spikes = currents.new_empty(shape)
        v = None
        for t in range(n_steps):
            if weight is not None and t:
                torch.addmm(currents[:, t], spikes[t - 1], weight.t(), out=before[t])
                before[t].addcmul_(decay, v)
            elif t:
                torch.addcmul(currents[:, t], decay, v, out=before[t])
            else:
                before[0] = currents[:, 0]
            torch.gt(before[t], threshold, out=spikes[t])
            v = torch.sub(before[t], spikes[t], alpha=threshold)
        ctx.save_for_backward(before, spikes, decay, weight)
        ctx.threshold, ctx.slope = threshold, slope
        return spikes.transpose(0, 1)

    @staticmethod
    def backward(ctx, grad):
        # With s the surrogate slope at step t, the gradient of the potential
        # before the reset is
        #   g_t = s * (grad_t + g_(t+1) @ W) + decay * (1 - threshold * s) * g_(t+1)
        # (the first term through the spike, the second through the potential
        # kept after the reset). Everything but g_(t+1) is known up front.
        before, spikes, decay, weight = ctx.saved_tensors
        threshold = ctx.threshold
        grad = grad.transpose(0, 1)  # time-major, like the saved tensors
        surrogate = (before - threshold).abs_().mul_(ctx.slope).add_(1).square_().reciprocal_()
        through_spike = surrogate * grad
        kept = (1 - threshold * surrogate).mul_(decay)
        g = torch.empty_like(through_spike)
        g[-1] = through_spike[-1]  # the last potential only matters through its spike
        for t in reversed(range(grad.shape[0] - 1)):
            torch.addcmul(through_spike[t], kept[t], g[t + 1], out=g[t])
            if weight is not None:
                g[t].addcmul_(surrogate[t], g[t + 1] @ weight)
        g_decay = g_weight = None
        if ctx.needs_input_grad[1]:
            after = before[:-1] - threshold * spikes[:-1]
            g_decay = (g[1:] * after).sum((0, 1))
        if weight is not None and ctx.needs_input_grad[2]:
            n = grad.shape[-1]
            g_weight = g[1:].reshape(-1, n).t() @ spikes[:-1].reshape(-1, n)
        return g.transpose(0, 1), g_decay, g_weight, None, None


class Leaky(nn.Module):
    """Neurons that integrate their input and leak, without spiking: the
    membrane potential itself is the output. Used as the readout layer.
    Every neuron has its own decay per step (0 = no memory, near 1 = long)."""

    def __init__(self, size, decay, learn_decay=True):
        super().__init__()
        decay = min(max(decay, 1e-4), 1 - 1e-4)
        raw = torch.full((size,), math.log(decay / (1 - decay)))  # sigmoid(raw) = decay
        self.raw_decay = nn.Parameter(raw, requires_grad=learn_decay)

    @property
    def decay(self):
        return torch.sigmoid(self.raw_decay)

    def step(self, current, v, decay=None):
        """One time step: (output, new membrane potential)."""
        v = torch.addcmul(current, self.decay if decay is None else decay, v)
        return v, v

    def run(self, currents, feedback=None):
        """Every time step of currents (B, T, size), from rest -> outputs
        (B, T, size). feedback: a layer that turns the outputs of one step
        into extra current for the next (recurrent connections). The same as
        run_steps(), but faster."""
        if feedback is not None:
            return self.run_steps(currents, feedback)
        return _LeakyRun.apply(currents, self.decay)

    def run_steps(self, currents, feedback=None):
        """run() as a loop over step(), with autograd through every step."""
        decay = self.decay  # once, not in every step
        v = torch.zeros_like(currents[:, 0])
        y, out = v, []
        for current in currents.unbind(1):
            if feedback is not None:
                current = current + feedback(y)
            y, v = self.step(current, v, decay)
            out.append(y)
        return torch.stack(out, 1)


class LIF(Leaky):
    """Leaky integrate-and-fire neurons: output 1 (a spike) or 0."""

    def __init__(self, size, decay, threshold=1.0, slope=25.0, learn_decay=True):
        super().__init__(size, decay, learn_decay)
        self.threshold, self.slope = threshold, slope

    def step(self, current, v, decay=None):
        v = torch.addcmul(current, self.decay if decay is None else decay, v)
        s = spike(v - self.threshold, self.slope)
        return s, torch.sub(v, s, alpha=self.threshold)

    def run(self, currents, feedback=None):
        """feedback: None or a Linear without bias (otherwise run_steps())."""
        if feedback is not None and not (isinstance(feedback, nn.Linear) and feedback.bias is None):
            return self.run_steps(currents, feedback)
        weight = None if feedback is None else feedback.weight
        return _LIFRun.apply(currents, self.decay, weight, self.threshold, self.slope)


class Smooth(Leaky):
    """Not a spiking neuron: the control that shows what the spikes cost in
    quality. The same leaky membrane potential, but the output is a smooth
    value between 0 and 1 (a sigmoid around the threshold) and nothing is reset."""

    def __init__(self, size, decay, threshold=1.0, learn_decay=True):
        super().__init__(size, decay, learn_decay)
        self.threshold = threshold

    def step(self, current, v, decay=None):
        v = torch.addcmul(current, self.decay if decay is None else decay, v)
        return torch.sigmoid(4 * (v - self.threshold)), v

    def run(self, currents, feedback=None):
        return self.run_steps(currents, feedback)
