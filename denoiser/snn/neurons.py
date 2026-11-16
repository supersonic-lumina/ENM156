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
# which is most of the cost for small layers. The loops stay in Python.

class _LeakyRun(torch.autograd.Function):
    """v_t = decay * v_(t-1) + current_t, from v = 0; output v_t."""

    @staticmethod
    def forward(ctx, currents, decay):
        out = torch.empty_like(currents)
        v = torch.zeros_like(currents[:, 0])
        for t in range(currents.shape[1]):
            v = torch.addcmul(currents[:, t], decay, v)
            out[:, t] = v
        ctx.save_for_backward(out, decay)
        return out

    @staticmethod
    def backward(ctx, grad):
        out, decay = ctx.saved_tensors
        g_current = torch.empty_like(grad)
        g = torch.zeros_like(grad[:, 0])
        for t in reversed(range(grad.shape[1])):
            g = torch.addcmul(grad[:, t], decay, g)  # through v_t, now and in all later steps
            g_current[:, t] = g
        g_decay = None
        if ctx.needs_input_grad[1]:
            g_decay = (g_current[:, 1:] * out[:, :-1]).sum((0, 1))
        return g_current, g_decay


class _LIFRun(torch.autograd.Function):
    """LIF.step() over every time step from rest, optionally with recurrent
    weights (the spikes of the step before as extra current) -> spikes."""

    @staticmethod
    def forward(ctx, currents, decay, weight, threshold, slope):
        before = torch.empty_like(currents)  # membrane potentials before the reset
        spikes = torch.empty_like(currents)
        v = torch.zeros_like(currents[:, 0])
        s = None
        for t in range(currents.shape[1]):
            current = currents[:, t]
            if weight is not None and t:
                current = torch.addmm(current, s, weight.t())
            v = torch.addcmul(current, decay, v)
            s = (v > threshold).to(v.dtype)
            before[:, t], spikes[:, t] = v, s
            v = torch.sub(v, s, alpha=threshold)
        ctx.save_for_backward(before, spikes, decay, weight)
        ctx.threshold, ctx.slope = threshold, slope
        return spikes

    @staticmethod
    def backward(ctx, grad):
        before, spikes, decay, weight = ctx.saved_tensors
        threshold = ctx.threshold
        surrogate = (before - threshold).abs_().mul_(ctx.slope).add_(1).square_().reciprocal_()
        g_before = torch.empty_like(grad)
        g_next = None  # gradient of the potential before the reset, one step later
        for t in reversed(range(grad.shape[1])):
            if g_next is None:
                g = grad[:, t] * surrogate[:, t]  # the last potential only matters through its spike
            else:
                g_after = decay * g_next  # potential after the reset, carried into the next step
                g_spike = grad[:, t] if weight is None else torch.addmm(grad[:, t], g_next, weight)
                g = torch.addcmul(g_after, surrogate[:, t], g_spike - threshold * g_after)
            g_before[:, t] = g
            g_next = g
        g_decay = g_weight = None
        if ctx.needs_input_grad[1]:
            after = before[:, :-1] - threshold * spikes[:, :-1]
            g_decay = (g_before[:, 1:] * after).sum((0, 1))
        if weight is not None and ctx.needs_input_grad[2]:
            n = grad.shape[-1]
            g_weight = g_before[:, 1:].reshape(-1, n).t() @ spikes[:, :-1].reshape(-1, n)
        return g_before, g_decay, g_weight, None, None


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
