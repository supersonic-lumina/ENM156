"""
Checks the spiking neurons on hand-made inputs: leak, threshold, reset, the
stand-in gradient of a spike, that a learnable decay gets a gradient, and
that run() gives what step() in a loop gives, gradients included.

  .venv/bin/python tests/test_snn_neurons.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from denoiser.snn.neurons import LIF, Leaky, spike  # noqa: E402


def run(neuron, currents):
    v, out = torch.zeros(1), []
    for c in currents:
        y, v = neuron.step(torch.tensor([c]), v)
        out.append(round(y.item(), 4))
    return out


def main():
    checks = {}

    # decay 0.5, input 1 then nothing: the potential halves every step
    checks["leak"] = run(Leaky(1, 0.5), [1, 0, 0]) == [1.0, 0.5, 0.25]

    # decay 0.5, threshold 1, constant input 0.6: v = 0.6, 0.9, 1.05 -> spike, back to 0.05, ...
    checks["threshold and reset"] = run(LIF(1, 0.5), [0.6] * 5) == [0, 0, 1, 0, 0]

    # a big input leaves the rest above the threshold on the membrane: 2.5 -> spike, 1.5 stays
    lif = LIF(1, 0.5)
    _, v = lif.step(torch.tensor([2.5]), torch.zeros(1))
    checks["reset by subtraction"] = abs(v.item() - 1.5) < 1e-6

    # spikes are exactly 0 or 1; the gradient is 1 at the threshold and smaller away from it
    x = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
    s = spike(x, 25.0)
    s.sum().backward()
    checks["spike values"] = s.tolist() == [0.0, 0.0, 1.0]
    checks["surrogate gradient"] = (abs(x.grad[1].item() - 1) < 1e-6
                                    and 0 < x.grad[0].item() < 0.01 and x.grad[0] == x.grad[2])

    # the decay is trained through time
    lif = LIF(4, 0.9)
    v, total = torch.zeros(4), 0
    for _ in range(5):
        s, v = lif.step(torch.full((4,), 0.4), v)
        total = total + s.sum()
    total.backward()
    checks["decay gets a gradient"] = lif.raw_decay.grad.abs().sum().item() > 0
    checks["fixed decay"] = not LIF(4, 0.9, learn_decay=False).raw_decay.requires_grad

    # run() (hand-written backward pass) gives what step() in a loop gives, gradients included;
    # also on the GPU when there is one (float32 there, so gradients to float32 rounding)
    devices = [("cpu", torch.float64)] + ([("cuda", torch.float32)] if torch.cuda.is_available() else [])
    for device, dtype in devices:
        torch.manual_seed(0)
        for name, neuron, feedback in (("leaky", Leaky(16, 0.7), None), ("LIF", LIF(16, 0.8, slope=5.0), None),
                                       ("recurrent LIF", LIF(16, 0.8, slope=5.0),
                                        torch.nn.Linear(16, 16, bias=False))):
            neuron = neuron.to(device, dtype)
            currents = torch.randn(3, 40, 16, dtype=dtype).to(device)
            weights = torch.randn(3, 40, 16, dtype=dtype).to(device)
            params = [neuron.raw_decay] + ([feedback.to(device, dtype).weight] if feedback is not None else [])
            results = []
            for how in (neuron.run, neuron.run_steps):
                c = currents.clone().requires_grad_()
                out = how(c, feedback)
                results.append((out.detach(), torch.autograd.grad((out * weights).sum(), [c, *params])))
            (out_run, g_run), (out_steps, g_steps) = results
            tol = 1e-8 if dtype == torch.float64 else 1e-5
            checks[f"{name} on {device}: run() = step() loop"] = (
                torch.equal(out_run, out_steps)
                and all(torch.allclose(a, b, rtol=tol, atol=tol) for a, b in zip(g_run, g_steps)))

    for name, ok in checks.items():
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
