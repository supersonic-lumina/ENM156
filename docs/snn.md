# How the spiking network works

This page explains the spiking network (SNN) in `denoiser/snn/` without
requiring you to read the code. After reading it you should be able to say
what every setting in the `[snn]` section of `config.toml` does. Terms in
*italics* are explained in the [glossary](#glossary) at the end.

## Contents

- [The big picture](#the-big-picture)
- [The LIF neuron](#the-lif-neuron)
- [Training through spikes: the surrogate gradient](#training-through-spikes-the-surrogate-gradient)
- [From frame to mask](#from-frame-to-mask)
- [Counting the cost](#counting-the-cost)
- [Common problems](#common-problems)
- [Every `[snn]` setting](#every-snn-setting)
- [Glossary](#glossary)
- [Further reading](#further-reading)

## The big picture

The denoiser works on short STFT frames. With the defaults (`stft.n_fft = 128`,
`stft.hop = 80` at 16 kHz) a new frame arrives every 5 ms. For each frame the
model gets the noisy spectrum and predicts something that turns it into clean
speech (see `model.output`), usually a *mask*.

The three model types (`model.type`) differ in how they remember the past:

| | Remembers the past by | Output of a neuron |
|---|---|---|
| `mlp` | not at all: every frame on its own | a real number |
| `tcn` | looking at a window of past frames (its receptive field) | a real number |
| `snn` | the state of its neurons, carried from frame to frame | a spike: 1 or nothing |

The SNN is therefore the same kind of frame-in, frame-out model as the others,
but with a different neuron. It looks at only one frame at a time (`context = 1`)
and keeps everything else it knows in the *membrane potentials* of its neurons.

Why bother? A spiking neuron only does work when it spikes. If few neurons
spike, few operations are needed, which is what makes SNNs interesting for
battery-powered devices such as hearing aids and neuromorphic chips (for
example Intel Loihi 2). The price is that SNNs are harder to train and usually
end up with somewhat lower sound quality than a conventional network of the
same size. That is why both quality (PESQ, STOI) and cost are measured.

The network (`LIFNet` in `denoiser/snn/nets.py`) looks like this:

```
noisy frame ──► Linear ──► LIF layer 1 ──spikes──► Linear ──► LIF layer 2 ──► ... ──► Linear ──► leaky readout ──► mask
               (current)                         (current)                                    (no spikes)
```

`snn.hidden = [256, 256, 256]` gives three spiking layers of 256 neurons each.

## The LIF neuron

Every hidden neuron is a *leaky integrate-and-fire* (LIF) neuron
(`LIF` in `denoiser/snn/neurons.py`). It has one number of state, its
membrane potential `v`. In every time step it does three things:

1. **Leak and integrate.** Keep a share of the old potential and add the input current:
   `v = decay · v + current`
2. **Fire.** If `v` is above the threshold, send a spike (output 1); otherwise output 0.
3. **Reset by subtraction.** A spike takes the threshold off the potential:
   `v = v − threshold`

Written as equations, with β for `decay`, θ for `threshold`, I for the input
current and S for the spike:

```
U[t] = β · (U[t−1] − θ · S[t−1]) + I[t]
S[t] = 1 if U[t] > θ, else 0
```

(This is the presentation's formula, except that here the reset happens right
after the spike, so the subtracted θ also leaks.)

What each part means in practice:

- **Decay** (`snn.decay`, default 0.9) is the share of the potential kept per
  time step. 0.9 means a neuron forgets about 10 % per step, so it remembers
  roughly the last 10 steps (50 ms with one step per frame). 0 means no memory at all.
- **Learned decay per neuron** (`snn.learn_decay = true`). Every neuron has its
  own decay, which starts at `snn.decay` and is trained along with the weights.
  Some neurons can learn to remember for a long time, others to react quickly.
  Internally the decay is stored as a raw number passed through a sigmoid, so it
  always stays between 0 and 1.
- **Threshold** (`snn.threshold`, default 1.0) is the potential at which the
  neuron spikes. Because the input weights are trained, the threshold mostly sets
  the scale; lowering it makes neurons spike more easily at the start.
- **Reset by subtraction** (rather than resetting to 0) keeps whatever was above
  the threshold. A strong input therefore gives more spikes over time, not just one.

All neurons start at rest (`v = 0`) at the start of every clip. Live, the state
is carried from frame to frame instead (`denoiser/stream.py`).

`snn.spiking = false` replaces the LIF neurons with *smooth* neurons (`Smooth`):
the same leaky potential, but the output is a value between 0 and 1 (a sigmoid
around the threshold) and nothing is reset. That is **not** a spiking network.
It is a control experiment that shows how much quality the spikes themselves cost.

## Training through spikes: the surrogate gradient

Training uses backpropagation: for every weight we need to know how the loss
changes if the weight changes a little. The problem is the spike. It is a step
function: 0 below the threshold, 1 above. Its derivative is 0 everywhere except
exactly at the threshold, where it is infinite. Backpropagation through it would
give no training signal at all.

The fix is a *surrogate gradient*. The forward pass is unchanged: the network
really sends 0s and 1s. In the backward pass the step's derivative is replaced
by a smooth bump around the threshold, the derivative of a "fast sigmoid":

```
d spike / d v  ≈  1 / (1 + slope · |v − threshold|)²
```

It is 1 at the threshold and falls off on both sides. `snn.surrogate_slope`
(default 25) is the `slope` in this formula:

- **Higher slope:** a narrow bump. Only neurons whose potential is close to the
  threshold get a training signal. Closer to the true step, but neurons far from
  the threshold, for example a neuron that never spikes, learn almost nothing.
- **Lower slope:** a wide bump. More neurons learn, but the gradient is a
  rougher approximation of what the spikes actually do.

Because the neurons remember, the gradient also has to flow backwards through
time: the loss at frame 100 depends on the potential at frame 99, which depends
on frame 98, and so on. This is *BPTT* (backpropagation through time). It is why
an SNN needs more memory to train than an MLP: every time step of the clip has
to be kept for the backward pass. `neurons.py` has hand-written forward and
backward passes over the whole sequence (`_LIFRun`, `_LeakyRun`) that give the
same result as stepping one time step at a time, but much faster.

Train the SNN with `train_pesq.py` (see the README):

```bash
.venv/bin/python train_pesq.py --set model.type=snn --set pesq.init_from= --set train.output_dir=runs/snn
```

## From frame to mask

### Input: a frame becomes current

The first layer is not fed by spikes. The noisy input frame (the same features
the other models get, see [`denoiser/pipeline.py`](denoiser/pipeline.py)) goes through an ordinary
linear layer, and its output is used directly as the input current of the first
LIF layer. This is called *direct coding*: there is no conversion of the audio
into spikes first.

Every later layer gets its current from the spikes of the layer below, through
another linear layer. Since spikes are 0 or 1, this only means adding up the
weights of the neurons that spiked, with no multiplications.

### `steps_per_frame`: time steps per frame

By default there is one time step per frame (`snn.steps_per_frame = 1`).
With a higher value every frame is run for that many time steps, all with the
same input current. The neurons then have several chances to spike per frame,
so information can be carried in finer spike timing and spike counts. The
output is read after the last step of each frame. It costs that many times
more operations (and training memory), and with more steps per frame each
step is a shorter time, so the decays mean less memory in milliseconds.

### `recurrent`: neurons that listen to their own layer

With `snn.recurrent = true` every spiking layer also gets the spikes of its own
layer from the time step before, through a separate weight matrix. This gives the
network a second kind of memory, besides the leaking potential. It costs extra
synops: every spike now also reaches every neuron in its own layer.

### Readout: spikes become a mask

The last layer does not spike. The spikes of the last LIF layer go through a
linear layer into a *leaky readout* layer (`Leaky`): neurons with a membrane
potential that leaks and integrates like a LIF neuron, but never fire or reset.
The potential itself is the output. A bias is added and then the output
activation for `model.output`:

| `model.output` | Output activation | What the readout predicts |
|---|---|---|
| `mask` | sigmoid | a gain between 0 and 1 per frequency bin |
| `complex_mask` | none | real and imaginary parts of a *complex mask*, which also fixes the phase |
| `magnitude` | ReLU | the clean magnitude directly |

`snn.readout_decay` (default 0.5) is the decay of the readout potential. Higher
means the output changes more slowly and is smoother over time; lower means it
follows the newest spikes more closely.

### Options that help deep networks train

- **`snn.skip`** adds each layer's input current to the next layer's as well
  (where the two layers have the same size). The training signal then has a
  shortcut past every layer, like a residual connection in the TCN.
- **`snn.norm`** applies batch normalisation to each layer's input current. Once
  training is done it is just a fixed scale and shift per neuron, which could be
  folded into the weights, so it adds no operations on hardware.
- **`snn.init_gain`** scales the starting weights of the layers fed by spikes.
  With the default (1) the deeper layers start out nearly silent; a larger value
  makes them spike from the start.

## Counting the cost

`evaluate.py` prints a cost line for every model, computed by
`denoiser/snn/stats.py`. Nothing is measured on hardware; everything is counted
from the layer sizes and, for an SNN, from the spikes it actually sent on the
test clips.

### MACs and synops

- A ***MAC*** (multiply-accumulate) is a weight times a value, added to a sum.
  A conventional layer does one MAC per weight per frame. For the MLP and TCN,
  the count is simply the number of weights in their linear and convolution
  layers.
- A ***synop*** (synaptic operation) is a spike arriving at a weight. Because the
  spike is 1, the weight is just added: no multiplication. Synops only happen
  when a neuron spikes, which is where the energy savings come from.

For the SNN, per frame:

- **MACs** = the first layer's input weights (the input is a real-valued frame,
  so those are real multiplications, done once per frame, pruned weights left
  out) + one multiplication per neuron per time step for its leak.
- **Synops** = for each spiking layer, its spikes × the number of weights one
  spike reaches (the size of the next layer, plus its own layer's size if
  `recurrent`), + with `skip`, one addition per neuron per time step for the
  current passed on.

Since the first layer is the only one that multiplies, most of the estimated
energy usually goes there. That is what `prune.py` attacks: it removes the least
important input weights of a trained SNN (`LIFNet.input_mask`) so it can be
fine-tuned without them.

### Spike rate

The *spike rate* is the number of spikes per neuron per time step, between 0
(silent) and 1 (spikes every step). `evaluate.py` prints it for each layer.
During training the network also computes the mean spike rate with gradient, and
`snn.rate_weight` adds it to the loss in `train_pesq.py`. A higher weight asks
for fewer spikes, which lowers the cost but can lower the quality.

### The energy estimate

The operation counts are turned into energy with fixed costs per operation:
4.6 pJ per MAC and 0.9 pJ per synop, the usual estimates for a 45 nm chip with
32-bit floats (Horowitz 2014). Multiplied by the frame rate it becomes power,
printed in µW.

What it **does** mean: a fair, simple way to compare networks with each other,
for example an SNN against an MLP, or two SNNs with different `rate_weight`.

What it **does not** mean:

- It is not a measurement of any real chip, and real neuromorphic hardware has
  different costs per operation.
- It leaves out moving data in and out of memory, which on real hardware often
  costs more than the arithmetic itself.
- It leaves out the STFT, the inverse STFT, normalisation, activations and other
  small parts around the network.

## Common problems

- **Silent neurons.** A layer that never spikes passes no signal forward and, with
  a steep surrogate, gets almost no gradient back. Look at the spike rates per
  layer. Try `snn.init_gain` above 1, `snn.skip`, `snn.norm`, a lower `threshold`,
  or a lower `surrogate_slope`.
- **Saturated neurons.** Neurons that spike in every step carry no information and
  save no energy. A small `snn.rate_weight` pushes the rate down.
- **Leftover state.** Membrane potentials from one clip must not leak into the
  next. The code starts every clip from rest; keep it that way if you change it.
- **Memory during training.** BPTT keeps every time step, so long crops,
  big layers or a high `steps_per_frame` quickly fill up the VRAM.

## Every `[snn]` setting

| Setting | Default | What it does |
|---|---|---|
| `hidden` | `[256, 256, 256]` | Number of neurons in each spiking layer; the list length is the number of layers. |
| `recurrent` | `false` | Each layer also gets its own spikes from the step before, through its own weights. More memory, more synops. |
| `decay` | `0.9` | Share of the membrane potential kept per time step, at the start of training. 0 = no memory. |
| `readout_decay` | `0.5` | The same for the readout layer. Higher = smoother output over time. |
| `learn_decay` | `true` | Every neuron (and readout neuron) learns its own decay. |
| `threshold` | `1.0` | Membrane potential at which a neuron spikes. |
| `surrogate_slope` | `25` | Steepness of the surrogate gradient. Higher = only neurons near the threshold learn. |
| `steps_per_frame` | `1` | Time steps per frame, all with the same input. More = finer spike timing at that many times the cost. |
| `skip` | `false` | Adds each layer's input current to the next layer's (same-size layers). Helps deep networks train. |
| `norm` | `false` | Batch normalisation of each layer's input current. Free once trained (fold into the weights). |
| `init_gain` | `1.0` | Scales the starting weights of the spike-fed layers. Above 1 = livelier deep layers at the start. |
| `spiking` | `true` | `false` = smooth neurons instead of spikes: a control, not an SNN. |
| `rate_weight` | `0.0` | `train_pesq.py` only: loss weight on the mean spike rate. Higher = fewer spikes. 0 = off. |

## Glossary

| Term | Meaning |
|---|---|
| **BPTT** | Backpropagation through time. The network is unrolled over all time steps and the gradient follows the membrane potentials backwards through them. |
| **Complex mask** | A complex number per frequency bin that is multiplied with the noisy spectrum. Its size changes the loudness of the bin, its angle shifts the phase, so it can fix both. (`model.output = "complex_mask"`) |
| **Direct coding** | Feeding the real-valued input straight into the first spiking layer as current, instead of turning it into spikes first. |
| **LIF neuron** | Leaky integrate-and-fire neuron: integrates input current into a leaking membrane potential and spikes when it passes the threshold. |
| **MAC** | Multiply-accumulate: one multiplication and one addition. The basic operation of a conventional network. |
| **Mask** | Per frequency bin, how much of the noisy signal to keep (0 to about 1). |
| **Membrane potential** | The internal state of a spiking neuron, carried between time steps. |
| **Readout** | The last layer, which does not spike: its membrane potential is the network's output. |
| **Spike rate** | Spikes per neuron per time step, between 0 and 1. A rough measure of activity and energy. |
| **Surrogate gradient** | A smooth function used instead of the spike's derivative in the backward pass, so the network can be trained with backpropagation. |
| **Synop** | Synaptic operation: one spike arriving at one weight, which is then added. No multiplication needed. |
| **Threshold** | The membrane potential at which a neuron spikes. |

## Further reading

- Neftci, Mostafa & Zenke (2019). *Surrogate Gradient Learning in Spiking Neural Networks.* IEEE Signal Processing Magazine.
- Eshraghian et al. (2023). *Training Spiking Neural Networks Using Lessons From Deep Learning.* Proceedings of the IEEE. (snnTorch)
- Timcheck et al. (2023). *The Intel Neuromorphic DNS Challenge.* Neuromorphic Computing and Engineering.
- Horowitz (2014). *Computing's Energy Problem (and what we can do about it).* ISSCC. (energy per operation)
