import torch.nn as nn
import torch
import numpy as np

# Nonlinearity
def phi(h):
    return torch.erf(torch.sqrt(torch.tensor(np.pi)) * h / 2)

class RNNCell(nn.Module):
    def __init__(self, N, N_in, dt=0.05, W_scale=None,
                 W_in_scale=None, trainable_input=False):
        super(RNNCell, self).__init__()
        self.N = N
        self.N_in = N_in
        self.trainable_input = trainable_input
        self.dt = dt

        # Initialize recurrent weights (N x N)
        if W_scale is None:
            scale = 1.0 / np.sqrt(N)
        else:
            scale = W_scale
        self.W = nn.Parameter(torch.randn(N, N) * scale)

        # Initialize input weights (N x N_in)
        if W_in_scale is None:
            scale_factor = 1.0 / np.sqrt(N_in)
        else:
            scale_factor = W_in_scale
        input_weights = torch.randn(N, N_in) * scale_factor

        # Set input weights as either a trainable parameter or a fixed tensor
        if trainable_input:
            self.W_in = nn.Parameter(input_weights)
        else:
            self.W_in = nn.Parameter(input_weights, requires_grad=False)

    def forward(self, h, x):
        # h: (batch_size, N)
        # x: (batch_size, N_in)
        # W: (N, N)
        # W_in: (N, N_in)

        # Calculate the dot product term
        h_dot = -h + torch.einsum('ij,bj->bi', self.W, phi(h)) + torch.einsum('ik,bk->bi', self.W_in, x)

        # Return the updated hidden state using Euler discretization
        return h + self.dt * h_dot

class RNNModel(nn.Module):
    def __init__(self, N, N_in, init_scale=0, output_bias=False, dt=0.05,
                 trainable_input=False, W_scale=None,
                 W_in_scale=None, gp_samples=None,
                 pulse_duration=None):
        super(RNNModel, self).__init__()
        self.rnn_cell = RNNCell(N, N_in, dt=dt,
                                trainable_input=trainable_input,
                                W_scale=W_scale,
                                W_in_scale=W_in_scale)
        self.W_out = nn.Linear(N, 1, bias=output_bias)
        self.init_scale = init_scale
        self.hidden_states = []  # To store hidden states
        self.N = self.rnn_cell.N
        self.N_in = self.rnn_cell.N_in
        self.gp_samples = gp_samples
        self.pulse_duration = pulse_duration

    def forward(self, x, time_steps, prev_h=None):
        batch_size = x.shape[0]
        if prev_h is None:
            h = torch.randn(batch_size, self.N, device=0) * self.init_scale  # Initialize hidden state with batch size
        else:
            h = prev_h
        self.hidden_states = []  # Reset hidden states storage

        outputs = []
        for t in range(time_steps):
            if self.pulse_duration is not None:
                if t > self.pulse_duration:
                    h = self.rnn_cell(h, torch.zeros_like(x))
                else:
                    h = self.rnn_cell(h, x)
            else:
                h = self.rnn_cell(h, x)  # Process the batch
            self.hidden_states.append(h)
            outputs.append(self.W_out(h).unsqueeze(1))  # Shape (batch_size, 1) -> (batch_size, 1, 1)

        outputs = torch.cat(outputs, dim=1)  # Concatenate along the time dimension -> (batch_size, time_steps, 1)
        return outputs

    def get_hidden_states(self, dt_save=1):
        # Concatenate the hidden states across the time dimension
        # Shape: (time_steps * batch_size, N)
        H = self.hidden_states[::int(dt_save / self.rnn_cell.dt)]
        return torch.stack(H)