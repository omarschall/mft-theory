import numpy as np
import torch

def sample_gp(C, N_samples, ft_cutoff=None):
    N_t = len(C)
    if N_t % 2 != 0:
        raise ValueError("Length of C must be even")

    device = C.device
    C_ft = torch.fft.rfft(C, norm='forward').real
    C_ft_sqrt = torch.sqrt(torch.clamp(C_ft, min=0))

    if ft_cutoff is not None:
        C_ft_sqrt[ft_cutoff:] = 0.0

    out = torch.zeros(N_samples, N_t // 2 + 1, device=device) * 1j
    out[:, 0] = torch.randn(N_samples, device=device)
    out[:, 1:] = (
        torch.randn(N_samples, N_t // 2, device=device) +
        1j * torch.randn(N_samples, N_t // 2, device=device)
    ) / np.sqrt(2.)
    out *= C_ft_sqrt[None, :]

    eta = torch.fft.irfft(out, norm='forward', dim=1)
    return eta

def create_training_data(GP_samples, global_sample_size, random=True):
    """
    Converts GP samples into training data for RNN.

    Parameters:
    - GP_samples: numpy array of shape (N_in, time_steps)
    - global_sample_size: the desired size of the global sample (number of duplicated GP samples)

    Returns:
    - training_data: list of tuples (GP_sample, one_hot_input)
    """
    N_in, time_steps = GP_samples.shape
    device = GP_samples.device

    # Initialize an empty list to store the training data
    training_data = []

    for i in range(global_sample_size):
        # Randomly select a GP sample index
        if random:
            selected_idx = np.random.randint(0, N_in)
        else:
            selected_idx = i

        # Get the corresponding GP sample and one-hot vector
        GP_sample = GP_samples[selected_idx, :]  # Shape: (time_steps,)
        one_hot_input = np.zeros(N_in)
        one_hot_input[selected_idx] = 1  # Create the one-hot encoded vector

        # Convert to PyTorch tensors
        GP_sample_tensor = GP_sample.clone().view(time_steps, 1).to(device)  # Shape: (time_steps, 1)
        one_hot_input_tensor = torch.from_numpy(one_hot_input).type(torch.float32).to(device)  # Shape: (N_in,)

        # Append to the training data list
        training_data.append((GP_sample_tensor, one_hot_input_tensor))

    return training_data

def extract_zero_crossing_samples(gp_samples, T_short=20, dt=0.05):
    # Get the shape of the input
    N_samples, T_long = gp_samples.shape

    # Find the first zero crossing for each sample
    crossings = (gp_samples[:, :-1] * gp_samples[:, 1:] < 0).int()
    zero_crossings = torch.argmax(crossings, dim=1)

    # Initialize tensor to store the short samples
    short_samples = torch.zeros((N_samples, int(T_short/dt)),
                                dtype=gp_samples.dtype, device=gp_samples.device)

    for i in range(N_samples):
        start_idx = zero_crossings[i].item()
        end_idx = start_idx + int(T_short/dt)

        if end_idx <= T_long:
            short_samples[i] = gp_samples[i, start_idx:end_idx]
        else:
            # If not enough points after zero crossing, fill the rest with NaN or zero
            short_samples[i, :(T_long - start_idx)] = gp_samples[i, start_idx:T_long]

    return short_samples