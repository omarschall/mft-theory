import matplotlib.pyplot as plt
import torch
import math

def visualize_rnn_performance(model, GP_sample, one_hot_input, time_vector):
    """
    Visualizes the RNN's performance on a given GP sample.

    Parameters:
    - model: Trained RNN model
    - GP_sample: Tensor of shape (time_steps, 1) representing the GP sample
    - one_hot_input: Tensor of shape (N_in,) representing the one-hot input vector
    - time_steps: Number of time steps in the GP sample
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # Run the model on the provided GP sample input
    with torch.no_grad():
        output = model(one_hot_input.unsqueeze(0), len(time_vector)).squeeze(0).cpu().detach().numpy().flatten()

    # Plot the GP sample vs RNN output
    plt.figure(figsize=(10, 5))
    plt.plot(time_vector, GP_sample.cpu().detach().numpy(), label='Original GP Sample')
    plt.plot(time_vector, output, label='RNN Output', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Output Value')
    plt.title('RNN Performance on GP Sample')
    plt.legend()
    plt.show()

def visualize_rnn_performance_multiple_samples(model, GP_samples, one_hot_inputs, time_vector,
                                               ylim=[-3.5,3.5], n_samples=1):
    """
    Visualizes the RNN's performance on a list of GP samples.

    Parameters:
    - model: Trained RNN model
    - GP_samples: List or Tensor of shape (num_samples, time_steps, 1) representing GP samples
    - one_hot_input: Tensor of shape (N_in,) representing the one-hot input vector
    - time_vector: Tensor of shape (time_steps,) representing the time steps
    """

    # Ensure the model is in evaluation mode
    model.eval()

    # Calculate grid size for subplots based on the number of samples
    num_samples = len(GP_samples)
    grid_size = math.ceil(math.sqrt(num_samples))  # Arrange as close to a square as possible

    # Create subplots
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axs = axs.flatten()  # Flatten to make indexing easier

    with torch.no_grad():
        # Run the model on the provided GP sample input
        # one_hot_input = torch.from_numpy(one_hot_input).type(torch.float32).to(GP_sample.device)
        all_outputs = []
        for i_sample in range(n_samples):
            outputs = model(one_hot_inputs, len(time_vector)).squeeze(0).cpu().detach().numpy()
            all_outputs.append(outputs)

    # Iterate over each GP sample and plot
    for i, GP_sample in enumerate(GP_samples):
        for i_sample in range(n_samples):
            outputs = all_outputs[i_sample]
            output = outputs[i]
            axs[i].plot(time_vector, output, color='C1', label='RNN Output {}'.format(i_sample))
        # Plot the GP sample vs RNN output
        gp = GP_sample.cpu().detach().numpy()
        axs[i].plot(time_vector[:len(gp)], gp, label='Original GP Sample')
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel('Output Value')
        axs[i].set_ylim(ylim)
    axs[0].legend()

    # Hide any unused subplots
    for i in range(num_samples, grid_size ** 2):
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()

def evaluate_rnn_performance(model, GP_samples, one_hot_inputs, time_vector):
    """
    Evaluates the RNN's performance on a list of GP samples.

    Parameters:
    - model: Trained RNN model
    - GP_samples: List or Tensor of shape (num_samples, time_steps, 1) representing GP samples
    - one_hot_input: Tensor of shape (N_in,) representing the one-hot input vector
    - time_vector: Tensor of shape (time_steps,) representing the time steps
    """
    # Ensure the model is in evaluation mode
    model.eval()

    with torch.no_grad():
        # Run the model on the provided GP sample input
        outputs = model(one_hot_inputs, len(time_vector)).squeeze(0).cpu().detach().numpy()

    # Calculate the mean squared error for each GP sample
    mse_values = []
    for i, GP_sample in enumerate(GP_samples):
        output = outputs[i]
        mse = ((output.squeeze() - GP_sample.cpu().detach().numpy()) ** 2).mean()
        mse_values.append(mse)

    return mse_values