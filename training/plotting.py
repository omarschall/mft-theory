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
                                               ylim=[-3.5, 3.5]):
    """
    Visualizes the RNN's performance on a list of GP samples.

    Parameters:
    - model: Trained RNN model
    - GP_samples: Tensor of shape (num_samples, time_steps, 1) representing GP samples
    - one_hot_inputs: Tensor of shape (num_samples, N_in) representing the one-hot input vectors
    - time_vector: Tensor of shape (time_steps,) representing the time steps
    - ylim: List specifying y-axis limits for the plots
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # Calculate grid size for subplots based on the number of samples
    num_samples = GP_samples.shape[0]
    grid_size = math.ceil(math.sqrt(num_samples))  # Arrange as close to a square as possible

    # Create subplots
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axs = axs.flatten()  # Flatten to make indexing easier

    with torch.no_grad():
        # Run the model on the batch of one_hot_inputs
        batch_inputs = one_hot_inputs.to(next(model.parameters()).device)
        outputs = model(batch_inputs, len(time_vector))  # Shape: (batch_size, time_steps, N_out)

        if model.N_out == 1:
            # If using a common output dimension
            outputs = outputs.squeeze(-1).cpu().numpy()  # Shape: (batch_size, time_steps)
        else:
            # Get active task indices from the one-hot inputs
            active_indices = one_hot_inputs.argmax(dim=1).to(outputs.device)  # Shape: (batch_size,)
            batch_size, time_steps, _ = outputs.shape
            # Prepare indices to gather the active outputs
            indices = active_indices.unsqueeze(1).unsqueeze(2).expand(-1, time_steps, 1)
            # Gather the outputs corresponding to the active tasks
            outputs = torch.gather(outputs, dim=2, index=indices).squeeze(-1).cpu().numpy()  # Shape: (batch_size, time_steps)

    # Iterate over each GP sample and plot
    for i in range(num_samples):
        axs[i].plot(time_vector, outputs[i], color='C1', label='RNN Output')
        # Plot the GP sample vs RNN output
        gp = GP_samples[i].cpu().numpy().flatten()
        axs[i].plot(time_vector[:len(gp)], gp, label='Original GP Sample')
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel('Output Value')
        axs[i].set_ylim(ylim)
        axs[i].set_title(f'Sample {i+1}')
        axs[i].legend()

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
    - GP_samples: Tensor of shape (num_samples, time_steps, 1) representing GP samples
    - one_hot_inputs: Tensor of shape (num_samples, N_in) representing the one-hot input vectors
    - time_vector: Tensor of shape (time_steps,) representing the time steps

    Returns:
    - mse_values: List of MSE values for each sample
    """
    # Ensure the model is in evaluation mode
    model.eval()

    with torch.no_grad():
        # Run the model on the batch of one_hot_inputs
        batch_inputs = one_hot_inputs.to(next(model.parameters()).device)
        outputs = model(batch_inputs, len(time_vector))  # Shape: (batch_size, time_steps, N_out)

        if model.N_out == 1:
            # If using a common output dimension
            outputs = outputs.squeeze(-1).cpu()  # Shape: (batch_size, time_steps)
        else:
            # Get active task indices from the one-hot inputs
            active_indices = one_hot_inputs.argmax(dim=1).to(outputs.device)  # Shape: (batch_size,)
            batch_size, time_steps, _ = outputs.shape
            # Prepare indices to gather the active outputs
            indices = active_indices.unsqueeze(1).unsqueeze(2).expand(-1, time_steps, 1)
            # Gather the outputs corresponding to the active tasks
            outputs = torch.gather(outputs, dim=2, index=indices).squeeze(-1).cpu()  # Shape: (batch_size, time_steps)

    # Calculate the mean squared error for each GP sample
    GP_samples = GP_samples.squeeze(-1).cpu()  # Shape: (batch_size, time_steps)
    mse_values = ((outputs - GP_samples) ** 2).mean(dim=1).numpy().tolist()

    return mse_values
