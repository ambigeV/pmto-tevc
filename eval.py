import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from problems import get_problem
from models import ModelList, VanillaGP, ArdGP
import gpytorch


def uniform_sample(sample_width=100, task_num=2):
    # Prepare the np.linspace
    list_linspace = []
    list_ravel = []
    for i in range(task_num):
        list_linspace.append(np.linspace(0.05, 0.95, sample_width))

    # Grid creation
    grids = np.meshgrid(*list_linspace)

    # List of ravel grids
    for grid in grids:
        list_ravel.append(grid.ravel())

    points = np.vstack(list_ravel).T
    samples = torch.from_numpy(points).float()
    return samples


def load_saved_model(results_dict, dim_size, task_dim_size):
    """
    Load the saved model directly from the results dictionary
    """
    # Extract saved state dictionaries
    model_state = results_dict["model_tasks"]
    likelihood_state = results_dict["likelihood_tasks"]

    # Get the training data from saved results
    record_tasks = results_dict["record_tasks"]
    n_samples = record_tasks.shape[0]

    # Recreate the model structure (inverse models for each dimension)
    model_list_prepare = []
    likelihood_list_prepare = []

    for d in range(dim_size):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        # Use ArdGP if that's what was used in training (based on your discussion)
        model = ArdGP(
            record_tasks[:, dim_size:(dim_size + task_dim_size)],  # Task parameters
            record_tasks[:, d],  # Target dimension
            likelihood
        )
        likelihood_list_prepare.append(likelihood)
        model_list_prepare.append(model)

    # Create ModelList
    model_list = ModelList(model_list_prepare, likelihood_list_prepare, train_iter=0)

    # Load the saved states
    model_list.model.load_state_dict(model_state)
    model_list.likelihood.load_state_dict(likelihood_state)

    # Set to eval mode
    model_list.model.eval()
    model_list.likelihood.eval()

    return model_list


def testing(tasks, result_model_list, problem):
    trial, dim = tasks.shape
    tasks_optimas, tasks_uncertainty = result_model_list.test(tasks)
    tasks_sol = torch.cat([tasks_optimas, tasks], dim=1)

    tasks_ans = torch.zeros(trial)
    for tr in range(trial):
        tasks_ans[tr] = problem.evaluate(tasks_sol[tr, :])

    return tasks_ans, tasks_sol


def plot_quantile_boxplot(results, labels, my_name, percentiles=(5, 25, 50, 75, 95), gap=1, width=0.50):
    # Compute quantiles
    percentiles_tensor = torch.tensor(percentiles).float() / 100
    quantile_results = [
        torch.quantile(method_results, percentiles_tensor, dim=1).numpy()
        for method_results in results
    ]

    # percentiles_empty = torch.Tensor([])
    # for result in quantile_results:
    #     percentiles_empty = torch.cat([percentiles_empty, torch.mean(torch.from_numpy(result), dim=1).unsqueeze(0)])
    #     percentiles_empty = torch.cat([percentiles_empty, torch.std(torch.from_numpy(result), dim=1).unsqueeze(0)])
    # df = pd.DataFrame(percentiles_empty.T)

    # Set up plot
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'orange', 'green', 'red', 'purple'][:len(percentiles)]
    x_positions = []

    # Plot each method
    for method_idx, (method_name, method_quantiles) in enumerate(zip(labels, quantile_results)):
        base_position = method_idx * (len(percentiles) * width + gap)
        for quantile_idx, (color, quantile_data) in enumerate(zip(colors, method_quantiles)):
            x_pos = base_position + quantile_idx * width
            x_positions.append(x_pos)
            plt.boxplot(
                quantile_data,
                positions=[x_pos],
                widths=width,
                patch_artist=True,
                boxprops=dict(facecolor=color, color=color),
                medianprops=dict(color='black'),
                whis=[0, 100]
            )

    # Set x-ticks and legend
    x_ticks = [(idx * (len(percentiles) * width + gap) + (len(percentiles) * width) / 2) for idx in range(len(labels))]
    plt.xticks(x_ticks, labels)
    plt.xlabel("Methods")
    plt.ylabel("Objective Values")
    plt.title("Quantile Results for Different Methods")
    plt.legend(handles=[plt.Line2D([0], [0], color=color, lw=4) for color in colors],
               labels=[f"Q-{p}%" for p in percentiles],
               title="Quantiles")
    plt.grid(axis='y', linestyle='--', alpha=0.5)


# Configuration
task_number = 2  # No. of tasks
dim_size = 4  # dimension of solutions
task_dim_size = 5  # dimension of task parameters
trial_number_tot = 2  # total trials to evaluate
beta_ucb = 1.0  # UCB coeff
ec_gen = 100  # EC generation size (from your pool_gp_soo)
ec_iter = 50  # EC iterations (from your pool_gp_soo)

# Missing data structure initializations
models_tot = {}
results_tot = {}
sols_tot = {}
results_list_for_plot = []  # For plot_quantile_boxplot
method_labels = []

# Configure method and problem
# Updated to match the actual saving pattern from pool_gp_soo
method_name_list = ["{}_{}_{}_{}_{}".format(task_number, beta_ucb, "pool_gp_soo", ec_gen, ec_iter)]
problem_name = "nonlinear_sphere_high"  # Update to match your problem
direct_name = "{}_result_{}_{}".format(problem_name, dim_size, task_dim_size)

# Test data generation
sample_width = 10
test_tot = sample_width ** task_dim_size
test_data = uniform_sample(sample_width, task_dim_size)

# Problem initialization
problem = get_problem(name=problem_name, problem_params=dim_size, task_params=task_dim_size)

# Main evaluation loop
for m_id, method_name in enumerate(method_name_list):
    model_lists = []
    results_lists = torch.zeros(trial_number_tot, test_tot)
    sols_lists = torch.zeros(trial_number_tot, test_tot, dim_size + task_dim_size)

    for trial_number in range(trial_number_tot):
        # Load saved results - filename pattern matches pool_gp_soo saving
        filepath = "./{}/{}_{}.pth".format(direct_name, method_name, trial_number)
        print("Loading: {}".format(filepath))

        try:
            results = torch.load(filepath)
        except FileNotFoundError:
            print("File not found: {}".format(filepath))
            continue

        print("Model Name: {} in trial {}".format(method_name, trial_number))

        # Load the saved model directly instead of retraining
        temp_model = load_saved_model(results, dim_size, task_dim_size)

        model_lists.append(temp_model)
        temp_result, temp_sol = testing(test_data, temp_model, problem)
        results_lists[trial_number, :] = temp_result
        sols_lists[trial_number, :, :] = temp_sol

        print("Trial {} - Mean result: {:.4f}".format(trial_number, torch.mean(temp_result)))

    models_tot[method_name] = model_lists
    results_tot[method_name] = results_lists
    sols_tot[method_name] = sols_lists

    # Prepare data for plotting
    results_list_for_plot.append(results_lists)
    method_labels.append(method_name)

    # Print statistics
    mean_result = torch.mean(results_lists)
    std_result = torch.std(results_lists)
    print("\nMethod: {}".format(method_name))
    print("Overall Mean: {:.4f} ± {:.4f}".format(mean_result, std_result))

# Plot the results using plot_quantile_boxplot
if len(results_list_for_plot) > 0:
    plot_quantile_boxplot(
        results_list_for_plot,
        method_labels,
        problem_name,
        percentiles=(5, 25, 50, 75, 95)
    )
    plt.show()