import random
import time
import argparse

import gpytorch
import torch
import numpy as np
from sympy import ceiling

from models import VanillaGP, ModelList, next_sample, device, ArdGP
from problems import get_problem
from model_problems import ec_active_myopic_moo

# Global configuration
task_number = 2
dim_size = 4
task_params = 5
beta_ucb = 1.0


def configure_problem(problem_name):
    params = dict()
    params["ind_size"] = task_number
    # params["tot_init"] = 200
    # params["tot_budget"] = 2000
    params["tot_init"] = task_number * 2
    params["tot_budget"] = task_number * 3
    params["aqf"] = "ucb"
    params["train_iter"] = 500
    params["test_iter"] = 50
    params["problem_name"] = problem_name
    params["n_obj"] = 1
    params["n_dim"] = dim_size
    params["n_task_params"] = task_params
    return params


def fetch_task_lhs(task_param=2, task_size=10):
    """Load pre-computed Latin Hypercube Samples from init directory"""
    import os
    from scipy.stats import qmc

    init_dir = "./init"
    filename = f"task_list_{task_size}_{task_param}.pth"
    filepath = os.path.join(init_dir, filename)

    if not os.path.exists(filepath):
        # Create init directory if it doesn't exist
        if not os.path.exists(init_dir):
            os.makedirs(init_dir)
            print(f"Created directory: {init_dir}")

        # Generate and save the requested LHS file
        print(f"Generating {filename}...")
        sampler = qmc.LatinHypercube(task_param)
        samples = sampler.random(task_size)
        task_list = [torch.from_numpy(samples[i]).float() for i in range(task_size)]

        torch.save(task_list, filepath)
        print(f"Created: {filepath}")

        return task_list

    # Load existing file
    results = torch.load(filepath)
    return results


def solver_pool_gp_soo(problem_params, ec_config, trial):
    """
    Pool GP SOO solver implementation
    """

    # Extract problem parameters
    ind_size = problem_params["ind_size"]
    tot_init = problem_params["tot_init"]
    tot_budget = problem_params["tot_budget"]
    train_iter = problem_params["train_iter"]
    test_iter = problem_params["test_iter"]
    problem_name = problem_params["problem_name"]
    n_obj = problem_params["n_obj"]
    n_dim = problem_params["n_dim"]
    n_task_params = problem_params["n_task_params"]

    # Extract EC configuration
    ec_gen = ec_config["ec_gen"]
    ec_iter = ec_config["ec_iter"]

    # Initialize problem
    problem = get_problem(name=problem_name, problem_params=n_dim, task_params=n_task_params)

    # Initialize pool variables
    pool_max = ind_size + ceiling(tot_budget // ind_size)
    pool_active = ind_size
    pool_budget = 0

    pool_bayesian_vector = torch.zeros(tot_budget, n_dim + n_task_params + n_obj)
    pool_budget_vector = torch.zeros(pool_max)
    pool_bayesian_best_results = torch.ones(pool_max, n_dim + n_task_params + n_obj) * 1e6

    # Initialize tasks using Latin Hypercube Sampling
    task_list = torch.stack(fetch_task_lhs(n_task_params, ind_size))
    sample_size = tot_init // ind_size

    # Initialization phase
    print(f"Trial {trial + 1}: Initialization phase")
    for i in range(pool_active):
        for j in range(sample_size):
            # Generate random solution
            pool_bayesian_vector[pool_budget, :n_dim] = torch.rand(n_dim)
            # Set task parameter
            pool_bayesian_vector[pool_budget, n_dim:(n_dim + n_task_params)] = task_list[i, :]
            # Evaluate solution
            pool_bayesian_vector[pool_budget, (n_dim + n_task_params):(n_dim + n_task_params + n_obj)] = \
                problem.evaluate(pool_bayesian_vector[pool_budget, :(n_dim + n_task_params)])

            # Update best results
            if pool_bayesian_vector[pool_budget, -1] < pool_bayesian_best_results[i, -1]:
                pool_bayesian_best_results[i, :] = pool_bayesian_vector[pool_budget, :]
                print(f"Task {i + 1} in Iteration {j + 1}: Best Obj {pool_bayesian_vector[pool_budget, -1]:.6f}")

            # Update counters
            pool_budget_vector[i] += 1
            pool_budget += 1

    # Main optimization loop
    iteration = 0
    while pool_budget < tot_budget:
        iteration += 1
        print(
            f"Trial {trial + 1}: Main Loop Iteration {iteration}, Budget {pool_budget}/{tot_budget}, Active Tasks: {pool_active}")

        # Train Forward Model
        temp_vectors = pool_bayesian_vector[:pool_budget, :]

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = VanillaGP(temp_vectors[:, :(n_dim + n_task_params)],
                          temp_vectors[:, (n_dim + n_task_params):(n_dim + n_task_params + n_obj)].squeeze(1),
                          likelihood)

        model_list = ModelList([model], [likelihood], train_iter)
        model_list.train()

        # Train Inverse Model
        temp_bayesian_best_results = pool_bayesian_best_results[:pool_active, :].clone()

        inverse_model_list_prepare = []
        inverse_likelihood_list_prepare = []

        for d in range(n_dim):
            inverse_likelihood = gpytorch.likelihoods.GaussianLikelihood()
            inverse_model = ArdGP(
                temp_bayesian_best_results[:, n_dim:(n_dim + n_task_params)],
                temp_bayesian_best_results[:, d],
                inverse_likelihood)

            inverse_likelihood_list_prepare.append(inverse_likelihood)
            inverse_model_list_prepare.append(inverse_model)

        # Train inverse models
        inverse_model_list = ModelList(inverse_model_list_prepare,
                                       inverse_likelihood_list_prepare,
                                       train_iter * 3)
        inverse_model_list.train()

        # Generate new task using SOO-enhanced EC
        ec_task_results = ec_active_myopic_moo(inverse_model_list,
                                               ec_gen, ec_iter,
                                               n_dim, n_task_params,
                                               1, 1, task_list,  # method_mode = 1
                                               True)  # if_soo = True
        ec_size, _ = ec_task_results.shape
        new_task = ec_task_results[np.random.randint(ec_size), :].view(1, -1)

        # Add new task to pool
        task_list = torch.cat([task_list, new_task])
        pool_active += 1

        # Optimize each task in the pool (newest to oldest)
        for i in range(pool_active - 1, -1, -1):
            # Find optimal solution for current task using forward model
            ans = next_sample([model_list.model.models[0]],
                              [model_list.likelihood.likelihoods[0]],
                              n_dim,
                              torch.tensor([1], dtype=torch.float32).to(device),
                              mode=2,  # fixed task parameter mode
                              fixed_solution=task_list[i, :],
                              opt_iter=test_iter,
                              if_debug=False)

            # Clamp solution to valid bounds
            ans = torch.clamp(ans, 0, 1)
            param = ans.unsqueeze(0)

            # Store the evaluation
            pool_bayesian_vector[pool_budget, :n_dim] = param.clone()
            pool_bayesian_vector[pool_budget, n_dim:(n_dim + n_task_params)] = task_list[i, :]

            # Evaluate the solution
            pool_bayesian_vector[pool_budget, (n_dim + n_task_params):(n_dim + n_task_params + n_obj)] = \
                problem.evaluate(pool_bayesian_vector[pool_budget, :(n_dim + n_task_params)])

            # Update best results if improved
            if pool_bayesian_vector[pool_budget, -1] < pool_bayesian_best_results[i, -1]:
                pool_bayesian_best_results[i, :] = pool_bayesian_vector[pool_budget, :]
                print(f"Task {i + 1} in Budget {pool_budget}: Best Obj {pool_bayesian_vector[pool_budget, -1]:.6f}")

            # Update budget counters
            pool_budget_vector[i] += 1
            pool_budget += 1

            # Break if budget exhausted
            if pool_budget >= tot_budget:
                break

    # Build final inverse model and save results
    model_list_prepare = []
    likelihood_list_prepare = []

    for d in range(n_dim):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ArdGP(  # Changed from VanillaGP to ArdGP
            pool_bayesian_best_results[:pool_active, n_dim:(n_dim + n_task_params)],
            pool_bayesian_best_results[:pool_active, d],
            likelihood)
        likelihood_list_prepare.append(likelihood)
        model_list_prepare.append(model)

    # Train final models
    model_list = ModelList(model_list_prepare, likelihood_list_prepare, train_iter * 3)
    model_list.train()

    # Save model records
    model_records = {
        "ind": True,
        "dim": n_dim,
        "tasks": task_list,
        "record_tasks": pool_bayesian_best_results[:pool_active, :],
        "all_tasks": pool_bayesian_vector,
        "model_tasks": model_list.model.state_dict(),
        "likelihood_tasks": model_list.likelihood.state_dict()
    }

    # Save to file
    import os
    if not os.path.exists(direct_name):
        os.makedirs(direct_name)
        print(f"Created results directory: {direct_name}")
    torch.save(model_records, f"./{direct_name}/{task_number}_{beta_ucb}_pool_gp_soo_{ec_gen}_{ec_iter}_{trial}.pth")

    print(f"Trial {trial + 1} completed. Results saved.")
    return None


def main_solver_pool_gp_soo(trials, ec_config):
    """Main solver for pool_gp_soo only"""
    problem_params = configure_problem(problem_name)

    for trial in range(trials):
        solver_pool_gp_soo(problem_params, ec_config, trial)


if __name__ == "__main__":
    # Parse command-line arguments for EC parameters
    parser = argparse.ArgumentParser(description='Pool GP SOO Optimization')
    parser.add_argument('--ec_gen', type=int, default=100, help='EC population size')
    parser.add_argument('--ec_iter', type=int, default=50, help='EC iteration count')
    parser.add_argument('--trials', type=int, default=2, help='Number of trials')
    parser.add_argument('--template', type=str, default='nonlinear',
                        choices=['nonlinear', 'middle_nonlinear'], help='Problem name template')

    args = parser.parse_args()

    # Create EC configuration
    ec_config = {
        "ec_gen": args.ec_gen,
        "ec_iter": args.ec_iter
    }

    # Define problem types
    problem_name_list = ["sphere", "ackley", "rastrigin_20", "griewank"]
    problem_name_template = args.template

    # Run optimization for each problem
    for cur_name in problem_name_list:
        problem_name = f"{problem_name_template}_{cur_name}_high"
        direct_name = f"{problem_name}_result_{dim_size}_{task_params}"

        print(f"Running {problem_name} with Pool GP SOO")
        print(f"Using EC parameters: gen={args.ec_gen}, iter={args.ec_iter}")

        # Run solver
        main_solver_pool_gp_soo(trials=args.trials, ec_config=ec_config)