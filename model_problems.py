import torch
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from models import ModelList

# Note: These imports need to be available from your models.py file
# from models import ModelList, evaluation, compute_gradient_list

def compute_gradient(model, likelihood, x_test: torch.Tensor, mode=1):
    model.eval()
    likelihood.eval()

    X = torch.autograd.Variable(torch.Tensor(x_test), requires_grad=True)

    def mean_f(X):
        return likelihood(model(X)).mean.sum()

    def var_f(X):
        return likelihood(model(X)).var.sum()

    def mean_df(X):
        return torch.autograd.functional.jacobian(mean_f, X, create_graph=True).sum()

    def var_df(X):
        return torch.autograd.functional.jacobian(var_f, X, create_graph=True).sum()

    if mode == 1:
        dydtest_x_ag = torch.autograd.functional.jacobian(mean_f, X)
        ag = torch.abs(dydtest_x_ag.detach())
        return ag
    elif mode == 2:
        dy2dtest_x2_ag = torch.autograd.functional.jacobian(mean_df, X)
        ag = torch.abs(dy2dtest_x2_ag.detach())
        return ag
    else:
        return None

def compute_gradient_list(inverse_model_list, x_test: torch.Tensor, mode=1):
    num_models = len(inverse_model_list.model.models)
    num_samples, _ = x_test.shape
    ans = torch.zeros(num_samples)

    for i in range(num_models):
        cur_ans = compute_gradient(inverse_model_list.model.models[i],
                                   inverse_model_list.likelihood.likelihoods[i],
                                   x_test,
                                   mode)

        ans = ans + torch.mean(cur_ans, dim=1)

    return ans / num_models

def model_det(models, data):
    """Compute determinant of covariance matrix for model selection"""
    data_tot = torch.Tensor([])
    for cur_model in models:
        cur_model.eval()
        cur_data = torch.linalg.det(cur_model.covar_module.base_kernel(data).to_dense()).detach()[:, None]
        data_tot = torch.cat([data_tot, cur_data], dim=1)
    data_tot = torch.mean(data_tot, dim=1)
    return data_tot


class OPT(Problem):
    """Optimization problem wrapper for pymoo"""

    def __init__(self, input_func, n_var, n_obj, xl=0, xu=1):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)
        self.input_func = input_func

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.input_func.evaluate(x)


class ModelGradientMyopic:
    """
    Myopic model gradient evaluation for SOO-enhanced task selection
    Used specifically in Pool GP SOO for generating new tasks
    """

    def __init__(self, inverse_model_list: ModelList, n_dim, n_task_params, n_tasks, mode, ref_tasks, if_soo):
        self.inverse_model_list = inverse_model_list
        self.n_dim = n_dim
        self.n_obj = 2
        self.n_task_params = n_task_params
        self.n_tasks = n_tasks
        self.mode = mode
        self.ref_tasks = ref_tasks
        self.if_soo = if_soo

    def evaluate(self, solution: torch.tensor = None):
        """
        Evaluate task candidates using gradient information and covariance determinant

        Args:
            solution: Task parameter candidates of shape (n_sols, n_task_params * n_tasks)

        Returns:
            numpy array of shape (n_sols, 2) with negative gradient and determinant values
        """
        n_sols, n_task_dim = solution.shape
        assert n_task_dim == self.n_task_params * self.n_tasks

        # Compute gradient information
        new_solution = solution.reshape(-1, self.n_task_params)
        new_solution_torch = torch.from_numpy(new_solution).float()

        grad_result = compute_gradient_list(self.inverse_model_list, new_solution_torch, self.mode)
        # grad_result: Tensor Shape: (n_sols * n_tasks, )
        grad_result = grad_result.view(n_sols, -1)
        grad_result = torch.mean(grad_result, dim=1)

        # SOO mode: disable gradient information
        if self.if_soo:
            grad_result = grad_result * 0
        # grad_result: Tensor Shape: (n_sols, )

        # Compute determinant of covariance matrix
        new_solution_torch = new_solution_torch.view(n_sols, self.n_tasks, self.n_task_params)
        new_solution_torch = torch.cat([new_solution_torch,
                                        self.ref_tasks.unsqueeze(0).repeat(n_sols, 1, 1)], dim=1)
        data_result = model_det(self.inverse_model_list.model.models, new_solution_torch)
        # data_result: Tensor Shape: (n_sols, )

        # Return negative values for minimization
        result = - torch.cat([grad_result.unsqueeze(1), data_result.unsqueeze(1)], dim=1)

        return result.numpy()


def ec_active_myopic_moo(inverse_model_list: ModelList, ec_gen: int, ec_iter: int, n_dim: int,
                         n_task_params: int, n_tasks: int, mode: int, ref_tasks: torch.Tensor, if_soo: bool = False):
    """
    Evolutionary computation for myopic multi-objective optimization of task selection

    This is the core function used by Pool GP SOO for generating new promising tasks
    using SOO (Simultaneous Optimistic Optimization) enhanced evolutionary computation.

    Args:
        inverse_model_list: Trained inverse GP models
        ec_gen: Population size for EC
        ec_iter: Number of EC iterations
        n_dim: Dimension of solution space
        n_task_params: Dimension of task parameter space
        n_tasks: Number of tasks (should be 1 for myopic)
        mode: Gradient computation mode
        ref_tasks: Reference tasks for comparison
        if_soo: Whether to enable SOO mode

    Returns:
        torch.Tensor of selected task parameters from Pareto front
    """
    assert n_tasks == 1, "Myopic optimization requires n_tasks=1"

    # Create problem instance
    problem_current = ModelGradientMyopic(inverse_model_list,
                                          n_dim,
                                          n_task_params,
                                          n_tasks,
                                          mode,
                                          ref_tasks,
                                          if_soo)

    # Setup optimization problem
    obj_problem = OPT(problem_current, n_var=problem_current.n_task_params * n_tasks, n_obj=problem_current.n_obj)
    algorithm = NSGA2(pop_size=ec_gen)

    # Run optimization
    res = minimize(obj_problem,
                   algorithm,
                   ('n_gen', ec_iter),
                   seed=1,
                   eliminate_duplicates=True,
                   verbose=False)

    tot_pf, _ = res.X.shape
    sample_tasks = torch.from_numpy(res.X).float()

    return sample_tasks