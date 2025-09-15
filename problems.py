import numpy as np
import torch


def get_linear_mat(k, task_params=2):
    """Generate linear transformation matrix for task parameters"""
    mat_main_placeholder = np.zeros((task_params, k))
    mat_placeholder = np.zeros((2, k))

    assert k > 1
    mat_placeholder[1, :] = np.arange(0, k) / (k - 1)
    mat_placeholder[0, :] = np.arange(k - 1, -1, -1) / (k - 1)

    if task_params > 2:
        task_limit_num = task_params - 1
        task_indices = np.arange(k)
        for i in range(task_limit_num):
            temp_anchor = task_indices % task_limit_num
            mat_main_placeholder[i:(i + 2), temp_anchor == i] = mat_placeholder[:, temp_anchor == i]
        return mat_main_placeholder.T
    else:
        return mat_placeholder.T


def nonlinear_map(x):
    """Nonlinear transformation for task parameters"""
    return (np.sin(5 * (x + 0.5)) + 1) / 2


def middle_nonlinear_map(x):
    """Middle nonlinear transformation for task parameters"""
    return 0.3 * (1 + np.sin(5 * np.pi * x - np.pi / 2)) + 0.3 * np.square(x - 0.2)


def get_problem(name, problem_params=None, task_params=2):
    """Factory function to create problem instances"""
    name = name.lower()

    PROBLEM = {
        # Sphere variants
        'nonlinear_sphere_high': Sphere(n_dim=problem_params, mode="nonlinear", task_param=task_params),
        'middle_nonlinear_sphere_high': Sphere(n_dim=problem_params, mode="middle_nonlinear", task_param=task_params),

        # Ackley variants
        'nonlinear_ackley_high': Ackley(n_dim=problem_params, mode="nonlinear", task_param=task_params),
        'middle_nonlinear_ackley_high': Ackley(n_dim=problem_params, mode="middle_nonlinear", task_param=task_params),

        # Rastrigin variants
        'nonlinear_rastrigin_20_high': Rastrigin(n_dim=problem_params, mode="nonlinear",
                                                 task_param=task_params, factor=20, mean=0, std=1),
        'middle_nonlinear_rastrigin_20_high': Rastrigin(n_dim=problem_params, mode="middle_nonlinear",
                                                        task_param=task_params, factor=20, mean=0, std=1),

        # Griewank variants
        'nonlinear_griewank_high': Griewank(n_dim=problem_params, mode="nonlinear",
                                            task_param=task_params, factor=600, mean=0, std=1),
        'middle_nonlinear_griewank_high': Griewank(n_dim=problem_params, mode="middle_nonlinear",
                                                   task_param=task_params, factor=600, mean=0, std=1),
    }

    if name not in PROBLEM:
        raise Exception(f"Problem '{name}' not found.")

    return PROBLEM[name]


class Sphere:
    def __init__(self, n_dim=10, mode="linear", task_param=2, factor=4):
        self.n_dim = n_dim
        self.n_obj = 1
        self.mode = mode
        self.shift_mat = get_linear_mat(self.n_dim, task_param)
        self.factor = factor

    def evaluate(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        sol = x[:self.n_dim]
        hook = x[self.n_dim:]
        shift = np.squeeze(np.matmul(self.shift_mat, np.expand_dims(hook, axis=1)), axis=1)

        if self.mode == "nonlinear":
            shift = nonlinear_map(shift)
        elif self.mode == "middle_nonlinear":
            shift = middle_nonlinear_map(shift)

        return np.sum(np.power(sol * self.factor - shift * self.factor, 2))


class Ackley:
    def __init__(self, n_dim=10, mode="linear", task_param=2, factor=4):
        self.n_dim = n_dim
        self.n_obj = 1
        self.mode = mode
        self.shift_mat = get_linear_mat(self.n_dim, task_param)
        self.factor = factor

    def evaluate(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        sol = x[:self.n_dim]
        hook = x[self.n_dim:]
        shift = np.squeeze(np.matmul(self.shift_mat, np.expand_dims(hook, axis=1)), axis=1)

        if self.mode == "nonlinear":
            shift = nonlinear_map(shift)
        elif self.mode == "middle_nonlinear":
            shift = middle_nonlinear_map(shift)

        return -20 * np.exp(-0.2 * np.sqrt(np.mean(np.power(sol * self.factor - shift * self.factor, 2)))) - \
            np.exp(np.mean(np.cos(2 * np.pi * self.factor * (sol - shift)))) + 20 + np.exp(1)


class Griewank:
    def __init__(self, n_dim=10, mode="linear", task_param=2, factor=4, mean=0, std=1):
        self.n_dim = n_dim
        self.n_obj = 1
        self.mode = mode
        self.shift_mat = get_linear_mat(self.n_dim, task_param)
        self.factor = factor
        self.mean = mean
        self.std = std

    def evaluate(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        sol = x[:self.n_dim]
        hook = x[self.n_dim:]
        shift = np.squeeze(np.matmul(self.shift_mat, np.expand_dims(hook, axis=1)), axis=1)

        if self.mode == "nonlinear":
            shift = nonlinear_map(shift)
        elif self.mode == "middle_nonlinear":
            shift = middle_nonlinear_map(shift)

        return (np.sum(np.power(sol * self.factor - shift * self.factor, 2) / 4000) - \
                np.prod(
                    np.cos(sol * self.factor - shift * self.factor / np.sqrt(np.linspace(1, self.n_dim, self.n_dim)))) \
                + 1 - self.mean) / self.std


class Rastrigin:
    def __init__(self, n_dim=10, mode="linear", task_param=2, factor=4, mean=0, std=1):
        self.n_dim = n_dim
        self.n_obj = 1
        self.mode = mode
        self.shift_mat = get_linear_mat(self.n_dim, task_param)
        self.factor = factor
        self.mean = mean
        self.std = std

    def evaluate(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        sol = x[:self.n_dim]
        hook = x[self.n_dim:]
        shift = np.squeeze(np.matmul(self.shift_mat, np.expand_dims(hook, axis=1)), axis=1)

        if self.mode == "nonlinear":
            shift = nonlinear_map(shift)
        elif self.mode == "middle_nonlinear":
            shift = middle_nonlinear_map(shift)

        return (np.sum(np.power((sol - shift) * self.factor, 2) -
                       10 * np.cos(2 * np.pi * self.factor * (sol - shift)) + 10) - self.mean) / self.std