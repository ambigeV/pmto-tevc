import gpytorch
import numpy
from gpytorch.mlls import SumMarginalLogLikelihood
import torch
import torch.distributions as dist

device = torch.device("cpu")


class VanillaGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.no_of_data, self.no_of_dim = train_x.shape
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ArdGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.no_of_data, self.no_of_dim = train_x.shape
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=self.no_of_dim))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ModelList:
    def __init__(self, model_list, likelihood_list, train_iter):
        self.model = gpytorch.models.IndependentModelList(*model_list)
        self.likelihood = gpytorch.likelihoods.LikelihoodList(*likelihood_list)
        self.train_iter = train_iter
        self.model_len = len(self.model.models)

    def train(self):
        self.model.train()
        self.likelihood.train()
        mll = SumMarginalLogLikelihood(self.likelihood, self.model)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)

        for i in range(self.train_iter):
            optimizer.zero_grad()
            output = self.model(*self.model.train_inputs)
            loss = -mll(output, self.model.train_targets)
            loss.backward()
            optimizer.step()

    def test(self, test_x):
        # return tensors of size [sample_size, no_of_models(dimension)]
        if isinstance(test_x, numpy.ndarray):
            test_x = torch.from_numpy(test_x).float()

        self.model.eval()
        self.likelihood.eval()
        dimensions = len(self.model.models)
        test_x_list = [test_x for i in range(dimensions)]
        mean_ans, std_ans = torch.Tensor([]), torch.Tensor([])
        predictions = self.likelihood(*self.model(*test_x_list))

        for i in range(dimensions):
            mean_ans = torch.cat([mean_ans, predictions[i].mean.unsqueeze(1)], dim=1)
            std_ans = torch.cat([std_ans, predictions[i].stddev.unsqueeze(1)], dim=1)

        mean_ans = torch.clamp(mean_ans.detach(), 0, 1)
        std_ans = torch.clamp(std_ans.detach(), 0, 1)

        return mean_ans, std_ans


def evaluation(model, likelihood, test_x, if_grad=True):
    model.eval()
    likelihood.eval()

    if isinstance(test_x, numpy.ndarray):
        observed_pred = likelihood(model(torch.from_numpy(test_x).float()))
    elif isinstance(test_x, tuple):
        observed_pred = likelihood(model(*test_x))
    else:
        observed_pred = likelihood(model(test_x))

    if if_grad:
        return observed_pred.mean, torch.sqrt(observed_pred.variance)
    else:
        return observed_pred.mean.detach(), torch.sqrt(observed_pred.variance).detach()


def upper_confidence_bound(mean_tensor, std_tensor, beta=-0.5, beta_mean=1):
    return beta_mean * mean_tensor + beta * std_tensor


def scalar_obj(x, a, alpha):
    return 0 * torch.max(a * x, dim=1).values + torch.matmul(x, a) * alpha


def next_sample(model_list, likelihood_list, sol_dim, weights, mode, fixed_solution,
                alpha=0.05, beta=-0.5, beta_mean=1, opt_iter=20, num_restart=8,
                aq_func="UCB", if_cuda=False, if_debug=False, y_best=None):
    """
    Acquisition function optimization for Bayesian Optimization

    Args:
        model_list: List of GP models
        likelihood_list: List of likelihoods
        sol_dim: Dimension of solution space
        weights: Weights for scalarization
        mode: Evaluation mode (1: solutions only, 2: concat fixed after, 4: maximize)
        fixed_solution: Fixed part of the solution
        opt_iter: Number of optimization iterations
        num_restart: Number of random restarts
    """

    # Initialize random starting points
    solutions = torch.rand(num_restart, sol_dim)
    solutions.requires_grad = True
    optimizer = torch.optim.Adam([solutions], lr=1e-2)

    for k in range(opt_iter):
        optimizer.zero_grad()

        means_list = []
        stds_list = []

        # Evaluate each model
        for i in range(len(model_list)):
            if mode == 1:
                mean_values, std_values = evaluation(model_list[i], likelihood_list[i], solutions)
            elif mode == 2:
                # Concatenate solutions with fixed solution
                combined_input = torch.cat([solutions, fixed_solution.unsqueeze(0).repeat(num_restart, 1)], dim=1)
                mean_values, std_values = evaluation(model_list[i], likelihood_list[i], combined_input)
            elif mode == 4:
                # Maximization mode
                mean_values, std_values = evaluation(model_list[i], likelihood_list[i], solutions)
                mean_values = -mean_values
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            means_list.append(mean_values.unsqueeze(1))
            stds_list.append(std_values.unsqueeze(1))

        means = torch.cat(means_list, dim=1)
        stds = torch.cat(stds_list, dim=1)

        # Calculate acquisition function (UCB)
        ucb = upper_confidence_bound(means, stds, beta, beta_mean)
        outputs = scalar_obj(ucb, weights, alpha)

        # Optimize or return result
        if k < opt_iter - 1:
            loss = outputs.sum()
            loss.backward()
            optimizer.step()

            # Clamp to valid bounds
            with torch.no_grad():
                solutions.clamp_(0, 1)
        else:
            # Return best solution
            sorted_outputs, ind = torch.sort(outputs)
            ans = solutions[ind, :][0].detach()
            ans.clamp_(0, 1)
            return ans