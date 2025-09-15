# Parametric Multi-Task Optimization



A Bayesian optimization framework that combines Gaussian Processes with task parameters for multi-task optimization problems.



## Quick Start



### Running Optimization



Execute the main optimization script via command line:



```bash

python main.py --ec_gen 100 --ec_iter 50 --trials 2 --template nonlinear

```



#### Command Line Arguments



- `--ec_gen`: EC population size (default: 100)

- `--ec_iter`: EC iteration count (default: 50) 

- `--trials`: Number of optimization trials (default: 2)

- `--template`: Problem template - `nonlinear` or `middle_nonlinear` (default: nonlinear)



The script will optimize across 4 benchmark problems: sphere, ackley, rastrigin_20, and griewank.



### Evaluating Results



Run the evaluation script to analyze and visualize results:



```bash

python eval.py

```




## Output



- **Results**: Saved in `{problem_name}_result_{dim_size}_{task_params}/` directories

- **Models**: Trained GP models with state dictionaries

- **Visualization**: Quantile boxplots showing performance across different percentiles



The evaluation script loads saved models and generates performance statistics and plots for comprehensive result analysis.

