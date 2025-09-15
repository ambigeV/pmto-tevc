# Parametric Multi-Task Optimization



A Bayesian optimization framework that combines Gaussian Processes with task parameters for multi-task optimization problems.



## Quick Start



### Running Optimization



Execute the main optimization script via command line:



```bash

python main.py --ec_gen 100 --ec_iter 50 --trials 2 --template nonlinear

```



#### Command Line Arguments



\- `--ec\_gen`: EC population size (default: 100)

\- `--ec\_iter`: EC iteration count (default: 50) 

\- `--trials`: Number of optimization trials (default: 2)

\- `--template`: Problem template - `nonlinear` or `middle\_nonlinear` (default: nonlinear)



The script will optimize across 4 benchmark problems: sphere, ackley, rastrigin\_20, and griewank.



\### Evaluating Results



Run the evaluation script to analyze and visualize results:



```bash

python eval\_script.py

```



\#### Customizing Evaluation



Edit the configuration section in `eval\_script.py`:



```python

\# Configuration

task\_number = 2          # Number of tasks

dim\_size = 4            # Solution dimensions  

task\_dim\_size = 5       # Task parameter dimensions

trial\_number\_tot = 2    # Trials to evaluate

sample\_width = 10       # Test grid resolution

problem\_name = "nonlinear\_sphere\_high"  # Problem to evaluate

```



\## Output



\- \*\*Results\*\*: Saved in `{problem\_name}\_result\_{dim\_size}\_{task\_params}/` directories

\- \*\*Models\*\*: Trained GP models with state dictionaries

\- \*\*Visualization\*\*: Quantile boxplots showing performance across different percentiles



The evaluation script loads saved models and generates performance statistics and plots for comprehensive result analysis.

