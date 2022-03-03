## When Proxy-Driven Learning Is No Better Than Random:The Consequences of Representational Incompleteness
Paper submitted for PLoS 2022.

This repository conatins the **minimal data set** to reproduce the reported findings in the study and the complete **implementation of the simulation models**. 

## Minimal data set:

All the data points used to build the graphs in csv format are in the folder `output/`
Summaries and explanations of the data for all graphs are also found in the folder `output/` 


## The simulation code:

Run the following python scripts in the source folder `src/`

```
python3 simulate_baseline.py
python3 plot_baseline.py

python3 simulate_divergent.py
python3 plot_divergent.py

python3 simulate_hidden.py
python3 plot_hidden.py
```

The simulation results are written to the `npy/` folder; 
the subfolders `m1/`, `clickon/`, and `hidden/` correspond to the three models,  
User model 1, model 2 of dissatisfied user, and the model with hidden categories.

Similarly, the figures are stored under the three subfolders of the `plot/` folder.
