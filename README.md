This is the source code for the paper: When Proxy-Driven Learning Is No Better Than Random:
The Consequences of Representational Incompleteness
submitted for PLoS 2022.

In the source folder src/

Run the simulations: run the following python simulation scripts:

python3 simulate_baseline.py
python3 plot_baseline.py

python3 simulate_divergent.py
python3 plot_divergent.py

python3 simulate_hidden.py
python3 plot_hidden.py


The simulation results are written to the npy folder; 
the subfolders m1/, clickon/, and hidden/ corresponds to the three models  
User Model 1, Model 2 of dissatisfied user, and model with hidden categories.

Similarly, the figures are stored under the three subfolders under the plot folder.
