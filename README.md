# Code for Trajectory Inference with Smooth Schr¨odinger Bridges

Run command **pip install -r requirements.txt** to install all the required packages

The three tutorial notebooks correspond to reproducing experiments for different tasks:

**Tutorial_hold_out.ipynb**: code for hold out time step tasks

**Tutorial_one_by_one_matching.ipynb**: code for one by one particle tracking tasks

**Tutorial_trajectory_inference_tasks.ipynb**: code for pattern matching (no hold out) tasks

To run the three notebooks, make sure datasets files exist in the G_data folder, change the **exp_index** variable to reproduce experiment for different kinds of datasets. The second block of all notebooks contains parameters for the algorithm that can be tuned (check paper and code explanations for tuning), the remaining parts need not to be changed. Simply running all the blocks will yield the results.

To experiment on customized dataset, the dataset should be a **npz** file stored in the **G_data** folder with the **arr_0** being the key to the dataset. The dataset should have shape **(T,N,D)** where **T is not total number of snapshots including initial observation, N is the number of observations for each snapshot (we assume it is constant over time), D is the dimension of the data (make sure the data dimension do not exceed 5)**.

Results: directory to save trajectories and matching for each experiment.

Experiments: directory to save precomputed tensor and messages for each experiment.

