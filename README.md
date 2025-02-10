# Code for Trajectory Inference with Smooth Schr¨odinger Bridges

Run command **pip install -r requirements.txt** to install all the required packages

The three tutorial notebooks correspond to reproducing experiments for different tasks:

**Tutorial_hold_out.ipynb**: code for hold out time step tasks

**Tutorial_one_by_one_matching.ipynb**: code for one by one particle tracking tasks

**Tutorial_trajectory_inference_tasks.ipynb**: code for pattern matching (no hold out) tasks

To run the three notebooks, make sure datasets files exist in the G_data folder, change the **exp_index** variable to reproduce experiment for different kinds of datasets. The second block of all notebooks contains parameters for the algorithm that can be tuned, the remaining parts need not to be changed.
