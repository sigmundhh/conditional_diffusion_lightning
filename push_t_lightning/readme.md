## Diffusion Policy in the Push-T environment, PyTorch Lightning implementation

This repo contains code to train and visualize the performance of policy based on diffusion.

It consists of the following parts:
- The `push_t_env` folder defines the Push-T  environment
- The `push_t_dataset` folder defines the DataLoader emitting demonstration data used in training.
- The `ddpm_policy` folder defines the policy architecture, and the training-related functionality.
- The notebook `train_and_eval_policy.ipynb` trains the model on the dataset and visualizes the resulting performance.