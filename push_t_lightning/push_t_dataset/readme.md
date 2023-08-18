## The dataset
The dataset gives us samples on the form:
- image: (obs_horizon, 3, 96, 96)
- agent_pos (obs_horizon, 2)
- action (pred_horizon, 2)

The data is also normalized, in contrast to data we would get from the environment. 

We construct the dataset given a dataset-path, pred_horizon, obs_horizon and an action horizon. We then can construct a DataLoader that we can use during training. 