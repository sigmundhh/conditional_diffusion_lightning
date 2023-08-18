## The policy (DDPM-policy)
This note is about the policy, which consist of a DDPM diffusion model to infer the actions to take.

**Inputs (observations):** 
- Images
- agent_pos
Both for `obs_horizon` steps into the past.
**Outputs:** 
- Predicted actions for the future `pred_horizon` steps (B, pred_horizon, action_dim)
We then do reverse diffusion to come up with actions over a future horizon with length `pred_horizon`. 

There are several components of the policy:
- Noise predictor
- Visual encoder
- Sinusoidal embeddings

### Visual encoder
They use a [[ResNet18]] to encode the images. They train it too. 

Input: 
- Image batch `(B, C, W, H)`
Output:
- Batch of visual features: `(B, visual_emb_dim)`

In the case of [[ResNet18]], `visual_emb_dim=512`.


### Noise predictor (ContextUNet)
This note concerns the noise-predictor used in the policy.

The input is:
- x: (B, act_horizon, act_dim) noisy action sequence
- c: (B, obs_horizon*(obs_dim + vis_emb_dim)) context
- t: (B,) time step

The output is:
- x: (B, act_horizon, act_dim) predicted noise

The policy utilizes the trained backward diffusion model to make action predictions. Following DDPM, the procedure goes:
- Sample some noise
- For timestep: t=T...1
	- Get noise-prediction
	- Sample from denoised distribution