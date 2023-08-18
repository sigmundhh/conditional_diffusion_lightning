import numpy as np
import pytorch_lightning as pl
import wandb
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from ddpm_policy.visual_encoder import VisualEncoder
from ddpm_policy.cond_unet_1d import ConditionalUnet1D

class DDPMPolicy(pl.LightningModule):
    """
    Implements a DDPM policy that takes in an image and agent position
    and outputs actions over pred_horizon timesteps.
    """
    def __init__(
            self,
            mask_probability=0.5, 
            num_diff_steps=1000,
            beta_min = 1e-4,
            beta_max = 1e-2,
            pred_horizon = 16,
            obs_horizon = 2,
            action_dim = 2,
            numerical_obs_dim = 2,
            dataloader = None,
            num_epochs = 100
    ) -> None: 
        super().__init__()
        self.visual_encoder = VisualEncoder()
        visual_emb_dim = self.visual_encoder(torch.randn(1, 3, 96, 96)).shape[-1]
        obs_dim = visual_emb_dim + numerical_obs_dim
        self.noise_predictor = ConditionalUnet1D(
                                    input_dim=action_dim,
                                    global_cond_dim=obs_dim*obs_horizon)
        self.criterion = torch.nn.MSELoss()
        self.mask_probability = mask_probability
        self.num_diff_steps = num_diff_steps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.noise_scheduler = DDPMScheduler(
                                    num_train_timesteps=num_diff_steps,
                                    # the choice of beta schedule has big impact on performance
                                    # we found squared cosine works the best
                                    beta_schedule='squaredcos_cap_v2',
                                    # clip output to [-1,1] to improve stability
                                    clip_sample=True,
                                    # our network predicts noise (instead of denoised action)
                                    prediction_type='epsilon'
                                )
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_dim = action_dim
        test_sample = next(iter(dataloader))
        self.val_img = torch.tensor(test_sample["image"]).to(self.device)
        self.val_agent_pos = torch.tensor(test_sample["agent_pos"]).to(self.device)

        self.dataset_stats = dataloader.dataset.stats
        self.dataset_length = len(dataloader)
        self.num_epochs = num_epochs

    def generate_actions(self, image, agent_pos):
        """
        Does non-guided generation
        Args:
            image (B, obs_horizon, 3, 96, 96)
            agent_pos (B, obs_horizon, numerical_obs_dim)
        Output:
            generated_actions (B, pred_horizon, action_dim)
        """
        with torch.no_grad():
            image = image.to(self.device)
            agent_pos = agent_pos.to(self.device)
            n_samples = image.shape[0]

            # Sample actions
            noisy_action = torch.randn(
                (n_samples, self.pred_horizon, self.action_dim), 
                device=self.device)
            visual_emb = self.visual_encoder(image.flatten(end_dim=1)) # (batch_size*obs_horizon, 512)
            visual_features = visual_emb.reshape(*image.shape[:2], -1) # (batch_size, obs_horizon, 512)
            full_cond = torch.cat([visual_features, agent_pos], dim=-1).view(n_samples, -1) # (batch_size, (visual_emb_dim + obs_dim) * obs_horizon)

            for k in tqdm(self.noise_scheduler.timesteps):
                    # predict noise
                    noise_pred = self.noise_predictor(
                        sample=noisy_action,
                        timestep=k,
                        global_cond=full_cond
                    )

                    # inverse diffusion step (remove noise)
                    noisy_action = self.noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=noisy_action
                    ).prev_sample
        return noisy_action # Hopefully not noisy anymore

    def forward(self, image, agent_pos):
        return self.generate_actions(image, agent_pos)
    
    def training_step(self, batch):
        """
        Args:
            batch: 
                image (batch_size, 3, 96, 96)
                agent_pos (batch_size, obs_horizon, obs_dim)
                actions (batch_size, pred_horizon, action_dim)

        Procedure:
            1. Get the visual features
            2. Pass into noise-predictor
            3. Compare predicted with actual noise
            4. Return loss
        """
        image, agent_pos, actions = batch["image"], batch["agent_pos"], batch["action"]
        # (batch_size, obs_horizon, 3, 96, 96), (batch_size, obs_horizon, numerical_obs_dim), (batch_size, pred_horizon, action_dim)
        n_samples = image.shape[0]

        # Prepare observations
        visual_features_flattened = self.visual_encoder(image.flatten(end_dim=1)) # (batch_size*obs_horizon, 512)
        visual_features = visual_features_flattened.reshape(*image.shape[:2], -1) # (batch_size, obs_horizon, 512)
        full_cond = torch.cat([visual_features, agent_pos], dim=-1).view(n_samples, -1) # (batch_size, (visual_emb_dim + numerical_obs_dim) * obs_horizon)

        # Sample noise
        eps = torch.randn_like(actions) # (batch_size, pred_horizon, action_dim)
        n_samples = len(actions)

        # sample ts as torch.long from a uniform distribution
        ts = torch.randint(0, self.num_diff_steps, (n_samples,), dtype=torch.long, device=self.device)

        full_cond = torch.cat([visual_features, agent_pos], dim=-1).view(n_samples, -1) # (batch_size, (visual_emb_dim + obs_dim) * obs_horizon)

        preds = self.noise_predictor(
            self.noise_scheduler.add_noise(actions, eps, ts),
            ts,
            full_cond)
        loss = self.criterion(preds, eps)
        
        # log the loss
        self.log("train_loss", loss, prog_bar=True)
        
        return loss
        
    def configure_optimizers(self, lr=1e-3):
        #return torch.optim.Adam(self.parameters(), lr=lr)
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": get_scheduler(
                                name='cosine',
                                optimizer=optimizer,
                                num_warmup_steps=500,
                                num_training_steps=self.dataset_length * self.num_epochs
                            )
            },
        }
    
    def unnormalize_data(self, data, key):
        min = self.dataset_stats[key]["min"]
        max = self.dataset_stats[key]["max"]
        return 0.5 * (data + 1) * (max - min) + min

    
    def on_train_epoch_end(self, show_plot=False):
        # Visualize the prediction on the test sample
        action_pred = self.generate_actions(self.val_img, self.val_agent_pos).cpu()

        # Convert to numpy
        image = self.val_img[0][0].numpy() # most recent image
        # Convert to int
        image = (image).astype(np.uint8)

        action = action_pred[0].numpy()
        agent_pos = self.val_agent_pos[0].numpy()

        # scale the action and agent_pos and shift them to be positive
        action = (action + 1) * 0.5 * image.shape[-1]
        agent_pos = (agent_pos + 1) * 0.5 * image.shape[-1]
        
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.imshow(image.transpose(1, 2, 0))
        ax.scatter(agent_pos[:, 0], agent_pos[:, 1], c="r", s=100)
        plt.scatter(action[:, 0], action[:, 1], c=np.arange(action.shape[0]), cmap='Blues')
        ax.set_title("Diffusion Prediction")
        ax.set_xlim(0, image.shape[-1])
        ax.set_ylim(0, image.shape[-1])
        plt.tight_layout()

        if show_plot:
            plt.show()
        else:
            # convert plot to image
            temp_file_name = "temp_plot.png"
            plt.savefig(temp_file_name)
            plt.close()
            self.logger.experiment.log({
                "diffusion_prediction": [wandb.Image(temp_file_name)]
            })