## The environment
The authors use a pygame-environment where we can push a t-shaped figure around. 
**Action space:** 
- Desired cursor (agent position): (2,) floats
**Observation space:** 
- Image of the scene: (3, 96, 96) floats
- agent position: (2,) floats

For the evaluation we just want to:
1. Initialize the environment
2. Act out the model until completion
3. Visualize the behaviour

### Environment dynamics
The environment simulation frequency is higher than the control frequency. We give a position into the environment, and a PD-controller moves the "cursor" towards the desired position:
```python
acceleration = self.k_p * (action - self.agent.position) + self.k_v * (Vec2d(0, 0) - self.agent.velocity)
```
