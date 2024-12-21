import torch
import torch.nn as nn

from trainer import Trainer

from magent2.environments import battle_v4

env = battle_v4.env(map_size=45, render_mode=None)

trainer = Trainer(
    env = env, 
    input_shape=env.observation_space("red_0").shape, 
    action_shape=env.action_space("red_0").n
)
trainer.train()
print("Training done!")

torch.save(trainer.q_network.state_dict(), "blue.pt")
print("Saved model!")