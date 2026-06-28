"""
Stage 2 — PPO networks (Actor + Critic).

PPO is an actor-critic method, so there are two networks:

  ACTOR (the policy)  : observation -> a DISTRIBUTION over actions.
      It doesn't output one action; it outputs a Gaussian (mean + spread) that
      we sample from. The randomness IS the exploration — and PPO needs a
      probability distribution so it can compute "how much more/less likely is
      this action now vs. before" (the policy ratio at the heart of PPO).

  CRITIC (the value function) : observation -> one number, V(s).
      An estimate of "how much total reward do I expect from this state onward."
      It's not used to pick actions; it's used to judge whether an action did
      better or worse than expected (the advantage), which sharply reduces the
      noise in the policy-gradient signal.

This mirrors the `model` carried by the PPO/AMP agents in ProtoMotions: a policy
head and a value head, here kept as two separate MLPs for clarity.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def build_mlp(in_dim, out_dim, hidden=(256, 256), out_gain=1.0):
    """A plain feed-forward net with ELU activations and orthogonal init.

    Orthogonal init with a SMALL gain on the final layer is a standard PPO
    stabilizer: it makes the policy start near-deterministic and the value head
    start near zero, so early training doesn't explode.

    MLP is a stack of fully connected Linear layers with a nonlinearity between them. 
    obs -> Linear -> ELU -> Linear -> ELU -> Linear -> output

    Final layer has no activation, because the output is an action mean that needs to be any real number 

    Orthogonal initializes each weight matrix so its columns are orthonormal (scaled by a gain). 
    This keeps the variance of signals and gradients stable as they pass through layers at the start of the training

    Actor - "out gain of 0.1, why so small?" -> it's actually 0.01, and that's the point. 

    
    """
    sizes = [in_dim, *hidden, out_dim]
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ELU())
    net = nn.Sequential(*layers)
    # init
    last = None
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.zeros_(m.bias)
            last = m
    nn.init.orthogonal_(last.weight, gain=out_gain)   # small gain on output
    return net

# nn.Module means Actor inherits PyTorch's base Module.
class Actor(nn.Module):
    """
    Gaussian policy: obs -> Normal(mean(obs), std).

    Actor is the policy - it decides what to do 
    """

    def __init__(self, obs_dim, act_dim, hidden=(256, 256), init_std=0.6):
        # what lets PyTorch automatically find the network's parameters.
        super().__init__()
        # mean_net (an MLP from obs_dim -> act_dim, giving the mean action for a given observation)
        self.mean_net = build_mlp(obs_dim, act_dim, hidden, out_gain=0.01)
        # log_std is a learnable parameter, NOT a function of the observation.
        # State-independent exploration noise that the optimizer shrinks as the
        # policy grows confident. Stored as log so std = exp(log_std) stays > 0.

        # the global spread you actually understand
        self.log_std = nn.Parameter(torch.full((act_dim,), float(np.log(init_std))))

    # Gaussian Policy: for continuous control you can't output discrete action probabilities
    # so the policy is a normal distribution over actions - Normal (mean, std)
    def distribution(self, obs):
        mean = self.mean_net(obs)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std) # Normal is a diagonal Gaussian - 29 independent 1-D Gaussians, one per joint. 

    # acting in the world during rollout collection. 
    # we are only generating data to step the env, not training yet
    @torch.no_grad()
    def act(self, obs):
        """Sample an action for rollout collection. Returns action + its log-prob."""
        # builds the distribution, then sample() draws a random action from it 
        dist = self.distribution(obs)
        # draws a random action from it - the randomness of sampling of its exploration. 
        action = dist.sample()
        # the log probability of the action it just chose, which PPO will need later
        log_prob = dist.log_prob(action).sum(-1)     # sum over action  dims
        return action, log_prob

    # learning counterpart: takes action as an input 
    def evaluate(self, obs, action):
        """Used in the UPDATE: log-prob of stored actions under the CURRENT policy,
        plus entropy (an exploration bonus)."""
        dist = self.distribution(obs)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy


class Critic(nn.Module):
    """
    Critic is the value function - it judges how good a situation is. 
    Value function: obs -> V(s), a single scalar per state.
    """

    def __init__(self, obs_dim, hidden=(256, 256)):
        super().__init__()
        # Out gain can be one, because the value estimate can be any value. 
        self.v_net = build_mlp(obs_dim, 1, hidden, out_gain=1.0) # one MLP from obs_dim

    def forward(self, obs):
        # V(s) is a single number
        # How much total discounted reward do I expect from this state onward.
        return self.v_net(obs).squeeze(-1)            # drop the trailing dim


if __name__ == "__main__":
    obs_dim, act_dim = 97, 29
    actor, critic = Actor(obs_dim, act_dim), Critic(obs_dim)
    obs = torch.randn(8, obs_dim)                     # a batch of 8 observations

    a, logp = actor.act(obs)
    v = critic(obs)
    logp2, ent = actor.evaluate(obs, a)
    print("action       :", tuple(a.shape), "(batch, act_dim)")
    print("log_prob     :", tuple(logp.shape), "(one per sample)")
    print("value        :", tuple(v.shape), "(one per sample)")
    print("entropy      :", tuple(ent.shape))
    print("init std     :", actor.log_std.exp().mean().item())
    print("params actor/critic:", sum(p.numel() for p in actor.parameters()),
          "/", sum(p.numel() for p in critic.parameters()))