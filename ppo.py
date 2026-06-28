"""
Stage 2 — PPO trainer from scratch.
 
Ties env.py + networks.py into a learning loop. Three pieces:
 
  1. RolloutBuffer  : stores N steps of experience the policy generated.
  2. compute_gae    : turns rewards + values into ADVANTAGES (how much better
                      than expected each action was).
  3. update         : the clipped PPO loss that nudges the actor toward
                      positive-advantage actions, plus a value loss for the critic.
 
The loop repeats: collect a rollout -> compute advantages -> update -> repeat.

Bootstrap in RL means estimate a value using another estimate, instead of waiting for the real, fully-observed answers.

Bootstrap: take the one real reward you just observed, then for everything after that, plug the critic's guess: V(s').
           You don't wait for the future; you trust your current estimate of it. 
 
Two details that matter and connect back to the env design:
 
  TERMINAL vs TIMEOUT. The env reports `fell` and `timeout` separately. They are
  handled differently in GAE: a FALL is a true terminal (no future, so don't
  bootstrap V of the next state), but a TIMEOUT is just the episode clock running
  out (the robot was fine, so DO bootstrap). Conflating them teaches the agent
  that surviving to the time limit is as bad as falling — a classic silent bug.
 
  OBSERVATION NORMALIZATION. A 97-dim humanoid obs has wildly different scales
  (height ~0.1 vs joint velocities ~10). Feeding that raw makes learning crawl,
  so we keep a running mean/std and normalize. This is the `RunningMeanStd` you
  saw referenced in the ProtoMotions agents.
"""
 
import numpy as np
import torch
import torch.nn as nn
from networks import Actor, Critic
 
 
class RunningMeanStd:
    """Online mean/variance (Welford) for normalizing observations."""
    def __init__(self, shape, eps=1e-4):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = eps
    
    # It keeps a running estimate and merges each new batch into it
    def update(self, x):                       # x: [N, dim]
        bmean, bvar, bcount = x.mean(0), x.var(0), x.shape[0]
        delta = bmean - self.mean
        tot = self.count + bcount
        # shift mean toward the batch, weighted by batch size
        self.mean += delta * bcount / tot
        m_a, m_b = self.var * self.count, bvar * bcount
        self.var = (m_a + m_b + delta**2 * self.count * bcount / tot) / tot
        self.count = tot
 
    def normalize(self, x):
        return ((x - self.mean) / np.sqrt(self.var + 1e-8)).astype(np.float32)
 
# Storage plus the advantage computation
# fixed size scratchpad (pre-allocated numpy arrays of length horizon=2048)
class RolloutBuffer:
    def __init__(self, size, obs_dim, act_dim):
        # fixed length of buffer: how many transitions it can hold 
        # with the number of horizon it's the number of environment steps you collect per rollout before each update. 
        self.size = size 
        self.obs        = np.zeros((size, obs_dim), np.float32)
        self.actions    = np.zeros((size, act_dim), np.float32)
        self.logprobs   = np.zeros(size, np.float32)
        self.rewards    = np.zeros(size, np.float32)
        self.values     = np.zeros(size, np.float32)
        self.next_values = np.zeros(size, np.float32)
        self.dones      = np.zeros(size, np.float32)   # any episode end
        self.fells      = np.zeros(size, np.float32)   # TRUE terminal only
        self.advantages = np.zeros(size, np.float32)
        self.returns    = np.zeros(size, np.float32)
        self.ptr = 0
 
    def reset(self):
        self.ptr = 0
    
    # add() writes one transition  and bumps the cursor.
    def add(self, obs, action, logp, reward, value, next_value, done, fell):
        i = self.ptr
        self.obs[i], self.actions[i], self.logprobs[i] = obs, action, logp
        self.rewards[i], self.values[i], self.next_values[i] = reward, value, next_value
        self.dones[i], self.fells[i] = done, fell
        self.ptr += 1
    
    """ 
    Turns rewards + critic values into advantages 
    A(s,a) answers "How much better did this action turn out than the critic expected?
    and feeding that into the policy gradient, instead of the raw return, is what makes the learning signal low-noise
    """
    def compute_gae(self, gamma, lam):
        adv = 0.0
        # advantage computation is a recursion that runs backward in time
        # each step's advantage depends on the next step's advantage
        for t in reversed(range(self.size)):
            bootstrap = 1.0 - self.fells[t]    # 0 only 
            continue_ = 1.0 - self.dones[t]    # stop advantage flow on ANY episode end
            delta = self.rewards[t] + gamma * self.next_values[t] * bootstrap - self.values[t]
            # adv on the right hand side is the value computed in the prevoius loop iteration
            # "previous iteration" means "the step that comes later in time"
            adv = delta + gamma * lam * continue_ * adv
            self.advantages[t] = adv
        self.returns[:] = self.advantages + self.values
 
 
class PPO:
    def __init__(self, env, horizon=2048, gamma=0.99, lam=0.95, clip=0.2,
                 lr=3e-4, epochs=10, num_minibatches=8, ent_coef=0.005,
                 vf_coef=0.5, max_grad_norm=1.0, device="cpu", normalize_obs=True):
        self.env = env
        # CPU 
        self.device = device
        # Observation dim = 97: the size of the input vector
        # Everything the robot senses about its current situation each step
        # Action dim = 29; the size of the output vector
        # The command the policy issues
        obs_dim, act_dim = env.obs_dim, env.action_dim
        # actors reads 97 dim observations, outputs 29-dim action
        self.actor = Actor(obs_dim, act_dim).to(device)
        # reads the same 97, outputs a single value 
        self.critic = Critic(obs_dim).to(device)
        # .parameters() returns every learnable tensor in a network (all the Linear weights/bias)
        # concatenate the two network's parameter lists into one, so a single opt.step() updates actor and critic together. 
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        # 2048-slot scratchpad, presized to (horizon, obs_dim) and (horizon, act_dim)
        self.buf = RolloutBuffer(horizon, obs_dim, act_dim)
        # Observation normalizer
        self.obs_rms = RunningMeanStd(obs_dim) if normalize_obs else None
 
        """
        Horizon: step per rollout
        Gamma: reward discount
        Lam: GAEE smoothing
        CLIP: PPO clip range
        EPOCHS, NUM_MINIBATCHES: How many each rollout is reused
        ENT_COEF: Entropy/Exploration Weight
        VF_COEF: Critic-loss weight
        Max_grad_norm: Gradient Clipping ceiling 
        """
        self.horizon, self.gamma, self.lam, self.clip = horizon, gamma, lam, clip
        self.epochs, self.num_minibatches = epochs, num_minibatches
        self.ent_coef, self.vf_coef, self.max_grad_norm = ent_coef, vf_coef, max_grad_norm

        # current observation, the state the currenv is sitting in right now
        self._obs = env.reset()
        # ep_len: steps taken in the current episode so far
        # ep_ret: total award accumulated in the current episode
        self._ep_len, self._ep_ret = 0, 0.0
    
    # Normalizes an observation if the normalizer exists, otherwise just cast to float32 and returns it untouched. 
    def _norm(self, obs):
        return self.obs_rms.normalize(obs) if self.obs_rms else obs.astype(np.float32)
 
    # converts a numpy array into PyTorch tensor on the right device 
    def _t(self, x):
        return torch.as_tensor(x, device=self.device)
    
    # Generating data for each 2048 steps
    @torch.no_grad() # turn off the gradient tracking for the whole method.
    def collect(self):
        """Run the current policy in the env for `horizon` steps; fill the buffer."""
        self.buf.reset()
        ep_lens, ep_rets = [], []
        for _ in range(self.horizon): # just repeat 2048 times
            # the current state (persists across rollouts)
            raw = self._obs
            if self.obs_rms:
                # fold this obs into the running mean/std
                self.obs_rms.update(raw[None])
            # the normalized version the network will see
            obs_n = self._norm(raw)

            # critic scores the state -> V(s)
            value = self.critic(self._t(obs_n)).item()
            # actor samples an action and return its old_logp (the log-prob under the policy right now, which PPO will anchor to later)
            action, logp = self.actor.act(self._t(obs_n))
            # Brings the action back to CPU and into numpy, because env.step speaks numpy.
            a = action.cpu().numpy()

            # get the next state, reward, whether the episode ended, and an info dict
            next_obs, reward, done, info = self.env.step(a)
            # score the next state too - next value is V(s')
            # compute_gae needs for the bootstrap term
            next_value = self.critic(self._t(self._norm(next_obs))).item()

            # done and info["fell"] are stored separately
            self.buf.add(obs_n, a, logp.item(), reward, value, next_value,
                         float(done), float(info["fell"]))
            
            self._ep_len += 1
            self._ep_ret += reward
            # any episode end
            if done:
                ep_lens.append(self._ep_len)
                ep_rets.append(self._ep_ret)
                self._ep_len, self._ep_ret = 0, 0.0
                self._obs = self.env.reset() # reset the counter
            else:
                self._obs = next_obs
        
        # After the buffer is full, run the backward advantage pass, then hand back the finished episode stats for logging.
        self.buf.compute_gae(self.gamma, self.lam)
        return ep_lens, ep_rets
 
    def update(self):
        obs      = self._t(self.buf.obs)
        actions  = self._t(self.buf.actions)
        old_logp = self._t(self.buf.logprobs)
        adv      = self._t(self.buf.advantages)
        ret      = self._t(self.buf.returns)
        # Normalizes the advantages across the batch (zero mean, unit std)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)         # batch-normalize advantages
 
        n = self.horizon # 2048 
        # minibatch size (ensures its never 0 )
        mb = max(1, n // self.num_minibatches) # 2048 // 8 = 256 per minibatch
        # list of all transition indices, it will shuffle to break correlation between consecutive timesteps.
        idx = np.arange(n) # [0,1,2,...,2047]
        pi_l = v_l = ent_l = kl = 0.0 # placeholders to log the last values
        """
        heart of PPO's sample efficiency: 10 epochs x 8 minibatches = 80 gradient steps on the same 2048 transitions

        b is a set of 256 indices 
        obs[b] grabst those 256 rows 
        """
        for _ in range(self.epochs): # reuse the rollout 10 times
            np.random.shuffle(idx)   # new random order 
            for s in range(0, n, mb): # walk in steps of 256: 0, 256, 512, ..
                b = idx[s:s + mb]
                # evaluate rescores the stored actions under the current policy (gradients flowing this time)
                new_logp, entropy = self.actor.evaluate(obs[b], actions[b])
                # π_new(a|s) / π_old(a|s)
                ratio = (new_logp - old_logp[b]).exp()
                # Plain Policy-gradient term
                s1 = ratio * adv[b]
                # same but with the ratio clamped to [0.8, 1.2] since clip=0.2
                s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv[b]
                # torch.min has pessimistic estimate, once an update has moved the probability ~20%
                # further movement stops being rewarded
                # minimzies the loss
                pi_loss = -torch.min(s1, s2).mean()           # clipped surrogate
                # critic is regressed toward the GAE returns (MSE)
                value = self.critic(obs[b])
                v_loss = (value - ret[b]).pow(2).mean()       # critic regression
                # ent is entropy bonus (minus sign = encourage exploration)
                ent = entropy.mean()
                loss = pi_loss + self.vf_coef * v_loss - self.ent_coef * ent

                # clears last step's gradient 
                self.opt.zero_grad()
                # computes new gradients of loss
                loss.backward()
                # rescales every parameters so their combined norm doesn't exceed 1.0 
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm)
                # step() applies the Adam update
                self.opt.step()
                # Record the loss values for logging. 
                # approx_kl: cheap estimate of how far the policy moved this update -- a health check
                # If it ever spikes, my step was too aggressive 
                pi_l, v_l, ent_l = pi_loss.item(), v_loss.item(), ent.item()
                # With torch.no_grad means "This is just measurement, don't build gradient graph for it"
                with torch.no_grad():
                    kl = (old_logp[b] - new_logp).mean().item()
        return {"pi_loss": pi_l, "v_loss": v_l, "entropy": ent_l, "approx_kl": kl}
    
    """
    The outer loop, the whole algorithm at the top level: collect -> update -> repeat, iterations times. 
    Each iteration throws away the old rollout and gathers a new one with the improved policy.
    """
    
    def train(self, iterations, log_every=1):
        for it in range(iterations): # 2048 steps of fresh experience 
            ep_lens, ep_rets = self.collect()
            log = self.update() # 80 gradient steps on it 
            if it % log_every == 0:
                # The average episode length this iteration -- but if no episode finished within the 2048 steps (ep_lens is empty)
                mlen = np.mean(ep_lens) if ep_lens else float(self._ep_len)
                # ep_rets: how much reward each finished episode accumulated in total.
                mret = np.mean(ep_rets) if ep_rets else float("nan")
                # average exploration noise - the number you watch shrink as the policy grows confident
                std = self.actor.log_std.exp().mean().item()
                print(f"iter {it:4d} | ep_len {mlen:6.1f} | ep_ret {mret:8.2f} | "
                      f"pi {log['pi_loss']:+.3f} | v {log['v_loss']:.3f} | "
                      f"ent {log['entropy']:.2f} | std {std:.3f}")
 
    def save(self, path):
        torch.save({"actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict(),
                    "obs_mean": None if not self.obs_rms else self.obs_rms.mean,
                    "obs_var": None if not self.obs_rms else self.obs_rms.var}, path)
 
 
if __name__ == "__main__":
    import sys
    from env import StandEnv
    model = sys.argv[1] if len(sys.argv) > 1 else "mujoco_menagerie/unitree_g1/scene.xml"
    env = StandEnv(model)
    agent = PPO(env, horizon=2048)
    print(f"training PPO on {model}  (obs={env.obs_dim}, act={env.action_dim})")
    agent.train(iterations=200)
    agent.save("ppo_stand.pt")
    print("saved ppo_stand.pt")