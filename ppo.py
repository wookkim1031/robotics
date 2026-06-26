"""
Stage 2 — PPO trainer from scratch.
 
Ties env.py + networks.py into a learning loop. Three pieces:
 
  1. RolloutBuffer  : stores N steps of experience the policy generated.
  2. compute_gae    : turns rewards + values into ADVANTAGES (how much better
                      than expected each action was).
  3. update         : the clipped PPO loss that nudges the actor toward
                      positive-advantage actions, plus a value loss for the critic.
 
The loop repeats: collect a rollout -> compute advantages -> update -> repeat.
 
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
 
    def update(self, x):                       # x: [N, dim]
        bmean, bvar, bcount = x.mean(0), x.var(0), x.shape[0]
        delta = bmean - self.mean
        tot = self.count + bcount
        self.mean += delta * bcount / tot
        m_a, m_b = self.var * self.count, bvar * bcount
        self.var = (m_a + m_b + delta**2 * self.count * bcount / tot) / tot
        self.count = tot
 
    def normalize(self, x):
        return ((x - self.mean) / np.sqrt(self.var + 1e-8)).astype(np.float32)
 
 
class RolloutBuffer:
    def __init__(self, size, obs_dim, act_dim):
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
 
    def add(self, obs, action, logp, reward, value, next_value, done, fell):
        i = self.ptr
        self.obs[i], self.actions[i], self.logprobs[i] = obs, action, logp
        self.rewards[i], self.values[i], self.next_values[i] = reward, value, next_value
        self.dones[i], self.fells[i] = done, fell
        self.ptr += 1
 
    def compute_gae(self, gamma, lam):
        adv = 0.0
        for t in reversed(range(self.size)):
            bootstrap = 1.0 - self.fells[t]    # zero only on a true terminal (fell)
            continue_ = 1.0 - self.dones[t]    # stop advantage flow on ANY episode end
            delta = self.rewards[t] + gamma * self.next_values[t] * bootstrap - self.values[t]
            adv = delta + gamma * lam * continue_ * adv
            self.advantages[t] = adv
        self.returns[:] = self.advantages + self.values
 
 
class PPO:
    def __init__(self, env, horizon=2048, gamma=0.99, lam=0.95, clip=0.2,
                 lr=3e-4, epochs=10, num_minibatches=8, ent_coef=0.005,
                 vf_coef=0.5, max_grad_norm=1.0, device="cpu", normalize_obs=True):
        self.env = env
        self.device = device
        obs_dim, act_dim = env.obs_dim, env.action_dim
        self.actor = Actor(obs_dim, act_dim).to(device)
        self.critic = Critic(obs_dim).to(device)
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.buf = RolloutBuffer(horizon, obs_dim, act_dim)
        self.obs_rms = RunningMeanStd(obs_dim) if normalize_obs else None
 
        self.horizon, self.gamma, self.lam, self.clip = horizon, gamma, lam, clip
        self.epochs, self.num_minibatches = epochs, num_minibatches
        self.ent_coef, self.vf_coef, self.max_grad_norm = ent_coef, vf_coef, max_grad_norm
 
        self._obs = env.reset()
        self._ep_len, self._ep_ret = 0, 0.0
 
    def _norm(self, obs):
        return self.obs_rms.normalize(obs) if self.obs_rms else obs.astype(np.float32)
 
    def _t(self, x):
        return torch.as_tensor(x, device=self.device)
 
    @torch.no_grad()
    def collect(self):
        """Run the current policy in the env for `horizon` steps; fill the buffer."""
        self.buf.reset()
        ep_lens, ep_rets = [], []
        for _ in range(self.horizon):
            raw = self._obs
            if self.obs_rms:
                self.obs_rms.update(raw[None])
            obs_n = self._norm(raw)
 
            value = self.critic(self._t(obs_n)).item()
            action, logp = self.actor.act(self._t(obs_n))
            a = action.cpu().numpy()
 
            next_obs, reward, done, info = self.env.step(a)
            next_value = self.critic(self._t(self._norm(next_obs))).item()
 
            self.buf.add(obs_n, a, logp.item(), reward, value, next_value,
                         float(done), float(info["fell"]))
            self._ep_len += 1
            self._ep_ret += reward
            if done:
                ep_lens.append(self._ep_len)
                ep_rets.append(self._ep_ret)
                self._ep_len, self._ep_ret = 0, 0.0
                self._obs = self.env.reset()
            else:
                self._obs = next_obs
 
        self.buf.compute_gae(self.gamma, self.lam)
        return ep_lens, ep_rets
 
    def update(self):
        obs      = self._t(self.buf.obs)
        actions  = self._t(self.buf.actions)
        old_logp = self._t(self.buf.logprobs)
        adv      = self._t(self.buf.advantages)
        ret      = self._t(self.buf.returns)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)         # batch-normalize advantages
 
        n = self.horizon
        mb = max(1, n // self.num_minibatches)
        idx = np.arange(n)
        pi_l = v_l = ent_l = kl = 0.0
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for s in range(0, n, mb):
                b = idx[s:s + mb]
                new_logp, entropy = self.actor.evaluate(obs[b], actions[b])
                ratio = (new_logp - old_logp[b]).exp()
                s1 = ratio * adv[b]
                s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv[b]
                pi_loss = -torch.min(s1, s2).mean()           # clipped surrogate
                value = self.critic(obs[b])
                v_loss = (value - ret[b]).pow(2).mean()       # critic regression
                ent = entropy.mean()
                loss = pi_loss + self.vf_coef * v_loss - self.ent_coef * ent
 
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm)
                self.opt.step()
                pi_l, v_l, ent_l = pi_loss.item(), v_loss.item(), ent.item()
                with torch.no_grad():
                    kl = (old_logp[b] - new_logp).mean().item()
        return {"pi_loss": pi_l, "v_loss": v_l, "entropy": ent_l, "approx_kl": kl}
 
    def train(self, iterations, log_every=1):
        for it in range(iterations):
            ep_lens, ep_rets = self.collect()
            log = self.update()
            if it % log_every == 0:
                mlen = np.mean(ep_lens) if ep_lens else float(self._ep_len)
                mret = np.mean(ep_rets) if ep_rets else float("nan")
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