#!/usr/bin/env python3
"""
Double DQN Agent with Prioritized Experience Replay — v2
=========================================================
EXTENDED for 3-Layer Action Space:
  Layer 1 (Queue Control) : Actions 0-4  ->  -2, -1, 0, +1, +2 packets
  Layer 2 (Traffic Eng.)  : Action  5   ->  Reroute elephant flow
  Layer 3 (Monitoring)    : Action  6   ->  Sample elephant headers (safe)
  Layer 3 (Load Balance)  : Action  7   ->  ECMP: move second-largest flow

State Vector (10 features):
  [queue_length, throughput, drop_rate, packet_rate,
   queue_velocity, ewma_queue,
   max_flow_rate, flow_count, elephant_ratio,
   reroute_count_last_10s]                           <- Feature 10 (NEW)

Changes vs first draft:
  - state_size = 10  (was 9)
  - Feature 10: reroute_count_last_10s
    Gives the agent temporal awareness of its own recent interventions.
    Without it the agent cannot learn that spamming Action 5 yields
    diminishing returns — it has no memory of what it just did.
    Normalised by a max of 5 reroutes per 10 s window (clipped at 1.0).
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ── PER hyperparameters ──────────────────────────────────────────────────────
PER_ALPHA      = 0.6
PER_BETA_START = 0.4
PER_BETA_INC   = 5e-5
PER_EPS        = 1e-6

# ── Action index constants ───────────────────────────────────────────────────
# Import these in the controller to avoid magic numbers everywhere.
ACTION_QUEUE_MINUS2 = 0
ACTION_QUEUE_MINUS1 = 1
ACTION_QUEUE_HOLD   = 2
ACTION_QUEUE_PLUS1  = 3
ACTION_QUEUE_PLUS2  = 4
ACTION_REROUTE      = 5
ACTION_SAMPLE       = 6
ACTION_ECMP_SPLIT   = 7

QUEUE_ACTIONS = {
    ACTION_QUEUE_MINUS2, ACTION_QUEUE_MINUS1,
    ACTION_QUEUE_HOLD,
    ACTION_QUEUE_PLUS1,  ACTION_QUEUE_PLUS2,
}
FLOW_ACTIONS = {ACTION_REROUTE, ACTION_SAMPLE, ACTION_ECMP_SPLIT}

# ── State feature count ──────────────────────────────────────────────────────
STATE_SIZE  = 10
ACTION_SIZE = 8


# ────────────────────────────────────────────────────────────────────────────
# Neural network
# ────────────────────────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        return self.net(x)


# ────────────────────────────────────────────────────────────────────────────
# Prioritized experience replay buffer
# ────────────────────────────────────────────────────────────────────────────
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=PER_ALPHA):
        self.capacity   = capacity
        self.alpha      = alpha
        self.buffer     = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos        = 0

    def __len__(self):
        return len(self.buffer)

    def add(self, transition):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta):
        prios  = (self.priorities[:len(self.buffer)]
                  if len(self.buffer) < self.capacity
                  else self.priorities)
        probs  = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        total   = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, torch.FloatTensor(weights).unsqueeze(1)

    def update_priorities(self, indices, td_errors):
        for i, err in zip(indices, td_errors):
            self.priorities[i] = abs(err) + PER_EPS


# ────────────────────────────────────────────────────────────────────────────
# Agent
# ────────────────────────────────────────────────────────────────────────────
class DoubleDQNAgent:
    """
    DDQN + PER with 8-action space and action masking.

    Typical usage in the controller
    --------------------------------
        mask   = agent.build_action_mask(queue_util, elephant_exists, ecmp_capable)
        action = agent.choose_action(state_tensor, mask)
    """

    def __init__(
        self,
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        lr=3e-4,
        gamma=0.98,
        buffer_size=10000,
        batch_size=128,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9999,
        tau=0.005,
    ):
        self.state_size   = state_size
        self.action_size  = action_size
        self.gamma        = gamma
        self.batch_size   = batch_size
        self.tau          = tau

        self.q_main   = QNetwork(state_size, action_size)
        self.q_target = QNetwork(state_size, action_size)
        self.q_target.load_state_dict(self.q_main.state_dict())
        self.q_target.eval()

        self.optimizer = optim.Adam(self.q_main.parameters(), lr=lr)
        self.memory    = PrioritizedReplayBuffer(buffer_size)

        self.beta          = PER_BETA_START
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.train_interval = 2
        self.step_counter   = 0

    # ── State tensor ─────────────────────────────────────────────────────────
    def get_state_tensor(self, stats: dict) -> torch.FloatTensor:
        """
        Normalise the 10-feature stats dict into a FloatTensor.

        Expected keys
        -------------
        queue_length            packets in backlog              (raw count)
        throughput              port throughput                 (kbps)
        drop_rate               packets dropped per second      (pps)
        packet_rate             packets sent per second         (pps)
        queue_velocity          change in backlog vs prev step  (raw delta)
        ewma_queue              EWMA of queue_length            (raw count)
        max_flow_rate           highest per-flow byte rate      (bytes/s)
        flow_count              number of active flows          (count)
        elephant_ratio          top-flow / total port rate      (0-1)
        reroute_count_last_10s  reroutes fired in last 10 s     (count)  <- NEW

        Why reroute_count_last_10s matters
        -----------------------------------
        Without it the agent has no memory of its recent interventions.
        It can observe that Action 5 gives +0.5 reward but it cannot learn
        that repeating it three times in ten seconds causes oscillation and
        yields diminishing returns.  This feature closes that credit
        assignment gap by encoding the agent's own recent behaviour into
        the state.  Max of 5 reroutes per window is a reasonable ceiling
        for a k=4 fat-tree with 4-second decision periods.
        """
        return torch.FloatTensor([
            # ── Original 6 features (unchanged) ──────────────────────────
            stats.get('queue_length',   0.0) / 200.0,
            stats.get('throughput',     0.0) / 100_000.0,
            stats.get('drop_rate',      0.0) / 100.0,
            stats.get('packet_rate',    0.0) / 10_000.0,
            stats.get('queue_velocity', 0.0) / 20.0,
            stats.get('ewma_queue',     0.0) / 200.0,
            # ── Flow cache features (added in previous version) ───────────
            stats.get('max_flow_rate',  0.0) / 125_000.0,
            min(stats.get('flow_count', 0.0) / 50.0, 1.0),
            min(max(stats.get('elephant_ratio', 0.0), 0.0), 1.0),
            # ── Feature 10: reroute intensity (NEW) ──────────────────────
            min(stats.get('reroute_count_last_10s', 0.0) / 5.0, 1.0),
        ])

    # ── Action masking ────────────────────────────────────────────────────────
    def build_action_mask(
        self,
        queue_util: float,
        elephant_exists: bool,
        ecmp_capable: bool,
        reroute_count: int = 0,
    ) -> torch.BoolTensor:
        """
        Return a boolean mask (True = action is selectable).

        Gating rules
        ------------
        Actions 5, 6, 7 require BOTH:
            queue_util > 0.5   port is under meaningful load
            elephant_exists    a qualifying elephant flow is present

        Action 7 additionally requires ecmp_capable.

        Hard reroute cap
        ----------------
        If reroute_count >= 3 within the current 10-second window,
        Actions 5 and 7 are also blocked.  This prevents the agent from
        oscillating even when the Q-network would otherwise choose them.
        The cap works in tandem with the global cooldown table in the
        controller — the cooldown prevents cross-switch oscillation,
        the cap prevents per-switch oscillation.
        """
        mask = [True] * self.action_size

        congested_with_elephant = (queue_util > 0.5) and elephant_exists
        if not congested_with_elephant:
            mask[ACTION_REROUTE]    = False
            mask[ACTION_SAMPLE]     = False
            mask[ACTION_ECMP_SPLIT] = False

        if not ecmp_capable:
            mask[ACTION_ECMP_SPLIT] = False

        # Hard cap: too many recent reroutes → block further ones
        if reroute_count >= 3:
            mask[ACTION_REROUTE]    = False
            mask[ACTION_ECMP_SPLIT] = False

        return torch.BoolTensor(mask)

    # ── Action selection ──────────────────────────────────────────────────────
    def choose_action(
        self,
        state: torch.FloatTensor,
        mask: torch.BoolTensor = None,
    ) -> int:
        """
        Epsilon-greedy selection with action masking.

        The mask is respected during random exploration too, so the agent
        never wastes an exploration step on an illegal action.
        Invalid actions are set to -inf before argmax during greedy
        selection so they can never win regardless of Q-value.
        """
        if random.random() < self.epsilon:
            if mask is not None:
                valid = [i for i, m in enumerate(mask) if m]
                return random.choice(valid) if valid else ACTION_QUEUE_HOLD
            return random.randint(0, self.action_size - 1)

        with torch.no_grad():
            q_vals = self.q_main(state.unsqueeze(0)).squeeze(0)
            if mask is not None:
                q_vals = q_vals.masked_fill(~mask, float('-inf'))
            return int(q_vals.argmax().item())

    # ── Experience storage ────────────────────────────────────────────────────
    def store(self, state, action, reward, next_state):
        self.memory.add((state, action, reward, next_state))

    # ── Training step ─────────────────────────────────────────────────────────
    def train_step(self):
        """
        One Double-DQN update with importance-sampling weights from PER.
        Returns mean absolute TD error for diagnostics, or None if buffer
        is not yet large enough.
        """
        if len(self.memory) < self.batch_size:
            return None

        samples, indices, weights = self.memory.sample(self.batch_size, self.beta)
        self.beta = min(1.0, self.beta + PER_BETA_INC)

        states, actions, rewards, next_states = zip(*samples)
        states      = torch.stack(states)
        next_states = torch.stack(next_states)
        actions     = torch.LongTensor(actions).unsqueeze(1)
        rewards     = torch.FloatTensor(rewards).unsqueeze(1)

        q_values = self.q_main(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.q_main(next_states).argmax(1, keepdim=True)
            next_q       = self.q_target(next_states).gather(1, next_actions)
            targets      = rewards + self.gamma * next_q

        td_errors = (q_values - targets).detach().cpu().numpy().squeeze()
        loss      = (weights * (q_values - targets) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_main.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.memory.update_priorities(indices, td_errors)

        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        self._soft_update_target()
        return float(np.mean(np.abs(td_errors)))

    def _soft_update_target(self):
        with torch.no_grad():
            for t, s in zip(self.q_target.parameters(), self.q_main.parameters()):
                t.data.copy_(self.tau * s.data + (1.0 - self.tau) * t.data)

    # ── Persistence ───────────────────────────────────────────────────────────
    def save_model(self, path: str):
        torch.save(
            {
                'q_main_state_dict':   self.q_main.state_dict(),
                'q_target_state_dict': self.q_target.state_dict(),
                'epsilon':             self.epsilon,
                'step_counter':        self.step_counter,
            },
            path,
        )

    def load_trained_model(self, path: str):
        ckpt = torch.load(path, map_location='cpu')
        self.q_main.load_state_dict(ckpt['q_main_state_dict'])
        self.q_target.load_state_dict(ckpt['q_target_state_dict'])
        self.epsilon      = ckpt.get('epsilon', self.epsilon_end)
        self.step_counter = ckpt.get('step_counter', 0)
        self.q_main.eval()
        self.q_target.eval()
        print(f"Model loaded from {path}")
