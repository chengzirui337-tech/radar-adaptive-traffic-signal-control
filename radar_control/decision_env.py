"""One-second green decisions; forced transition seconds have no learned actions."""
import numpy as np

from .environment import IntersectionEnv


class DecisionEnv(IntersectionEnv):
    feature_dimension = 344

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.external_transitions = True
        self.transition_durations = []

    def _sense(self):
        super()._sense()
        # A per-second cost, aligned with delay rather than repeated past waiting.
        loss = sum(max(0., 1 - r['speed_mps'] / 13.89) for r in self.true_radar)
        wait = sum(r['accumulated_waiting_s'] for r in self.true_radar)
        # Reward is integrated over simulated seconds by decision_step.
        cost = loss + .001 * wait
        self.latest_delay_cost = cost
        if self.record and self.records and self.time <= self.config.horizon:
            self.records[-1]['reward'] = -cost / 10
        return -cost / 10

    def features(self):
        per_phase = np.zeros((4, 8), dtype=np.float32)
        per_phase[:, 4] = 1
        for r in self.observed:
            p = (0 if r['approach'] in 'NS' else 2) + int(r['lane_id'].endswith('_2'))
            v, d = r['speed_mps'], r['distance_m']
            eta = d / max(v, .1)
            per_phase[p, 0] += (v <= self.config.stop_speed) / 10
            per_phase[p, 1] += 1 / 20
            per_phase[p, 2] += (d <= 40) / 10
            per_phase[p, 3] += r['accumulated_waiting_s'] / 600
            per_phase[p, 4] = min(per_phase[p, 4], eta / 15)
            for j, horizon in enumerate((3, 6, 10), 5):
                per_phase[p, j] += (eta <= horizon) / 5
        current = per_phase[self.signal.phase]
        nxt = per_phase[(self.signal.phase + 1) % 4]
        return np.concatenate((self.state(), per_phase.ravel(), current, nxt)).astype(np.float32)

    def ready(self):
        return self.signal.mode == 'green' and self.signal.elapsed >= self.config.min_green

    def initial_state(self):
        self.step(0, step_seconds=self.config.min_green)
        return self.features()

    def decision_step(self, action, end_time=None):
        end_time = self.config.horizon if end_time is None else end_time
        before_time = self.time
        before = self.features() if self.record and self.time < self.config.horizon else None
        total_reward = 0.
        # Switching entails yellow + all-red + mandatory green before another action.
        duration = (self.config.yellow + self.config.all_red + self.config.min_green) if action else 1
        start = self.time
        _, reward, done = self.step(action, end_time=end_time, step_seconds=duration)
        total_reward += reward * (self.time - start) / 5
        if not self.ready() and not done:
            # The maximum-green guard may force a switch following a hold request.
            start = self.time
            forced = self.config.yellow + self.config.all_red + self.config.min_green
            _, reward, done = self.step(0, end_time=end_time, step_seconds=forced)
            total_reward += reward * (self.time - start) / 5
        after, duration = self.features(), self.time - before_time
        if before is not None:
            self.transitions.append((before, int(action), total_reward, after, done))
            self.transition_durations.append(duration)
        return after, total_reward, done, duration

    def phase_step(self, requested_phase, end_time=None):
        if requested_phase not in range(4):
            raise ValueError('Phase action must be 0..3')
        original_request = requested_phase
        # Direct phase selection must not starve a low-demand movement.
        # The shield uses only the same observed trajectory waiting times.
        waiting = np.zeros(4)
        for row in self.observed:
            p = (0 if row['approach'] in 'NS' else 2) + int(row['lane_id'].endswith('_2'))
            waiting[p] = max(waiting[p], row['waiting_s'])
        waiting[self.signal.phase] = 0
        if waiting.max() >= 90:
            requested_phase = int(waiting.argmax())
            if requested_phase != original_request:
                self.phase_override_count = getattr(self, 'phase_override_count', 0) + 1
        switching = requested_phase != self.signal.phase
        if switching:
            self.signal.pending_phase = int(requested_phase)
        result = self.decision_step(int(switching), end_time)
        if self.record and self.transitions and self.time <= self.config.horizon:
            state, _, reward, nxt, done = self.transitions[-1]
            self.transitions[-1] = (state, int(original_request), reward, nxt, done)
        return result
