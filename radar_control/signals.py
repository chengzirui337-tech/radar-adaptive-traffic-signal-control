"""Shared safety constraints for every controller; durations are exact seconds."""
class SignalMachine:
    def __init__(self, config, states):
        self.config, self.states = config, states
        self.phase = 0
        self.mode = 'green'
        self.elapsed = 0
        self.total_switches = 0

    @property
    def state(self):
        if self.mode == 'green':
            return self.states[self.phase]
        if self.mode == 'yellow':
            return self.states[self.phase].replace('G', 'y')
        return 'r' * len(self.states[0])

    def request(self, switch):
        if self.mode == 'green' and self.elapsed >= self.config.min_green and switch:
            self.mode, self.elapsed = 'yellow', 0
            self.total_switches += 1

    def tick(self):
        self.elapsed += 1
        if self.mode == 'green' and self.elapsed >= self.config.max_green:
            self.request(True)
        elif self.mode == 'yellow' and self.elapsed >= self.config.yellow:
            self.mode, self.elapsed = 'all_red', 0
        elif self.mode == 'all_red' and self.elapsed >= self.config.all_red:
            self.phase = getattr(self, 'pending_phase', (self.phase + 1) % 4)
            if hasattr(self, 'pending_phase'):
                del self.pending_phase
            self.mode, self.elapsed = 'green', 0


class FixedController:
    def action(self, env):
        s = env.signal
        return int(s.mode == 'green' and s.elapsed >= env.config.fixed_greens[s.phase])


class ActuatedController:
    """Extend when a vehicle/queue is approaching the active phase within 40 m."""
    def action(self, env):
        s = env.signal
        if s.mode != 'green' or s.elapsed < env.config.min_green:
            return 0
        active = False
        for row in env.observed:
            axis = 0 if row['approach'] in 'NS' else 2
            phase = axis + (1 if row['lane_id'].endswith('_2') else 0)
            if phase == s.phase and row['distance_m'] <= 40:
                active = True
                break
        return int(not active)
