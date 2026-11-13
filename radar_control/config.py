from dataclasses import asdict, dataclass

APPROACHES = ('N', 'S', 'E', 'W')
STAGES = ('low_1', 'normal_1', 'peak', 'normal_2', 'low_2')
DEMANDS = (600, 1000, 1500, 1000, 600)
TURNS = ((.15, .70, .15), (.20, .65, .15), (.25, .60, .15),
         (.20, .65, .15), (.15, .70, .15))  # left, straight, right
PHASES = ('NS_through_right', 'NS_left', 'EW_through_right', 'EW_left')


@dataclass
class Config:
    stage_seconds: int = 900
    decision_seconds: int = 5
    detection_m: float = 150.0
    bin_m: float = 15.0
    min_green: int = 10
    max_green: int = 60
    yellow: int = 3
    all_red: int = 1
    fixed_greens: tuple = (22, 10, 34, 14)
    demand_update_seconds: int = 60
    fluctuation: float = .15
    turning_jitter: float = .02
    stop_speed: float = .1
    position_noise_m: float = 0.0
    speed_noise_mps: float = 0.0
    reward_wait: float = .01
    reward_queue: float = 1.0
    reward_stop: float = 2.0
    gamma: float = .99
    learning_rate: float = .0003
    batch_size: int = 64
    replay_size: int = 50000
    warmup_steps: int = 1000
    target_update_steps: int = 250
    train_every: int = 2
    drain_seconds: int = 600

    @property
    def horizon(self):
        return 5 * self.stage_seconds

    @property
    def n_bins(self):
        return int(self.detection_m / self.bin_m)

    @property
    def state_dim(self):
        return 12 * self.n_bins * 2 + 12 * 4 + 4 + 3 + 1

    def to_dict(self):
        return asdict(self)
