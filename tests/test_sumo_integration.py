"""Short real-SUMO integration test, distinct from full coursework evaluation."""
from pathlib import Path
import sys
import tempfile
import unittest
from dataclasses import replace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '.packages'))
from radar_control.config import Config
from radar_control.environment import IntersectionEnv
from radar_control.scenario import build_network, generate_demand
from radar_control.signals import FixedController
from radar_control.metrics import summarize
from radar_control.decision_env import DecisionEnv


class SumoIntegrationTest(unittest.TestCase):
    def test_semi_markov_actions_wait_for_safe_decision_times(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            root = Path(d)
            c = replace(Config(stage_seconds=60), decision_seconds=1)
            build_network(root / 'network')
            route = generate_demand(root / 'demand', c, 50001)
            with DecisionEnv(root / 'network', route, root / 'run', c, 50001,
                             backend='libsumo', record=True) as env:
                state = env.initial_state()
                self.assertEqual(state.shape, (344,))
                self.assertEqual((env.time, env.signal.elapsed), (10, 10))
                state, reward, done, duration = env.decision_step(1)
                self.assertEqual(duration, 14)
                self.assertEqual((env.signal.phase, env.signal.mode, env.signal.elapsed), (1, 'green', 10))
                self.assertEqual(env.records[10]['signal_mode'], 'yellow')
                self.assertEqual(env.records[13]['signal_mode'], 'all_red')
                self.assertTrue(np.isfinite(state).all())
                # A hold at the maximum is followed by the same mandatory transition.
                env.signal.elapsed = 59
                _, _, _, duration = env.decision_step(0)
                self.assertEqual(duration, 15)
                self.assertEqual((env.signal.phase, env.signal.elapsed), (2, 10))

    def test_direct_phase_selection_preserves_clearance_and_waiting_guard(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            root = Path(d)
            c = replace(Config(stage_seconds=60), decision_seconds=1)
            build_network(root / 'network')
            route = generate_demand(root / 'demand', c, 50002)
            with DecisionEnv(root / 'network', route, root / 'run', c, 50002,
                             backend='libsumo', record=True) as env:
                env.initial_state()
                _, _, _, duration = env.phase_step(3)
                self.assertEqual(duration, 14)
                self.assertEqual((env.signal.phase, env.signal.elapsed), (3, 10))
                self.assertEqual([r['signal_mode'] for r in env.records[10:14]],
                                 ['yellow', 'yellow', 'yellow', 'all_red'])
                # An observed queue with long wait overrides an unrelated request.
                env.observed.append(dict(approach='N', lane_id='N_in_2', waiting_s=95,
                                         distance_m=5, speed_mps=0, accumulated_waiting_s=95))
                env.phase_step(0)
                self.assertEqual(env.signal.phase, 1)
                self.assertEqual(env.phase_override_count, 1)

    def test_coverage_and_collision_free_simulation(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            root = Path(d)
            c = Config(stage_seconds=30)
            meta = build_network(root / 'network')
            self.assertEqual(len(meta['green_states']), 4)
            self.assertEqual(len(meta['lane_lengths_m']), 12)
            route = generate_demand(root / 'demand', c, 2001)
            saw_vehicle = False
            with IntersectionEnv(root / 'network', route, root / 'run', c, 2001,
                                 backend='libsumo', record=True) as env:
                done = False
                while not done:
                    state, reward, done = env.step(0, FixedController())
                    saw_vehicle |= bool(env.observed)
                    for r in env.observed:
                        self.assertTrue(0 <= r['distance_m'] <= 150)
                    self.assertEqual(state.shape, (c.state_dim,))
                self.assertTrue(saw_vehicle)
                self.assertEqual(env.collisions, 0)
                self.assertEqual(env.teleports, 0)
                self.assertEqual(len(env.records), c.horizon)
            rows = summarize(root / 'run', root / 'demand', c, 'fixed', 2001)
            self.assertEqual(len(rows), 6)
            self.assertTrue((root / 'run' / 'metrics.json').exists())


if __name__ == '__main__':
    unittest.main()
