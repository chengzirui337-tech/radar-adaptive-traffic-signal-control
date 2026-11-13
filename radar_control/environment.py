"""TraCI-compatible environment. Observation is strictly limited to radar range."""
import csv
import gzip
import json
import os
from pathlib import Path

import numpy as np

from .config import APPROACHES, STAGES
from .scenario import binary
from .signals import SignalMachine


class IntersectionEnv:
    def __init__(self, network, routes, output, config, seed, observation='full',
                 backend='auto', record=False, export_radar=False, gui=False):
        self.network, self.routes = Path(network), Path(routes)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.config, self.seed, self.observation = config, int(seed), observation
        self.record = record
        self.meta = json.loads((self.network / 'network.json').read_text(encoding='utf-8'))
        self.lanes = [f'{a}_in_{i}' for a in APPROACHES for i in range(3)]
        self.lane_index = {lane: i for i, lane in enumerate(self.lanes)}
        self.lengths = self.meta['lane_lengths_m']
        self.signal = SignalMachine(config, self.meta['green_states'])
        self.rng = np.random.default_rng(seed + 900000)
        self.time = 0
        self.observed = []
        self.true_radar = []
        self.previous_incoming = set()
        self.crossed = 0
        self.collisions = self.teleports = 0
        self.records = []
        self.transitions = []
        self.radar_file = None
        self.closed = False
        # Resolve SUMO_HOME before importing libsumo so data paths are available.
        executable = binary('sumo-gui' if gui else 'sumo')
        if backend in ('auto', 'libsumo') and not gui:
            try:
                import libsumo as api
                self.backend = 'libsumo'
            except ImportError:
                if backend == 'libsumo':
                    raise
                import traci as api
                self.backend = 'traci'
        else:
            import traci as api
            self.backend = 'traci'
        self.api = api
        import traci.constants as tc
        self.tc = tc
        self.variables = (tc.VAR_LANE_ID, tc.VAR_LANEPOSITION, tc.VAR_SPEED,
                          tc.VAR_ACCELERATION, tc.VAR_WAITING_TIME, tc.VAR_ACCUMULATED_WAITING_TIME)
        relative = lambda path: Path(os.path.relpath(path)).as_posix()
        command = [executable, '-n', relative(self.network / 'intersection.net.xml'),
                   '-r', relative(self.routes), '--seed', str(seed), '--step-length', '1',
                   '--no-step-log', 'true', '--no-warnings', 'true',
                   '--xml-validation', 'never',
                   '--time-to-teleport', '-1', '--waiting-time-memory', str(config.horizon + config.drain_seconds + 1),
                   '--tripinfo-output', relative(self.output / 'tripinfo.xml'),
                   '--tripinfo-output.write-unfinished', 'true',
                   '--log', relative(self.output / 'sumo.log')]
        if self.backend == 'libsumo':
            self.api.start(command)
        else:
            import traci
            label = f'env_{id(self)}'
            traci.start(command, label=label, verbose=False)
            self.api = traci.getConnection(label)
        self.api.trafficlight.setRedYellowGreenState('J', self.signal.state)
        if export_radar:
            self.radar_file = gzip.open(self.output / 'radar_trajectories.csv.gz', 'wt', newline='', encoding='utf-8')
            fields = ('time_s', 'vehicle_id', 'approach', 'lane_id', 'distance_m', 'speed_mps',
                      'acceleration_mps2', 'waiting_s', 'accumulated_waiting_s', 'phase', 'signal_mode')
            self.radar_writer = csv.DictWriter(self.radar_file, fieldnames=fields)
            self.radar_writer.writeheader()

    def state(self):
        c = self.config
        grid = np.zeros((12, c.n_bins, 2), dtype=np.float32)
        stats = np.zeros((12, 4), dtype=np.float32)
        # Ablation: all observation statistics use only the near-stopline region.
        limit = 30.0 if self.observation == 'near' else c.detection_m
        for row in self.observed:
            if row['distance_m'] > limit:
                continue
            lane = self.lane_index[row['lane_id']]
            cell = min(int(row['distance_m'] / c.bin_m), c.n_bins - 1)
            grid[lane, cell, 0] += 1
            grid[lane, cell, 1] += row['speed_mps']
            stats[lane, 0] += 1
            stats[lane, 1] += row['speed_mps'] <= c.stop_speed
            stats[lane, 2] += row['accumulated_waiting_s']
            stats[lane, 3] += row['speed_mps']
        counts = grid[:, :, 0].copy()
        grid[:, :, 1] /= np.maximum(counts, 1) * 13.89
        grid[:, :, 0] /= 3.0
        stats[:, 3] /= np.maximum(stats[:, 0], 1) * 13.89
        stats[:, :2] /= 20.0
        stats[:, 2] /= 1200.0
        phase = np.eye(4, dtype=np.float32)[self.signal.phase]
        mode = np.eye(3, dtype=np.float32)[('green', 'yellow', 'all_red').index(self.signal.mode)]
        elapsed = [self.signal.elapsed / c.max_green]
        return np.concatenate((grid.ravel(), stats.ravel(), phase, mode, elapsed)).astype(np.float32)

    def _sense(self):
        tc, api, c = self.tc, self.api, self.config
        for vid in api.simulation.getDepartedIDList():
            api.vehicle.subscribe(vid, self.variables)
        results = api.vehicle.getAllSubscriptionResults()
        observed, true_radar, incoming = [], [], set()
        queue, wait, new_stops = 0, 0.0, 0
        max_lane_queue_m = 0.0
        prev_stopped = getattr(self, '_stopped', set())
        stopped = set()
        for vid, data in results.items():
            lane = data.get(tc.VAR_LANE_ID)
            if lane not in self.lengths:
                continue
            incoming.add(vid)
            distance = max(0.0, self.lengths[lane] - data[tc.VAR_LANEPOSITION])
            speed = float(data[tc.VAR_SPEED])
            if speed <= c.stop_speed:
                queue += 1
                stopped.add(vid)
                max_lane_queue_m = max(max_lane_queue_m, distance)
                if vid not in prev_stopped:
                    new_stops += 1
            wait += float(data[tc.VAR_ACCUMULATED_WAITING_TIME])
            if distance > c.detection_m:
                continue
            row = dict(time_s=self.time, vehicle_id=vid, approach=lane[0], lane_id=lane,
                       distance_m=distance, speed_mps=speed,
                       acceleration_mps2=float(data[tc.VAR_ACCELERATION]),
                       waiting_s=float(data[tc.VAR_WAITING_TIME]),
                       accumulated_waiting_s=float(data[tc.VAR_ACCUMULATED_WAITING_TIME]),
                       phase=self.signal.phase, signal_mode=self.signal.mode)
            true_radar.append(row)
            measured = row.copy()
            if c.position_noise_m:
                measured['distance_m'] = float(np.clip(distance + self.rng.normal(0, c.position_noise_m), 0, c.detection_m))
            if c.speed_noise_mps:
                measured['speed_mps'] = float(max(0, speed + self.rng.normal(0, c.speed_noise_mps)))
            observed.append(measured)
            if self.radar_file:
                self.radar_writer.writerow(measured)
        crossings = len(self.previous_incoming - incoming)
        self.crossed += crossings
        self.previous_incoming = incoming
        self._stopped = stopped
        self.observed, self.true_radar = observed, true_radar
        detected_ids = {r['vehicle_id'] for r in true_radar}
        radar_wait = sum(r['accumulated_waiting_s'] for r in true_radar)
        radar_queue = sum(r['speed_mps'] <= c.stop_speed for r in true_radar)
        radar_stops = len((stopped - prev_stopped) & detected_ids)
        reward = -(c.reward_wait * radar_wait + c.reward_queue * radar_queue + c.reward_stop * radar_stops) / 10.0
        self.collisions += api.simulation.getCollidingVehiclesNumber()
        self.teleports += api.simulation.getStartingTeleportNumber()
        # Records describe the signal applied during the just-completed second.
        if self.record and self.time <= c.horizon:
            self.records.append(dict(time_s=self.time, stage=STAGES[min((self.time - 1) // c.stage_seconds, 4)],
                                     queue_veh=queue, radar_queue_veh=radar_queue,
                                     max_lane_queue_m=max_lane_queue_m, radar_vehicle_count=len(observed),
                                     accumulated_waiting_s=wait, new_stops=new_stops,
                                     crossed_vehicles=crossings, arrived_vehicles=api.simulation.getArrivedNumber(),
                                     phase=self.signal.phase, signal_mode=self.signal.mode,
                                     signal_state=self.signal.state, green_elapsed_s=self.signal.elapsed,
                                     reward=reward))
        return reward

    def step(self, action, controller=None, end_time=None, step_seconds=None):
        end_time = self.config.horizon if end_time is None else end_time
        before = self.state() if (self.record and controller is None and self.time < self.config.horizon
                                  and not getattr(self, 'external_transitions', False)) else None
        if controller is None:
            self.signal.request(bool(action))
        reward, n = 0., 0
        for _ in range(self.config.decision_seconds if step_seconds is None else step_seconds):
            if self.time >= end_time:
                break
            if controller is not None:
                self.signal.request(bool(controller.action(self)))
            self.api.trafficlight.setRedYellowGreenState('J', self.signal.state)
            self.api.simulationStep()
            self.time += 1
            reward += self._sense()
            n += 1
            self.signal.tick()
        after = self.state()
        reward_mean = reward / max(n, 1)
        done = self.time >= end_time
        if before is not None:
            self.transitions.append((before, int(action), reward_mean, after, done))
        return after, reward_mean, done

    def has_pending(self):
        return self.api.simulation.getMinExpectedNumber() > 0

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.api.close()
        if self.radar_file:
            self.radar_file.close()
        if self.records:
            with (self.output / 'timeseries.csv').open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(self.records[0]))
                writer.writeheader()
                writer.writerows(self.records)
        if self.transitions:
            states, actions, rewards, next_states, dones = zip(*self.transitions)
            extra = {}
            if hasattr(self, 'transition_durations'):
                extra['duration_s'] = np.array(self.transition_durations)
                extra['discounts'] = self.config.gamma ** (extra['duration_s'] / 5)
            np.savez_compressed(self.output / 'decision_transitions.npz', states=np.array(states),
                                actions=np.array(actions), rewards=np.array(rewards),
                                next_states=np.array(next_states), dones=np.array(dones), **extra)
        (self.output / 'run_info.json').write_text(json.dumps({
            'backend': self.backend, 'seed': self.seed, 'observation': self.observation,
            'seconds_simulated': self.time, 'crossed_by_end_including_drain': self.crossed,
            'colliding_vehicle_events': self.collisions, 'teleport_events': self.teleports,
            'phase_waiting_guard_overrides': getattr(self, 'phase_override_count', 0),
            'config': self.config.to_dict()}, indent=2), encoding='utf-8')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
