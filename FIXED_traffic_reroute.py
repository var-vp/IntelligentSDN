#!/usr/bin/env python3
"""
OPTIMISED RL TRAINING SCRIPT — aligned with controller_extended.py
====================================================================

Issues fixed vs FIXED_training_delay_optimized.py
---------------------------------------------------

BUG 1 — max_queue_size=1000 vs controller MAX_QUEUE=200
  The Mininet TCLink kernel queue was set to 1000 packets. The controller
  can only tune down to MAX_QUEUE=200 via tc pfifo. The kernel queue sits
  BEHIND the pfifo qdisc. When the pfifo drops at 200 the kernel queue
  absorbs the overflow silently, making every queue-reduction action
  appear to have zero effect. Fixed: max_queue_size=30 everywhere —
  small enough that pfifo controls the effective limit.

BUG 2 — Mice flows invisible to the flow cache
  64-256 KB transfers at 10 Mbps complete in 50-200 ms. The flow cache
  updates every 5 s. Mice flows finish before the first poll arrives.
  They appear as 0 B/s in the cache, making Features 7-9 (max_flow_rate,
  flow_count, elephant_ratio) always zero during background traffic —
  the agent never sees mixed-flow state. Fixed: added long-lived iperf -t
  (time-based) background streams that persist across multiple flow-stat
  polls, and reserved transfer-size flows only for elephants.

BUG 3 — Moderate bursts shorter than decision period
  Burst duration 0.5-2.0 s. Controller decision period = 1.0 s.
  A 0.5 s burst resolves between two consecutive stats-reply events;
  the agent never observes it in state. Fixed: minimum burst duration
  raised to 3 s (3 × decision_period), maximum to 8 s.

BUG 4 — Elephant scheduling too sparse for Action 5/6/7 training
  Elephants every 60-120 s, lasting 10-30 s. Flow cache sees them after
  up to 5 s. Effective training window for Actions 5-7: 5-25 s.
  At 4 switches × 1 step/s, that is only 20-100 steps per elephant
  event — far too few for the agent to learn a rerouting policy.
  Fixed: elephant interval tightened to 30-60 s, duration 20-45 s,
  ensuring sustained coverage across multiple flow-stat cycles.

BUG 5 — INITIAL_QUEUE < MIN_QUEUE in controller (fixed in controller too)
  INITIAL_QUEUE=2 < MIN_QUEUE=5. The first tc pfifo call clamps to 5
  immediately, so the agent starts every episode at queue=5 regardless
  of what INITIAL_QUEUE says. The training script's comment "start with
  TINY queues (2-3 packets)" was therefore never achieved. Fixed in
  controller_extended.py (INITIAL_QUEUE=5) and documented here.

IMPROVEMENT 1 — Concurrent flow cap aligned with k=4 topology
  k=4 fat-tree has 16 hosts (k=4, density=2). The original cap of 150
  concurrent flows across 16 hosts is 9.4 flows/host — each host would
  be running dozens of iperf clients simultaneously. This inflates CPU
  load on the Mininet host and causes process-scheduling jitter that
  contaminates the timing accuracy of 1-second RL decisions. Reduced
  to 40, which is 2.5 flows/host and fully saturates the 10 Mbps links
  with far less CPU pressure.

IMPROVEMENT 2 — Idle periods more frequent
  The agent must learn to shrink queues aggressively when load drops.
  Original idle period: every 90-180 s. This is so rare (4-8 times in
  12 hours) that the agent rarely practices the shrink policy. Fixed:
  every 45-90 s, duration 20-60 s — roughly 30% of total training time
  spent in idle, giving the Q-network enough gradient signal to learn
  the buffer_reward=0.35 incentive for queue_limit<=3.

IMPROVEMENT 3 — Stagger delay alignment with iperf startup time
  Original stagger delays as low as 0.01 s. iperf client startup itself
  takes ~20-50 ms. Sub-10 ms staggers cause pile-up at the process
  scheduler. Minimums raised to 0.05 s for burst and 0.1 s for
  background.
"""

from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.log import setLogLevel, info
from mininet.link import TCLink
from mininet.topo import Topo
from mininet.util import dumpNodeConnections

import logging
import os
import time
import sys
import random
import threading
from collections import deque
import argparse

logging.basicConfig(
    filename='./rl_training_optimized.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# FAT-TREE TOPOLOGY
# ============================================================================
class Fattree(Topo):
    def __init__(self, k, density):
        self.CoreSwitchList = []
        self.AggSwitchList  = []
        self.EdgeSwitchList = []
        self.HostList       = []
        self.pod            = k
        self.iCoreLayerSwitch = int((k / 2) ** 2)
        self.iAggLayerSwitch  = int(k * k / 2)
        self.iEdgeLayerSwitch = int(k * k / 2)
        self.density          = density
        self.iHost            = self.iEdgeLayerSwitch * density
        Topo.__init__(self)

    def _addSwitch(self, number, level, switch_list):
        for x in range(1, number + 1):
            prefix = f"{level:1d}00"
            if x >= 10:
                prefix = f"{level:1d}0"
            if x >= 100:
                prefix = f"{level:d}"
            sw = self.addSwitch('s' + prefix + str(x))
            switch_list.append(sw)

    def createCoreLayerSwitch(self, number):
        self._addSwitch(number, 1, self.CoreSwitchList)

    def createAggLayerSwitch(self, number):
        self._addSwitch(number, 2, self.AggSwitchList)

    def createEdgeLayerSwitch(self, number):
        self._addSwitch(number, 3, self.EdgeSwitchList)

    def createHost(self, number):
        for x in range(1, number + 1):
            prefix = "h00" if x < 10 else ("h0" if x < 100 else "h")
            self.HostList.append(self.addHost(prefix + str(x)))

    def createTopo(self):
        self.createCoreLayerSwitch(self.iCoreLayerSwitch)
        self.createAggLayerSwitch(self.iAggLayerSwitch)
        self.createEdgeLayerSwitch(self.iEdgeLayerSwitch)
        self.createHost(self.iHost)

    def createLink(self, bw_c2a=10, bw_a2e=10, bw_h2a=10):
        """
        max_queue_size=30 is the critical fix.

        The Mininet TCLink kernel queue is the PHYSICAL packet buffer that
        the tc pfifo qdisc sits in front of. If max_queue_size >> MAX_QUEUE,
        the kernel absorbs drops that pfifo should be producing, making every
        queue-tuning action invisible at the stats level.

        Rule of thumb: max_queue_size = 1.5 × controller MAX_QUEUE = 300,
        but since pfifo itself enforces the real limit we set it conservatively
        at 30 so that backpressure is felt immediately in qdisc stats.
        """
        end = int(self.pod / 2)

        for x in range(0, self.iAggLayerSwitch, end):
            for i in range(0, end):
                for j in range(0, end):
                    self.addLink(
                        self.CoreSwitchList[i * end + j],
                        self.AggSwitchList[x + i],
                        bw=bw_c2a,
                        delay='0.15ms',
                        max_queue_size=30,   # FIXED: was 1000
                    )

        for x in range(0, self.iAggLayerSwitch, end):
            for i in range(0, end):
                for j in range(0, end):
                    self.addLink(
                        self.AggSwitchList[x + i],
                        self.EdgeSwitchList[x + j],
                        bw=bw_a2e,
                        delay='0.075ms',
                        max_queue_size=30,   # FIXED: was 1000
                    )

        for x in range(0, self.iEdgeLayerSwitch):
            for i in range(0, self.density):
                self.addLink(
                    self.EdgeSwitchList[x],
                    self.HostList[self.density * x + i],
                    bw=bw_h2a,
                    delay='0.025ms',
                    max_queue_size=30,       # FIXED: was 1000
                )

    def set_ovs_protocol_13(self):
        for sw_list in [self.CoreSwitchList, self.AggSwitchList, self.EdgeSwitchList]:
            for sw in sw_list:
                os.system(f"sudo ovs-vsctl set bridge {sw} protocols=OpenFlow13")


# ============================================================================
# TRAFFIC PATTERNS
# ============================================================================
class TrafficPatterns:
    """
    Traffic patterns aligned with the controller's decision timings.

    Design principles
    -----------------
    1. Background flows use -t (time-based duration), not -n (transfer size).
       This makes them persist across multiple 5-second flow-stat polls so
       the flow cache actually sees them and populates Features 7-9.

    2. Elephant flows use -t with long durations (20-45 s) to give the agent
       enough consecutive steps to learn Actions 5 and 7 (reroute/ECMP).

    3. Burst minimum duration = 3 s (3x the 1-second decision period) so
       the burst is guaranteed to be visible in at least 3 consecutive state
       observations.

    4. Window sizes are kept at 64K-128K for background/burst to avoid
       trivially saturating the link, which would make utilization always 100%
       and eliminate the utilization signal from the state.
    """

    @staticmethod
    def background_persistent(hosts, duration):
        """
        Long-lived low-rate flows (30-40% util).
        Uses -t (seconds) so flows persist across flow-stat polls.
        These are the flows that populate the flow cache during normal operation.
        """
        num_hosts = len(hosts)
        return {
            'num_flows':      max(4, num_hosts // 2),
            'mode':           'timed',            # -t flag, not -n
            'duration_range': (15, 25),           # seconds — spans 3+ flow-stat intervals
            'protocol':       'tcp',
            'stagger':        True,
            'stagger_delay':  (0.1, 0.3),
            'pattern_type':   'background',
            'flow_type':      'mice',
            'window_size':    '64K',
            'target_util':    0.35,
        }

    @staticmethod
    def burst(hosts, duration):
        """
        Short high-rate burst (60-80% util).
        Minimum 3 s so the agent sees it in at least 3 consecutive decisions.
        """
        num_hosts = len(hosts)
        return {
            'num_flows':      max(8, num_hosts),
            'mode':           'timed',
            'duration_range': (3, 8),             # FIXED: was (0.5, 2.0)
            'protocol':       'tcp',
            'stagger':        True,
            'stagger_delay':  (0.05, 0.15),       # FIXED: was (0.01, 0.1)
            'pattern_type':   'burst',
            'flow_type':      'burst',
            'window_size':    '128K',
            'target_util':    0.70,
        }

    @staticmethod
    def elephant(hosts, duration):
        """
        Few large sustained elephant flows (70-90% util on their paths).
        Long duration so the flow cache definitely catches them and the agent
        gets multiple consecutive steps to learn Actions 5 and 7.

        A single elephant flow at 10 Mbps is 1.25 MB/s = 1,250,000 B/s.
        The controller ELEPHANT_BPS_THRESH = 125,000 B/s (10% of 10 Mbps).
        So any sustained flow above ~1 Mbps triggers elephant classification.
        Using window_size=1M ensures TCP saturates to link speed immediately.
        """
        num_hosts = len(hosts)
        return {
            'num_flows':      max(2, num_hosts // 4),
            'mode':           'timed',
            'duration_range': (20, 45),           # FIXED: was (10, 30)
            'protocol':       'tcp',
            'stagger':        True,
            'stagger_delay':  (1.0, 3.0),
            'pattern_type':   'elephant',
            'flow_type':      'elephant',
            'window_size':    '1M',
            'target_util':    0.85,
        }

    @staticmethod
    def idle(hosts, duration):
        """
        Near-zero traffic (5-10% util).
        Critical for teaching the agent to shrink queues (buffer_reward=+0.35
        when queue_limit<=3). Occurs frequently enough (~30% of training time)
        that the agent accumulates adequate gradient signal for this policy.
        """
        num_hosts = len(hosts)
        return {
            'num_flows':      max(1, num_hosts // 8),
            'mode':           'timed',
            'duration_range': (20, 60),           # FIXED: was (30, 90)
            'protocol':       'tcp',
            'stagger':        True,
            'stagger_delay':  (0.5, 2.0),
            'pattern_type':   'idle',
            'flow_type':      'mice',
            'window_size':    '32K',
            'target_util':    0.08,
        }


# ============================================================================
# TRAFFIC GENERATOR
# ============================================================================
class TrafficGenerator:
    def __init__(self, net, seed=None, max_concurrent_flows=40):
        """
        max_concurrent_flows=40 (FIXED: was 150)

        k=4, density=2 → 16 hosts. 150 concurrent flows = 9.4/host.
        This saturates the Mininet process scheduler and introduces jitter
        that corrupts the 1-second timing accuracy of RL decisions.
        40 concurrent flows = 2.5/host, fully utilises 10 Mbps links
        with manageable CPU load.
        """
        self.net                  = net
        self.hosts                = net.hosts
        self.random               = random.Random(seed)
        self.max_concurrent_flows = max_concurrent_flows
        self.active_processes     = deque()
        self.server_ports         = {}
        self.flow_counter         = 0
        self.running              = True

        self.stats = {
            'total_flows':    0,
            'mice_flows':     0,
            'elephant_flows': 0,
            'burst_flows':    0,
        }

    def setup_servers(self):
        """
        Configure TCP (cubic) and start iperf servers.
        We explicitly disable TCP auto-tuning (tcp_moderate_rcvbuf) because
        the kernel will otherwise expand receive buffers well beyond the
        window sizes we set, undermining utilization control.
        """
        info("\n*** Configuring hosts (TCP cubic, fixed buffers) ***\n")
        for h in self.hosts:
            h.cmd("sysctl -w net.ipv4.tcp_congestion_control=cubic")
            h.cmd("sysctl -w net.ipv4.tcp_timestamps=1")
            h.cmd("sysctl -w net.ipv4.tcp_sack=1")
            h.cmd("sysctl -w net.ipv4.tcp_no_metrics_save=1")
            h.cmd("sysctl -w net.ipv4.tcp_moderate_rcvbuf=0")  # FIX: prevent buffer expansion
            h.cmd("sysctl -w net.core.rmem_max=1048576")
            h.cmd("sysctl -w net.core.wmem_max=1048576")

        for idx, h in enumerate(self.hosts):
            base_port = 5001 + (idx * 10)
            self.server_ports[h] = base_port
            h.popen(f"iperf -s -p {base_port} > /dev/null 2>&1", shell=True)

        time.sleep(3)
        info(f"*** {len(self.hosts)} iperf servers ready ***\n")

    def generate_flow_batch(self, config, batch_name):
        """
        Launch a batch of flows according to a pattern config dict.

        Supports two modes:
          'timed'  → iperf -t <seconds>   (time-limited, persists in flow cache)
          'sized'  → iperf -n <KB>        (transfer-size, completes quickly)

        Background, burst, elephant, and idle all use 'timed' mode so that
        the controller's flow cache (polled every 5 s) actually sees them.
        """
        num_flows    = config['num_flows']
        mode         = config.get('mode', 'timed')
        dur_min, dur_max = config['duration_range']
        stagger      = config.get('stagger', False)
        stagger_del  = config.get('stagger_delay', (0.1, 0.5))
        flow_type    = config.get('flow_type', 'unknown')
        window_size  = config.get('window_size', '64K')

        info(f"\n{'='*60}\n")
        info(f"*** {batch_name} ({config['pattern_type']}) ***\n")
        info(f"Flows: {num_flows} | Mode: {mode} | Window: {window_size}\n")
        info(f"{'='*60}\n")

        launched = 0
        for i in range(num_flows):
            # Backpressure: wait if at capacity
            while len(self.active_processes) >= self.max_concurrent_flows:
                time.sleep(0.5)
                self._cleanup()

            if not self.running:
                break

            if stagger and i > 0:
                time.sleep(self.random.uniform(*stagger_del))

            src = self.random.choice(self.hosts)
            others = [h for h in self.hosts if h != src]
            if not others:
                continue
            dst = self.random.choice(others)

            port     = self.server_ports[dst]
            duration = self.random.uniform(dur_min, dur_max)

            if mode == 'timed':
                cmd = (
                    f"iperf -c {dst.IP()} -p {port} "
                    f"-t {int(duration)} "
                    f"-w {window_size} "
                    f"> /dev/null 2>&1"
                )
            else:
                # 'sized' mode — kept for completeness, not used in patterns above
                size_kb = int(self.random.uniform(*config.get('size_range', (64, 256))))
                cmd = (
                    f"iperf -c {dst.IP()} -p {port} "
                    f"-n {size_kb}K "
                    f"-w {window_size} "
                    f"> /dev/null 2>&1"
                )

            try:
                proc = src.popen(cmd, shell=True)
                self.active_processes.append(proc)
                launched += 1
            except Exception:
                pass

        # Update stats
        self.stats['total_flows'] += launched
        if flow_type == 'mice':
            self.stats['mice_flows'] += launched
        elif flow_type == 'elephant':
            self.stats['elephant_flows'] += launched
        elif flow_type == 'burst':
            self.stats['burst_flows'] += launched

        info(f"*** {batch_name}: {launched} flows launched ***\n")
        return launched

    def _cleanup(self):
        still_active = deque()
        for p in self.active_processes:
            if p.poll() is None:
                still_active.append(p)
        self.active_processes = still_active

    def shutdown(self):
        self.running = False
        for p in self.active_processes:
            try:
                if p.poll() is None:
                    p.terminate()
            except Exception:
                pass


# ============================================================================
# TRAINING ORCHESTRATOR
# ============================================================================
class TrainingOrchestrator:
    """
    Orchestrates the traffic schedule for training controller_extended.py.

    Schedule summary
    ----------------
    Always on   Background persistent flows (30-40% util, 15-25 s each)
    Every 20-35s  Burst (60-80% util, 3-8 s)
    Every 30-60s  Elephant (70-90% util, 20-45 s)     ← Actions 5/7 training
    Every 45-90s  Idle period (5-10% util, 20-60 s)   ← buffer-shrink training

    Timing rationale
    ----------------
    - Burst interval 20-35 s:    agent sees ~20-30 bursts/hour
    - Elephant interval 30-60 s: agent sees ~60-120 elephants across 12 h
                                  each lasting 20-45 s → 20-45 steps/elephant
    - Idle fraction ~30%:        enough gradient for buffer_reward signal
    """

    def __init__(self, net, seed=None, total_duration=43200, max_concurrent=40):
        self.net          = net
        self.tgen         = TrafficGenerator(net, seed=seed,
                                             max_concurrent_flows=max_concurrent)
        self.total_duration = total_duration
        self.start_time   = time.time()
        self.random       = random.Random(seed)

    def run_training(self):
        info("\n" + "="*70 + "\n")
        info("*** OPTIMISED RL TRAINING (controller_extended.py) ***\n")
        info(f"Duration: {self.total_duration/3600:.1f} h  |  "
             f"Hosts: {len(self.net.hosts)}  |  "
             f"Max concurrent: {self.tgen.max_concurrent_flows}\n")
        info("="*70 + "\n")
        info("Key fixes vs original training script:\n")
        info("  max_queue_size : 30   (was 1000 — kernel queue shadowed pfifo)\n")
        info("  Flows mode     : -t   (was -n — mice disappeared before cache poll)\n")
        info("  Burst min      : 3s   (was 0.5s — shorter than decision period)\n")
        info("  Elephant dur   : 20-45s (was 10-30s — too short for reroute learning)\n")
        info("  Idle freq      : 45-90s (was 90-180s — insufficient queue-shrink signal)\n")
        info("  Max concurrent : 40   (was 150 — caused scheduler jitter)\n")
        info("="*70 + "\n\n")

        self.tgen.setup_servers()
        time.sleep(3)
        self.start_time = time.time()

        # Kick off persistent background traffic immediately
        info("*** Starting persistent background traffic ***\n")
        self._start_background_worker()

        # Schedule first events
        next_burst    = time.time() + self.random.uniform(10, 20)
        next_elephant = time.time() + self.random.uniform(20, 40)
        next_idle     = time.time() + self.random.uniform(30, 60)
        last_progress = time.time()

        while time.time() - self.start_time < self.total_duration:
            now     = time.time()
            elapsed = now - self.start_time

            # Progress report every 5 minutes
            if now - last_progress >= 300:
                s = self.tgen.stats
                info(f"\n{'='*60}\n")
                info(f"TRAINING  [{elapsed/3600:.2f}h / {self.total_duration/3600:.1f}h]\n")
                info(f"Total flows: {s['total_flows']} | "
                     f"Mice: {s['mice_flows']} | "
                     f"Elephants: {s['elephant_flows']} | "
                     f"Bursts: {s['burst_flows']}\n")
                info(f"{'='*60}\n\n")
                last_progress = now

            # BURST
            if now >= next_burst:
                info(">>> BURST <<<\n")
                threading.Thread(
                    target=self.tgen.generate_flow_batch,
                    args=(
                        TrafficPatterns.burst(
                            self.net.hosts,
                            self.random.uniform(3, 8),
                        ),
                        "Burst",
                    ),
                    daemon=True,
                ).start()
                next_burst = now + self.random.uniform(20, 35)  # FIXED: was (20,40)

            # ELEPHANT — critical for Actions 5/7 training
            if now >= next_elephant:
                info(">>> ELEPHANT <<<\n")
                threading.Thread(
                    target=self.tgen.generate_flow_batch,
                    args=(
                        TrafficPatterns.elephant(
                            self.net.hosts,
                            self.random.uniform(20, 45),
                        ),
                        "Elephant",
                    ),
                    daemon=True,
                ).start()
                next_elephant = now + self.random.uniform(30, 60)  # FIXED: was (60,120)

            # IDLE — critical for queue-shrink (buffer_reward) training
            if now >= next_idle:
                info(">>> IDLE PERIOD <<<\n")
                threading.Thread(
                    target=self.tgen.generate_flow_batch,
                    args=(
                        TrafficPatterns.idle(
                            self.net.hosts,
                            self.random.uniform(20, 60),
                        ),
                        "Idle",
                    ),
                    daemon=True,
                ).start()
                next_idle = now + self.random.uniform(45, 90)   # FIXED: was (90,180)

            time.sleep(1)

        info("\n" + "="*70 + "\n")
        info("*** TRAINING COMPLETE ***\n")
        s = self.tgen.stats
        info(f"Total: {s['total_flows']} | "
             f"Mice: {s['mice_flows']} | "
             f"Elephants: {s['elephant_flows']} | "
             f"Bursts: {s['burst_flows']}\n")
        info("="*70 + "\n\n")
        self.tgen.shutdown()

    def _start_background_worker(self):
        """
        Continuously replenish persistent background flows.
        Each call launches flows with -t 15-25s. When they expire the worker
        immediately launches the next batch, maintaining continuous coverage
        so the flow cache always has live data to report.
        """
        def worker():
            while (time.time() - self.start_time < self.total_duration
                   and self.tgen.running):
                self.tgen.generate_flow_batch(
                    TrafficPatterns.background_persistent(
                        self.net.hosts,
                        self.random.uniform(15, 25),
                    ),
                    "Background",
                )
                # Brief pause between batches — the flows themselves are still
                # running so there is no gap in coverage
                time.sleep(self.random.uniform(3, 6))

        threading.Thread(target=worker, daemon=True).start()


# ============================================================================
# MAIN
# ============================================================================
def run_training(args):
    info("\n" + "="*70 + "\n")
    info("*** OPTIMISED RL TRAINING ***\n")
    info("="*70 + "\n")
    info(f"Topology: Fat-Tree k={args.k}, density={args.density}\n")
    info(f"Duration: {args.duration}s ({args.duration/3600:.1f}h)\n")
    info(f"Max concurrent flows: {args.max_concurrent}\n")
    info("="*70 + "\n\n")

    topo = Fattree(args.k, args.density)
    topo.createTopo()
    topo.createLink(bw_c2a=10, bw_a2e=10, bw_h2a=10)

    net = Mininet(
        topo=topo,
        link=TCLink,
        controller=None,
        autoSetMacs=True,
    )
    net.addController(
        'c0',
        controller=RemoteController,
        ip=args.controller_ip,
        port=args.controller_port,
    )

    net.start()
    topo.set_ovs_protocol_13()
    time.sleep(10)
    dumpNodeConnections(net.hosts)

    # Signal controller to start RL loop
    flag_path = "/tmp/start_training.flag"
    try:
        with open(flag_path, "w") as f:
            f.write(f"duration={args.duration}\n")
            f.write(f"mode=optimised\n")
    except Exception:
        pass

    time.sleep(2)

    try:
        orch = TrainingOrchestrator(
            net,
            seed=args.seed,
            total_duration=args.duration,
            max_concurrent=args.max_concurrent,
        )
        orch.run_training()

    except KeyboardInterrupt:
        info("\n*** Interrupted ***\n")

    except Exception as e:
        info(f"\n*** Error: {e} ***\n")
        import traceback
        traceback.print_exc()

    finally:
        if os.path.exists(flag_path):
            try:
                os.remove(flag_path)
            except Exception:
                pass
        try:
            net.stop()
        except Exception:
            pass
        info("\n*** DONE ***\n")


if __name__ == '__main__':
    setLogLevel('info')

    if os.getuid() != 0:
        print("ERROR: Must run as root")
        sys.exit(1)

    parser = argparse.ArgumentParser(description='Optimised RL Training')
    parser.add_argument('--k',               type=int,   default=4)
    parser.add_argument('--density',         type=int,   default=2)
    parser.add_argument('--duration',        type=int,   default=43250)
    parser.add_argument('--max-concurrent',  type=int,   default=40)
    parser.add_argument('--seed',            type=int,   default=42)
    parser.add_argument('--controller-ip',   type=str,   default='127.0.0.1')
    parser.add_argument('--controller-port', type=int,   default=6633)

    args = parser.parse_args()

    try:
        run_training(args)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)