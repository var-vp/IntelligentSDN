#!/usr/bin/env python3
"""
EXTENDED RL CONTROLLER — 3-Layer Action Space (Final)
======================================================

Inherits everything from BEST_controller_predictive_state2.py and adds:

  Action 5  REROUTE     Redirect elephant via best end-to-end alternate path
  Action 6  SAMPLE      Install 1-second header-only sampling rule (safe)
  Action 7  ECMP_SPLIT  Move second-largest flow via best alternate path

Three upgrades applied in this version
---------------------------------------
1. Global Flow Cooldown Table (anti-flapping)
   self.global_flow_decisions[(src_ip, dst_ip)] is checked before every
   reroute.  If any switch rerouted this flow within REROUTE_COOLDOWN
   seconds, all other switches are blocked from touching it.  This prevents
   the route oscillation problem where switch A reroutes to path2, switch B
   then reroutes back to path1, and the loop repeats.

2. Dynamic Elephant Threshold
   ELEPHANT_BPS_THRESH is computed as a fraction of the link capacity
   (ELEPHANT_RATIO_THRESH * LINK_CAPACITY_BPS / 8), not hardcoded.
   Hardcoding 500,000 B/s is only correct for 10 Mbps links.  At 100 Mbps
   every flow would be a mouse and Action 5 would never fire.

3. Reroute Count Decay + 10th State Feature
   self.port_reroute_counts[(dpid, port)] tracks how many reroutes fired
   in the last REROUTE_COUNT_WINDOW seconds.  A background hub.spawn_after
   decrements the count automatically.  This value is passed to the agent
   as state feature 10 (reroute_count_last_10s), giving the RL agent
   temporal awareness of its own recent interventions so it can learn that
   spamming Action 5 yields diminishing returns.
   It is also used in build_action_mask() as a hard cap of 3 reroutes per
   window to block oscillation at the action-selection level.

Launch:  ryu-manager controller_extended.py ArpHandler.py
Requires: ddqn_per_agent_v2.py, ArpHandler.py, switch_selection_strategies.py
"""

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import (
    MAIN_DISPATCHER, DEAD_DISPATCHER, CONFIG_DISPATCHER, set_ev_cls
)
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, arp, ether_types
from ryu.lib import hub

import time
import csv
import os
import re
import subprocess
import networkx as nx

import ArpHandler
from ddqn_per_agent_v2 import (
    DoubleDQNAgent, STATE_SIZE, ACTION_SIZE,
    ACTION_QUEUE_MINUS2, ACTION_QUEUE_MINUS1, ACTION_QUEUE_HOLD,
    ACTION_QUEUE_PLUS1,  ACTION_QUEUE_PLUS2,
    ACTION_REROUTE, ACTION_SAMPLE, ACTION_ECMP_SPLIT,
    QUEUE_ACTIONS,
)
from switch_selection_strategies import IntelligentSwitchSelector


class ExtendedRLController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {"ArpHandler": ArpHandler.ArpHandler}

    # ── Tuneable constants ────────────────────────────────────────────────────

    # Link capacity and dynamic elephant threshold
    # Change LINK_CAPACITY_BPS to match your Mininet link speed.
    # Default Fat-Tree in your training script uses 10 Mbps.
    LINK_CAPACITY_BPS    = 10_000_000   # bits per second  (10 Mbps)
    ELEPHANT_RATIO_THRESH = 0.10        # flow is elephant if > 10% of link capacity

    # Derived at class level for readability; overridden in __init__ if needed.
    # bytes/s = (bits/s * ratio) / 8
    ELEPHANT_BPS_THRESH  = int(LINK_CAPACITY_BPS * ELEPHANT_RATIO_THRESH / 8)  # 125,000 B/s

    # Flow rule priorities
    REROUTE_PRIORITY     = 100   # overrides ArpHandler default (priority 10)
    SAMPLE_PRIORITY      = 200   # overrides reroute rules

    # Flow rule timeouts
    REROUTE_IDLE_TIMEOUT = 10    # seconds — rule expires when flow goes quiet
    REROUTE_HARD_TIMEOUT = 30    # seconds — rule always expires (TCAM safety)
    SAMPLE_HARD_TIMEOUT  = 1     # seconds — sampling rule always self-deletes
    SAMPLE_MAX_LEN       = 128   # bytes   — only IP/TCP headers to controller

    # Anti-flapping cooldown
    REROUTE_COOLDOWN     = 5.0   # seconds — global block on re-rerouting same flow

    # Flow stats cadence
    FLOW_STATS_INTERVAL  = 5.0   # seconds between OFPFlowStatsRequest per switch

    # Reroute count decay window
    REROUTE_COUNT_WINDOW = 10.0  # seconds — rolling window for Feature 10

    # ─────────────────────────────────────────────────────────────────────────
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arp_handler = kwargs["ArpHandler"]
        self.datapaths   = {}

        # ── RL agent (10-state, 8-action) ────────────────────────────────────
        self.agent = DoubleDQNAgent(
            state_size=STATE_SIZE,    # 10
            action_size=ACTION_SIZE,  # 8
            lr=3e-4,
            gamma=0.98,
            buffer_size=10000,
            batch_size=128,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.9999,
            tau=0.005,
        )

        self.training_mode   = True
        self.max_episodes    = 43250
        self.current_episode = 0

        # ── Timing ───────────────────────────────────────────────────────────
        self.stats_interval       = 1.0
        self.decision_period      = 1.0
        self.start_time           = time.time()
        self.last_decision_time   = {}
        self.last_flow_stats_time = {}

        # ── Checkpointing ────────────────────────────────────────────────────
        self.checkpoint_interval  = 4 * 3600
        self.last_checkpoint_time = time.time()
        self.checkpoint_counter   = 0

        # ── Queue config (identical to v1) ───────────────────────────────────
        self.MIN_QUEUE     = 5
        self.MAX_QUEUE     = 200
        self.INITIAL_QUEUE = 2
        self.queue_adjustments = {
            ACTION_QUEUE_MINUS2: -2,
            ACTION_QUEUE_MINUS1: -1,
            ACTION_QUEUE_HOLD:    0,
            ACTION_QUEUE_PLUS1:  +1,
            ACTION_QUEUE_PLUS2:  +2,
        }
        self.current_queue_limit = {}

        # ── Predictive state (identical to v1) ───────────────────────────────
        self.port_prev_queue_len = {}
        self.ewma_queue          = {}
        self.EWMA_ALPHA          = 0.2
        self.VEL_SCALE           = 20.0

        # ── Flow cache ───────────────────────────────────────────────────────
        # flow_cache[(dpid, port_no)] = list of flow dicts sorted by rate_bps desc
        # Each dict: {src_ip, dst_ip, bytes_total, rate_bps,
        #             packet_count, duration_sec, out_port}
        self.flow_cache      = {}
        self.flow_prev_bytes = {}   # (dpid, src_ip, dst_ip) -> byte count
        self.flow_prev_time  = {}   # (dpid, src_ip, dst_ip) -> timestamp

        # ── Upgrade 1: Global Flow Cooldown Table (anti-flapping) ────────────
        # Key:   (src_ip, dst_ip)  — identifies the flow network-wide
        # Value: {"timestamp": float, "switch_dpid": int, "port": int}
        #
        # Before any reroute we check if another switch already rerouted this
        # flow within REROUTE_COOLDOWN seconds.  If so we return False
        # immediately.  This prevents:
        #   Switch A reroutes -> Switch B reroutes back -> Switch A reroutes...
        # The key uses only (src, dst) so the block is truly global — it does
        # not matter which switch is checking.
        self.global_flow_decisions = {}

        # ── Upgrade 3: Per-port reroute count (Feature 10 source) ────────────
        # Tracks how many reroutes fired from (dpid, port) in the last
        # REROUTE_COUNT_WINDOW seconds.  Each entry is auto-decremented by
        # a hub.spawn_after callback after REROUTE_COUNT_WINDOW seconds.
        # This gives the RL agent memory of its own recent interventions.
        self.port_reroute_counts = {}   # (dpid, port_no) -> int

        # ── Switch selection ─────────────────────────────────────────────────
        self.selection_strategy     = 'CORE'
        self.k                      = 4   # fat-tree k
        # In IntelligentSwitchSelector, k = number of switches to select.
        # For CORE strategy on a k=4 fat-tree: (k/2)^2 = 4 core switches.
        # For HYBRID/BOTTLENECK you may want more — adjust _num_rl_switches.
        self._num_rl_switches       = (self.k // 2) ** 2   # = 4
        self.switch_selector        = IntelligentSwitchSelector(
            strategy=self.selection_strategy,
            k=self._num_rl_switches,
            pod_size=self.k,
        )
        self.intelligent_switch_ids = set()

        # ── Per-port byte/packet tracking ────────────────────────────────────
        self.port_bytes_prev   = {}
        self.port_time_prev    = {}
        self.port_packets_prev = {}
        self.iface_drop_prev   = {}
        self.iface_qdisc_init  = set()

        # ── MDP transition cache (Fix: S_t != S_{t+1}) ───────────────────────
        # The Bellman update requires a genuine (S_t, A_t, R_{t+1}, S_{t+1})
        # tuple where S_{t+1} reflects the network's reaction to A_t.
        # We cannot produce that inside a single stats-reply handler because
        # the action has not yet had time to affect the network.
        #
        # Solution: on each handler invocation for port (dpid, port_no):
        #   1. The stats we just received ARE S_{t+1} — the state AFTER the
        #      action we chose last cycle had 1 second to take effect.
        #   2. We close the previous MDP step:
        #        store(prev_state, prev_action, reward, current_state)
        #   3. We choose the new action and cache (current_state, action)
        #        for closure on the next invocation.
        #
        # Structure: (dpid, port_no) -> (state_tensor, action_int, action_name)
        # action_name is cached too so the log entry stays consistent with
        # the action that was actually taken in that step.
        self.last_state_action = {}

        # ── CSV logs ─────────────────────────────────────────────────────────
        self.metrics_log_path         = "extended_rl_metrics.csv"
        self.checkpoint_log_path      = "extended_rl_checkpoints.csv"
        self.per_diagnostics_log_path = "per_diagnostics.csv"
        self._init_metrics_log()

        # ── Rolling stats for monitoring ─────────────────────────────────────
        self.recent_rewards     = []
        self.recent_actions     = []
        self.recent_delays      = []
        self.recent_queue_sizes = []
        self.recent_velocities  = []
        self.recent_ewma        = []

        # ── PER diagnostics ──────────────────────────────────────────────────
        self.per_diagnostics = {
            'td_errors':            [],
            'avg_priority':         [],
            'priority_updates':     0,
            'last_diagnostic_time': time.time(),
        }
        self.diagnostic_interval = 300

        # ── Training start flag ──────────────────────────────────────────────
        self.training_started   = False
        self.training_flag_path = "/tmp/start_training.flag"

        self.monitor_thread  = hub.spawn(self._monitor)
        self.training_thread = hub.spawn(self._training_loop)
        self._log_startup()

    # ═════════════════════════════════════════════════════════════════════════
    # STARTUP LOGGING
    # ═════════════════════════════════════════════════════════════════════════

    def _log_startup(self):
        self.logger.info("=" * 70)
        self.logger.info("EXTENDED 3-LAYER RL CONTROLLER  (Final)")
        self.logger.info("  State  : %d features | Actions: %d", STATE_SIZE, ACTION_SIZE)
        self.logger.info("  Layer 1 Queue   : Actions 0-4  (-2 to +2 packets)")
        self.logger.info("  Layer 2 Reroute : Action  5  (best end-to-end path)")
        self.logger.info("  Layer 3 Sample  : Action  6  (1s header-only window)")
        self.logger.info("  Layer 3 ECMP    : Action  7  (move 2nd flow, best path)")
        self.logger.info("  Link capacity   : %d Mbps", self.LINK_CAPACITY_BPS // 1_000_000)
        self.logger.info("  Elephant thresh : %.0f B/s  (%.0f%% of link)",
                         self.ELEPHANT_BPS_THRESH, self.ELEPHANT_RATIO_THRESH * 100)
        self.logger.info("  Reroute cooldown: %.1f s (global anti-flapping)",
                         self.REROUTE_COOLDOWN)
        self.logger.info("  Reroute window  : %.1f s (Feature 10 decay)",
                         self.REROUTE_COUNT_WINDOW)
        self.logger.info("  Flow stats cadence: %.0f s", self.FLOW_STATS_INTERVAL)
        self.logger.info("=" * 70)

    # ═════════════════════════════════════════════════════════════════════════
    # CSV LOG INITIALISATION
    # ═════════════════════════════════════════════════════════════════════════

    def _init_metrics_log(self):
        if not os.path.isfile(self.metrics_log_path):
            with open(self.metrics_log_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "episode", "time_s", "dpid", "port",
                    "action", "action_name", "queue_limit",
                    "reward", "throughput_kbps", "drop_rate_pps",
                    "packet_rate_pps", "queue_length", "queue_velocity",
                    "ewma_queue", "max_flow_rate", "flow_count",
                    "elephant_ratio", "reroute_count_last_10s",  # Feature 10
                    "link_util", "queue_util", "epsilon", "delay_ms",
                ])
        if not os.path.isfile(self.checkpoint_log_path):
            with open(self.checkpoint_log_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "checkpoint", "episode", "time_s",
                    "avg_reward", "avg_delay", "avg_queue",
                    "avg_velocity", "epsilon", "model_file",
                ])
        if not os.path.isfile(self.per_diagnostics_log_path):
            with open(self.per_diagnostics_log_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "time_s", "avg_td_error", "max_td_error", "min_td_error",
                    "avg_priority", "priority_updates",
                    "buffer_size", "high_priority_ratio",
                ])

    # ═════════════════════════════════════════════════════════════════════════
    # OPENFLOW EVENT HANDLERS
    # ═════════════════════════════════════════════════════════════════════════

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        dp     = ev.msg.datapath
        ofp    = dp.ofproto
        parser = dp.ofproto_parser
        self.datapaths[dp.id] = dp
        self._update_intelligent_switches()
        self.add_flow(dp, 0, parser.OFPMatch(),
                      [parser.OFPActionOutput(ofp.OFPP_CONTROLLER,
                                              ofp.OFPCML_NO_BUFFER)])
        self.add_flow(dp, 65534,
                      parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IPV6), [])

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        dp = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            self.datapaths[dp.id] = dp
            self._update_intelligent_switches()
        elif ev.state == DEAD_DISPATCHER:
            self.datapaths.pop(dp.id, None)
            self._update_intelligent_switches()

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg     = ev.msg
        dp      = msg.datapath
        in_port = msg.match['in_port']
        pkt     = packet.Packet(msg.data)
        eth     = pkt.get_protocol(ethernet.ethernet)

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return
        if eth.ethertype == ether_types.ETH_TYPE_IPV6:
            self.add_flow(dp, 65534,
                          dp.ofproto_parser.OFPMatch(
                              eth_type=ether_types.ETH_TYPE_IPV6), [])
            return

        arp_pkt = pkt.get_protocol(arp.arp)
        if arp_pkt:
            self.arp_handler.arp_handler(dp, in_port, pkt, arp_pkt)
            return

        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        if ip_pkt:
            self.arp_handler.ip_handler(dp, in_port, pkt, ip_pkt)

    # ═════════════════════════════════════════════════════════════════════════
    # MONITOR LOOP
    # ═════════════════════════════════════════════════════════════════════════

    def _monitor(self):
        while True:
            if not self.training_started:
                if os.path.exists(self.training_flag_path):
                    self.training_started = True
                    self.logger.info("Training flag detected — starting RL loop")
                else:
                    hub.sleep(5)
                    continue

            now = time.time()
            for dp in list(self.datapaths.values()):
                self._request_port_stats(dp)
                if now - self.last_flow_stats_time.get(dp.id, 0) >= self.FLOW_STATS_INTERVAL:
                    self._request_flow_stats(dp)
                    self.last_flow_stats_time[dp.id] = now

            if now - self.last_checkpoint_time >= self.checkpoint_interval:
                self._checkpoint()
                self.last_checkpoint_time = now

            if now - self.per_diagnostics['last_diagnostic_time'] >= self.diagnostic_interval:
                self._log_per_diagnostics()
                self.per_diagnostics['last_diagnostic_time'] = now

            hub.sleep(self.stats_interval)

    # ═════════════════════════════════════════════════════════════════════════
    # ASYNC TRAINING LOOP — completely separate from OpenFlow event loop
    # ═════════════════════════════════════════════════════════════════════════

    def _training_loop(self):
        """
        PyTorch backpropagation runs here — never in an OpenFlow handler.

        Ryu uses eventlet cooperative multitasking. Any blocking C++ call
        (torch.backward) inside an OFP handler starves all other greenlets,
        causing switch heartbeat timeouts and OpenFlow disconnections.

        This greenlet wakes every TRAIN_INTERVAL seconds, performs one
        gradient step if the buffer is ready, then yields via hub.sleep so
        the eventlet scheduler can process pending OFP messages.

        TRAIN_INTERVAL is intentionally longer than stats_interval (1s) so
        training never competes with stats collection on the same tick.
        """
        TRAIN_INTERVAL = 2.0   # seconds between gradient steps

        while True:
            hub.sleep(TRAIN_INTERVAL)

            if not self.training_started or not self.training_mode:
                continue

            try:
                td_err = self.agent.train_step()
                if td_err is not None:
                    self.per_diagnostics['td_errors'].append(abs(td_err))
                    self.per_diagnostics['priority_updates'] += 1
            except Exception as e:
                self.logger.error("Training step failed: %s", e)


        dp.send_msg(dp.ofproto_parser.OFPPortStatsRequest(
            dp, 0, dp.ofproto.OFPP_ANY))

    def _request_flow_stats(self, dp):
        dp.send_msg(dp.ofproto_parser.OFPFlowStatsRequest(dp))

    # ═════════════════════════════════════════════════════════════════════════
    # FLOW STATS REPLY — builds the flow cache (async, slow cadence)
    # ═════════════════════════════════════════════════════════════════════════

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        dp   = ev.msg.datapath
        dpid = dp.id
        now  = time.time()

        port_flows = {}

        for stat in ev.msg.body:
            if stat.priority == 0:
                continue
            match = stat.match
            if match.get('eth_type') != ether_types.ETH_TYPE_IP:
                continue

            src_ip = match.get('ipv4_src', '0.0.0.0')
            dst_ip = match.get('ipv4_dst', '0.0.0.0')

            out_port = None
            for inst in stat.instructions:
                for act in getattr(inst, 'actions', []):
                    if hasattr(act, 'port'):
                        out_port = act.port
            if out_port is None or out_port in (dp.ofproto.OFPP_CONTROLLER,
                                                dp.ofproto.OFPP_FLOOD):
                continue

            fkey       = (dpid, src_ip, dst_ip)
            prev_bytes = self.flow_prev_bytes.get(fkey, stat.byte_count)
            prev_time  = self.flow_prev_time.get(fkey, now - self.FLOW_STATS_INTERVAL)
            delta_t    = max(1e-3, now - prev_time)
            rate_bps   = max(0.0, (stat.byte_count - prev_bytes) / delta_t)

            self.flow_prev_bytes[fkey] = stat.byte_count
            self.flow_prev_time[fkey]  = now

            port_flows.setdefault(out_port, []).append({
                'src_ip':       src_ip,
                'dst_ip':       dst_ip,
                'bytes_total':  stat.byte_count,
                'rate_bps':     rate_bps,
                'packet_count': stat.packet_count,
                'duration_sec': stat.duration_sec,
                'out_port':     out_port,
            })

        for port_no, flows in port_flows.items():
            flows.sort(key=lambda f: f['rate_bps'], reverse=True)
            self.flow_cache[(dpid, port_no)] = flows

    # ═════════════════════════════════════════════════════════════════════════
    # FLOW CACHE ACCESSORS
    # ═════════════════════════════════════════════════════════════════════════

    def _get_flow_features(self, dpid: int, port_no: int) -> dict:
        flows = self.flow_cache.get((dpid, port_no), [])
        if not flows:
            return {'max_flow_rate': 0.0, 'flow_count': 0, 'elephant_ratio': 0.0}
        total_rate     = sum(f['rate_bps'] for f in flows)
        max_rate       = flows[0]['rate_bps']
        elephant_ratio = (max_rate / total_rate) if total_rate > 0 else 0.0
        return {
            'max_flow_rate':  max_rate,
            'flow_count':     len(flows),
            'elephant_ratio': elephant_ratio,
        }

    def _get_elephant_flow(self, dpid: int, port_no: int):
        """
        Return the highest-rate flow if it is an elephant (Upgrade 2).
        Threshold is dynamic: ELEPHANT_RATIO_THRESH * LINK_CAPACITY_BPS / 8.
        """
        flows = self.flow_cache.get((dpid, port_no), [])
        if flows and flows[0]['rate_bps'] >= self.ELEPHANT_BPS_THRESH:
            return flows[0]
        return None

    def _get_second_largest_flow(self, dpid: int, port_no: int):
        flows = self.flow_cache.get((dpid, port_no), [])
        return flows[1] if len(flows) >= 2 else None

    # ═════════════════════════════════════════════════════════════════════════
    # UPGRADE 3 HELPERS — reroute count tracking + decay
    # ═════════════════════════════════════════════════════════════════════════

    def _increment_reroute_count(self, dpid: int, port_no: int):
        """
        Increment the rolling reroute counter for this port and schedule
        an automatic decrement after REROUTE_COUNT_WINDOW seconds.

        This keeps the count as a true sliding-window value rather than a
        monotonically increasing integer that would saturate Feature 10
        early in training and never recover.
        """
        key = (dpid, port_no)
        self.port_reroute_counts[key] = self.port_reroute_counts.get(key, 0) + 1
        hub.spawn_after(self.REROUTE_COUNT_WINDOW,
                        self._decrement_reroute_count, key)

    def _decrement_reroute_count(self, key):
        if self.port_reroute_counts.get(key, 0) > 0:
            self.port_reroute_counts[key] -= 1

    def _get_reroute_count(self, dpid: int, port_no: int) -> int:
        return self.port_reroute_counts.get((dpid, port_no), 0)

    # ═════════════════════════════════════════════════════════════════════════
    # MAIN RL LOOP — PORT STATS REPLY
    # ═════════════════════════════════════════════════════════════════════════

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        if not self.training_started:
            return

        dp   = ev.msg.datapath
        dpid = dp.id

        if dpid not in self.intelligent_switch_ids:
            return

        now = time.time()
        if now - self.last_decision_time.get(dpid, 0.0) < self.decision_period:
            return
        self.last_decision_time[dpid] = now

        for stat in ev.msg.body:
            if stat.port_no in (dp.ofproto.OFPP_LOCAL, 0):
                continue

            key = (dpid, stat.port_no)

            # ── Delta calculations ────────────────────────────────────────
            prev_bytes   = self.port_bytes_prev.get(key, stat.tx_bytes)
            prev_time    = self.port_time_prev.get(key, now - self.stats_interval)
            prev_packets = self.port_packets_prev.get(key, stat.tx_packets)

            delta_bytes   = max(0, stat.tx_bytes   - prev_bytes)
            delta_packets = max(0, stat.tx_packets - prev_packets)
            delta_t       = max(1e-6, now - prev_time)

            self.port_bytes_prev[key]   = stat.tx_bytes
            self.port_time_prev[key]    = now
            self.port_packets_prev[key] = stat.tx_packets

            throughput_kbps = (delta_bytes * 8.0) / delta_t / 1000.0
            packet_rate_pps = delta_packets / delta_t

            # ── qdisc backlog + drop rate ─────────────────────────────────
            iface = self._iface_name(dpid, stat.port_no)
            backlog_pkts, total_drops = self._get_qdisc_stats(iface)

            prev_drops         = self.iface_drop_prev.get(iface, total_drops)
            dropped_since_last = max(0, total_drops - prev_drops)
            self.iface_drop_prev[iface] = total_drops
            drop_rate_pps = dropped_since_last / delta_t

            # ── Predictive state ──────────────────────────────────────────
            prev_queue     = self.port_prev_queue_len.get(key, backlog_pkts)
            queue_velocity = backlog_pkts - prev_queue
            self.port_prev_queue_len[key] = backlog_pkts

            if key not in self.ewma_queue:
                self.ewma_queue[key] = float(backlog_pkts)
            else:
                self.ewma_queue[key] = (
                    self.EWMA_ALPHA * backlog_pkts
                    + (1.0 - self.EWMA_ALPHA) * self.ewma_queue[key]
                )

            # ── Flow features from cache ──────────────────────────────────
            flow_feats     = self._get_flow_features(dpid, stat.port_no)
            reroute_count  = self._get_reroute_count(dpid, stat.port_no)

            # ── Composite stats dict ──────────────────────────────────────
            queue_limit    = self.current_queue_limit.get(key, self.INITIAL_QUEUE)
            queue_util     = backlog_pkts / max(1, queue_limit)
            queue_delay_ms = (backlog_pkts / 83333.0 * 1000.0
                              if packet_rate_pps > 1.0 else 0.0)

            stats_dict = {
                # Original features
                'queue_length':   backlog_pkts,
                'queue_velocity': queue_velocity,
                'ewma_queue':     self.ewma_queue[key],
                'throughput':     throughput_kbps,
                'drop_rate':      drop_rate_pps,
                'packet_rate':    packet_rate_pps,
                'delay_ms':       queue_delay_ms,
                # Flow cache features
                'max_flow_rate':  flow_feats['max_flow_rate'],
                'flow_count':     flow_feats['flow_count'],
                'elephant_ratio': flow_feats['elephant_ratio'],
                # Feature 10 (Upgrade 3)
                'reroute_count_last_10s': reroute_count,
            }

            # ── Build action mask ─────────────────────────────────────────
            elephant_flow   = self._get_elephant_flow(dpid, stat.port_no)
            elephant_exists = elephant_flow is not None
            ecmp_capable    = self._has_alternate_path(dpid)

            mask = self.agent.build_action_mask(
                queue_util=queue_util,
                elephant_exists=elephant_exists,
                ecmp_capable=ecmp_capable,
                reroute_count=reroute_count,
            )

            # ══════════════════════════════════════════════════════════════
            # CORRECT MDP TRANSITION  (Fix: S_t != S_{t+1})
            #
            # The stats we just received are the network's response to the
            # action we chose ONE DECISION PERIOD AGO.  This means:
            #
            #   current_state  = S_t  = "what the network looks like NOW,
            #                            after the previous action had 1s
            #                            to take effect"
            #   reward         = R_t  = "how good is the state we arrived in"
            #
            # Step 1 — close the PREVIOUS transition using current_state
            #          as S_{t+1} (the genuine next-state for that step).
            # Step 2 — choose a NEW action from current_state.
            # Step 3 — cache (current_state, new_action) so the NEXT
            #          invocation can close this transition with real data.
            #
            # This guarantees the Bellman target uses a state that actually
            # reflects the consequence of the action, fixing the TD-error
            # collapse that occurs when S_t == S_{t+1}.
            # ══════════════════════════════════════════════════════════════

            # Step 1: S_{t+1} for the previous step
            current_state = self.agent.get_state_tensor(stats_dict)
            reward        = self._calculate_reward(stats_dict, queue_limit, action=0)
            # (action=0 is a placeholder — reward is state-only here;
            #  action shaping is applied below when we know the real action)

            if key in self.last_state_action:
                prev_state, prev_action, prev_action_name = self.last_state_action[key]

                # Re-compute reward shaped for the action that was actually taken
                shaped_reward = self._calculate_reward(stats_dict, queue_limit,
                                                       action=prev_action)

                # Store the genuine (S_{t-1}, A_{t-1}, R_t, S_t) transition.
                # Training happens in _training_loop (separate greenlet) —
                # never block the OpenFlow event loop with PyTorch backprop.
                self.agent.store(prev_state, prev_action, shaped_reward, current_state)
                self.agent.step_counter += 1

                self._track(shaped_reward, prev_action, queue_delay_ms,
                            queue_limit, queue_velocity, self.ewma_queue[key])
                self._log_step(dpid, stat.port_no, prev_action, prev_action_name,
                               shaped_reward, stats_dict, queue_limit)

            # Step 2: choose new action from current state
            action = self.agent.choose_action(current_state, mask)

            # Step 3: execute and cache for next cycle
            action_name, queue_limit = self._execute_action(
                dp, stat.port_no, action, dpid, elephant_flow
            )
            self.last_state_action[key] = (current_state, action, action_name)

            self.switch_selector.update_traffic_stats(dpid, delta_bytes,
                                                      dropped_since_last)

    # ═════════════════════════════════════════════════════════════════════════
    # ACTION DISPATCHER
    # ═════════════════════════════════════════════════════════════════════════

    def _execute_action(self, dp, port_no, action, dpid, elephant_flow):
        queue_limit = self.current_queue_limit.get((dpid, port_no), self.INITIAL_QUEUE)

        if action in QUEUE_ACTIONS:
            queue_limit = self._apply_action_queue(dp, port_no, action)
            return f"QUEUE{self.queue_adjustments[action]:+d}", queue_limit

        if action == ACTION_REROUTE:
            ok = self._apply_action_reroute(dp, port_no, elephant_flow)
            return ("REROUTE_OK" if ok else "REROUTE_FAIL"), queue_limit

        if action == ACTION_SAMPLE:
            ok = self._apply_action_sample(dp, port_no, elephant_flow)
            return ("SAMPLE_OK" if ok else "SAMPLE_FAIL"), queue_limit

        if action == ACTION_ECMP_SPLIT:
            second = self._get_second_largest_flow(dpid, port_no)
            ok = self._apply_action_ecmp(dp, port_no, second, dpid)
            return ("ECMP_OK" if ok else "ECMP_FAIL"), queue_limit

        return "UNKNOWN", queue_limit

    # ═════════════════════════════════════════════════════════════════════════
    # ACTION IMPLEMENTATIONS
    # ═════════════════════════════════════════════════════════════════════════

    def _apply_action_queue(self, datapath, port_no, action):
        """Layer 1: queue size tuning via tc pfifo (unchanged from v1)."""
        dpid  = datapath.id
        iface = self._iface_name(dpid, port_no)
        key   = (dpid, port_no)

        if key not in self.current_queue_limit:
            self.current_queue_limit[key] = self.INITIAL_QUEUE

        new_limit = max(
            self.MIN_QUEUE,
            min(self.MAX_QUEUE,
                self.current_queue_limit[key] + self.queue_adjustments[action])
        )

        if iface not in self.iface_qdisc_init:
            os.system(f"tc qdisc replace dev {iface} root handle 1: "
                      f"tbf rate 100mbit burst 150k latency 50ms")
            os.system(f"tc qdisc replace dev {iface} parent 1:1 handle 10: "
                      f"pfifo limit {new_limit}")
            self.iface_qdisc_init.add(iface)
        else:
            ret = os.system(f"tc qdisc replace dev {iface} parent 1:1 handle 10: "
                            f"pfifo limit {new_limit}")
            if ret != 0:
                self.logger.warning("qdisc update failed on %s", iface)

        self.current_queue_limit[key] = new_limit
        return new_limit

    # ── Layer 2: Reroute elephant via best end-to-end path ────────────────────
    def _apply_action_reroute(self, dp, port_no, elephant_flow) -> bool:
        """
        Redirect the elephant flow to the port leading to the best
        (lowest cumulative delay) alternate path to the destination.

        Upgrade 1 — Global Cooldown Check
        ----------------------------------
        Uses self.global_flow_decisions to enforce a network-wide
        REROUTE_COOLDOWN second block on re-rerouting the same (src,dst)
        flow regardless of which switch is requesting the reroute.

        This prevents:
          Switch A reroutes flow X -> Switch B reroutes flow X back ->
          Switch A reroutes flow X again -> ...

        The cooldown key is (src_ip, dst_ip) only — intentionally ignoring
        dpid so the block is truly global across all switches.
        """
        if elephant_flow is None:
            return False

        src_ip   = elephant_flow['src_ip']
        dst_ip   = elephant_flow['dst_ip']
        dpid     = dp.id
        now      = time.time()
        flow_key = (src_ip, dst_ip)   # global key — no dpid

        # ── Upgrade 1: global cooldown check ─────────────────────────────
        if flow_key in self.global_flow_decisions:
            elapsed = now - self.global_flow_decisions[flow_key]['timestamp']
            if elapsed < self.REROUTE_COOLDOWN:
                self.logger.debug(
                    "Cooldown active for %s->%s  (%.1fs remaining, "
                    "first rerouted by dpid=%d)",
                    src_ip, dst_ip,
                    self.REROUTE_COOLDOWN - elapsed,
                    self.global_flow_decisions[flow_key]['switch_dpid'],
                )
                return False

        best_port = self._find_best_alternate_port(dpid, port_no, dst_ip)
        if best_port is None:
            self.logger.debug("No alternate path: dpid=%d port=%d dst=%s",
                              dpid, port_no, dst_ip)
            return False

        parser = dp.ofproto_parser
        match  = parser.OFPMatch(
            eth_type=ether_types.ETH_TYPE_IP,
            ipv4_src=src_ip,
            ipv4_dst=dst_ip,
        )
        self.add_flow(dp,
                      priority=self.REROUTE_PRIORITY,
                      match=match,
                      actions=[parser.OFPActionOutput(best_port)],
                      idle_timeout=self.REROUTE_IDLE_TIMEOUT,
                      hard_timeout=self.REROUTE_HARD_TIMEOUT)

        # ── Upgrade 1: register global decision ──────────────────────────
        self.global_flow_decisions[flow_key] = {
            'timestamp':   now,
            'switch_dpid': dpid,
            'port':        best_port,
        }

        # ── Upgrade 3: increment rolling reroute count ────────────────────
        self._increment_reroute_count(dpid, port_no)

        self.logger.info(
            "REROUTE: dpid=%d  %s->%s  old_port=%d  best_port=%d",
            dpid, src_ip, dst_ip, port_no, best_port,
        )
        return True

    # ── Layer 3: Header-only sampling ─────────────────────────────────────────
    def _apply_action_sample(self, dp, port_no, elephant_flow) -> bool:
        """
        Install a 1-second, header-only sampling rule.
        hard_timeout=1 and max_len=128 keep the control channel safe.
        See class docstring for the bandwidth calculation.
        """
        if elephant_flow is None:
            return False

        src_ip  = elephant_flow['src_ip']
        dst_ip  = elephant_flow['dst_ip']
        dpid    = dp.id
        parser  = dp.ofproto_parser
        ofproto = dp.ofproto

        match = parser.OFPMatch(
            eth_type=ether_types.ETH_TYPE_IP,
            ipv4_src=src_ip,
            ipv4_dst=dst_ip,
        )
        self.add_flow(dp,
                      priority=self.SAMPLE_PRIORITY,
                      match=match,
                      actions=[parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                                      max_len=self.SAMPLE_MAX_LEN)],
                      idle_timeout=0,
                      hard_timeout=self.SAMPLE_HARD_TIMEOUT)

        self.logger.info("SAMPLE: dpid=%d  %s->%s  (%ds window, %dB headers)",
                         dpid, src_ip, dst_ip,
                         self.SAMPLE_HARD_TIMEOUT, self.SAMPLE_MAX_LEN)
        return True

    # ── Layer 3: ECMP split ────────────────────────────────────────────────────
    def _apply_action_ecmp(self, dp, port_no, second_flow, dpid) -> bool:
        """
        Move the second-largest flow via the best alternate path.
        The elephant flow stays on its current path to avoid TCP retransmits.
        The global cooldown also applies to ECMP to prevent the same
        second flow being bounced by multiple switches.
        """
        if second_flow is None:
            return False

        src_ip   = second_flow['src_ip']
        dst_ip   = second_flow['dst_ip']
        now      = time.time()
        flow_key = (src_ip, dst_ip)

        # Apply the same global cooldown for ECMP moves
        if flow_key in self.global_flow_decisions:
            elapsed = now - self.global_flow_decisions[flow_key]['timestamp']
            if elapsed < self.REROUTE_COOLDOWN:
                self.logger.debug("ECMP cooldown active for %s->%s", src_ip, dst_ip)
                return False

        best_port = self._find_best_alternate_port(dpid, port_no, dst_ip)
        if best_port is None:
            return False

        parser = dp.ofproto_parser
        match  = parser.OFPMatch(
            eth_type=ether_types.ETH_TYPE_IP,
            ipv4_src=src_ip,
            ipv4_dst=dst_ip,
        )
        self.add_flow(dp,
                      priority=self.REROUTE_PRIORITY,
                      match=match,
                      actions=[parser.OFPActionOutput(best_port)],
                      idle_timeout=self.REROUTE_IDLE_TIMEOUT,
                      hard_timeout=self.REROUTE_HARD_TIMEOUT)

        self.global_flow_decisions[flow_key] = {
            'timestamp':   now,
            'switch_dpid': dpid,
            'port':        best_port,
        }
        self._increment_reroute_count(dpid, port_no)

        self.logger.info("ECMP: dpid=%d  %s->%s  -> best_port=%d",
                         dpid, src_ip, dst_ip, best_port)
        return True

    # ═════════════════════════════════════════════════════════════════════════
    # TOPOLOGY HELPERS
    # ═════════════════════════════════════════════════════════════════════════

    def _has_alternate_path(self, dpid: int) -> bool:
        """O(1) check: does this switch have more than one outgoing edge?"""
        g = self.arp_handler.graph
        return dpid in g and g.out_degree(dpid) > 1

    def _find_best_alternate_port(self, dpid: int, current_port: int,
                                   dst_ip: str):
        """
        Return the local output port leading to the best (lowest cumulative
        link-delay) path to dst_ip that diverges from the current shortest path
        at the first hop.

        Uses nx.shortest_simple_paths() (Yen's algorithm) which yields paths
        in ascending total-weight order.  The first path whose first hop
        differs from the current shortest path is the global optimum — no
        further iteration is needed.

        Correctness example
        -------------------
        Current : dpid -[5]->  A -[100]-> dst   total = 105
        Alternate: dpid -[50]-> B -[10]->  dst   total =  60
        Decision : pick B (total 60), not A (first-link score would pick A)

        Returns None if no diverging alternate path exists.
        """
        g = self.arp_handler.graph

        dst_loc = self.arp_handler.get_host_location(dst_ip)
        if not dst_loc:
            return None

        dst_dpid, _ = dst_loc
        if dpid == dst_dpid:
            return None
        if not nx.has_path(g, dpid, dst_dpid):
            return None

        try:
            current_path     = nx.shortest_path(g, dpid, dst_dpid, weight='delay')
            current_next_hop = current_path[1] if len(current_path) > 1 else None
        except nx.NetworkXNoPath:
            return None

        try:
            for path in nx.shortest_simple_paths(g, dpid, dst_dpid, weight='delay'):
                if len(path) < 2:
                    continue
                first_hop = path[1]
                if first_hop == current_next_hop:
                    continue   # same first hop — skip

                edge_data = g[dpid].get(first_hop, {})
                best_port = edge_data.get('src_port')

                if best_port is not None:
                    path_delay = sum(
                        g[path[i]][path[i + 1]].get('delay', 0)
                        for i in range(len(path) - 1)
                    )
                    self.logger.info(
                        "BEST ALT PATH: dpid=%d  dst=%s  port=%d  "
                        "path=%s  cumulative_delay=%.1f",
                        dpid, dst_ip, best_port,
                        '->'.join(str(n) for n in path), path_delay,
                    )
                return best_port

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

        return None

    # ═════════════════════════════════════════════════════════════════════════
    # REWARD FUNCTION
    # ═════════════════════════════════════════════════════════════════════════

    def _calculate_reward(self, stats: dict, queue_limit: int, action: int) -> float:
        """
        Base reward (identical to v1) plus action-specific shaping.

        Base
        ----
        delay_reward        penalise queueing delay vs 0.5 ms target
        efficiency_penalty  penalise packet loss fraction
        drop_penalty        fast-signal direct drop punishment
        buffer_reward       prefer small stable queue sizes
        stability_bonus     bonus when both delay and drop are low

        Shaping
        -------
        REROUTE  +0.5 if queue_util > 0.7  (decisive under real congestion)
                 -0.2 otherwise            (discourage premature rerouting)
        SAMPLE   -0.1                      (small overhead cost)
        ECMP     +0.3 if queue_util > 0.6
                 -0.1 otherwise
        """
        queue_len   = stats['queue_length']
        packet_rate = stats['packet_rate']
        drop_rate   = stats['drop_rate']

        queue_delay_ms = (queue_len / 83333.0 * 1000.0
                          if packet_rate > 1 else 0.0)
        TARGET_DELAY   = 0.5
        delay_ratio    = min(queue_delay_ms / TARGET_DELAY, 2.5)
        delay_reward   = 1.0 - delay_ratio

        attempted_pps      = packet_rate + drop_rate
        efficiency         = (packet_rate / attempted_pps
                              if attempted_pps > 1 else 1.0)
        efficiency_penalty = -2.0 * (1.0 - efficiency)

        drop_probability = drop_rate / max(attempted_pps, 1)
        drop_penalty     = -3.0 * drop_probability

        if queue_limit <= 3:
            buffer_reward = 0.35
        elif queue_limit <= 10:
            buffer_reward = 0.10
        elif queue_limit <= 20:
            buffer_reward = -0.25
        else:
            buffer_reward = -0.60

        stability_bonus = (0.30
                           if queue_delay_ms < TARGET_DELAY
                           and drop_probability < 0.01
                           else 0.0)

        base = (1.8 * delay_reward + efficiency_penalty
                + drop_penalty + buffer_reward + stability_bonus)

        queue_util = queue_len / max(1, queue_limit)
        extra = 0.0
        if action == ACTION_REROUTE:
            extra = +0.5 if queue_util > 0.7 else -0.2
        elif action == ACTION_SAMPLE:
            extra = -0.1
        elif action == ACTION_ECMP_SPLIT:
            extra = +0.3 if queue_util > 0.6 else -0.1

        return base + extra

    # ═════════════════════════════════════════════════════════════════════════
    # UTILITY
    # ═════════════════════════════════════════════════════════════════════════

    def add_flow(self, dp, priority, match, actions, idle_timeout=0, hard_timeout=0):
        ofp  = dp.ofproto
        inst = [dp.ofproto_parser.OFPInstructionActions(
                    ofp.OFPIT_APPLY_ACTIONS, actions)]
        dp.send_msg(dp.ofproto_parser.OFPFlowMod(
            datapath=dp, priority=priority,
            match=match, instructions=inst,
            idle_timeout=idle_timeout, hard_timeout=hard_timeout,
        ))

    def _iface_name(self, dpid, port_no):
        return f"s{dpid}-eth{port_no}"

    def _get_qdisc_stats(self, iface):
        backlog_pkts = 0
        total_drops  = 0
        try:
            result = subprocess.run(
                ['tc', '-s', 'qdisc', 'show', 'dev', iface],
                capture_output=True, text=True, timeout=1.0,
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'dropped' in line:
                        parts = line.split()
                        for i, p in enumerate(parts):
                            if p == 'dropped' and i + 1 < len(parts):
                                try:
                                    total_drops = int(parts[i + 1].rstrip(','))
                                except ValueError:
                                    pass
                    if 'backlog' in line:
                        m = re.search(r'backlog\s+\S+\s+(\d+)p', line)
                        if m:
                            try:
                                backlog_pkts = int(m.group(1))
                            except ValueError:
                                pass
        except Exception:
            pass
        return backlog_pkts, total_drops

    def _update_intelligent_switches(self):
        valid = [d for d in self.datapaths if isinstance(d, int)]
        self.intelligent_switch_ids = (
            self.switch_selector.select_switches(valid) if valid else set()
        )

    # ═════════════════════════════════════════════════════════════════════════
    # LOGGING
    # ═════════════════════════════════════════════════════════════════════════

    def _log_step(self, dpid, port_no, action, action_name,
                  reward, stats, queue_limit):
        t = time.time() - self.start_time
        with open(self.metrics_log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                self.current_episode, f"{t:.3f}", dpid, port_no,
                action, action_name, queue_limit,
                f"{reward:.6f}",
                f"{stats['throughput']:.2f}",
                f"{stats['drop_rate']:.2f}",
                f"{stats['packet_rate']:.2f}",
                stats['queue_length'],
                f"{stats['queue_velocity']:.2f}",
                f"{stats['ewma_queue']:.2f}",
                f"{stats['max_flow_rate']:.1f}",
                stats['flow_count'],
                f"{stats['elephant_ratio']:.4f}",
                stats['reroute_count_last_10s'],
                f"{stats['throughput'] / 10000.0:.4f}",
                f"{stats['queue_length'] / max(1, queue_limit):.4f}",
                f"{self.agent.epsilon:.4f}",
                f"{stats['delay_ms']:.6f}",
            ])

    def _track(self, reward, action, delay, queue, vel, ewma):
        for lst, val in [
            (self.recent_rewards,     reward),
            (self.recent_actions,     action),
            (self.recent_delays,      delay),
            (self.recent_queue_sizes, queue),
            (self.recent_velocities,  vel),
            (self.recent_ewma,        ewma),
        ]:
            lst.append(val)
            if len(lst) > 1000:
                lst.pop(0)

    def _checkpoint(self):
        self.checkpoint_counter += 1
        fname = f"extended_rl_checkpoint_{self.checkpoint_counter}.pth"
        try:
            self.agent.save_model(fname)
            t   = time.time() - self.start_time
            avg = lambda lst: sum(lst[-1000:]) / len(lst[-1000:]) if lst else 0
            with open(self.checkpoint_log_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    self.checkpoint_counter, self.current_episode,
                    f"{t:.3f}",
                    f"{avg(self.recent_rewards):.6f}",
                    f"{avg(self.recent_delays):.6f}",
                    f"{avg(self.recent_queue_sizes):.2f}",
                    f"{avg(self.recent_velocities):.2f}",
                    f"{self.agent.epsilon:.4f}",
                    fname,
                ])
            self.logger.info("Checkpoint %d -> %s", self.checkpoint_counter, fname)
        except Exception as e:
            self.logger.error("Checkpoint failed: %s", e)

    def _log_per_diagnostics(self):
        t  = time.time() - self.start_time
        td = self.per_diagnostics['td_errors']
        if not td:
            return
        avg_td   = sum(td) / len(td)
        avg_prio = (sum(self.per_diagnostics['avg_priority']) /
                    len(self.per_diagnostics['avg_priority'])
                    if self.per_diagnostics['avg_priority'] else 0.0)
        buf_size = len(self.agent.memory.buffer)
        hi_ratio = sum(1 for e in td if e > avg_td) / len(td)
        with open(self.per_diagnostics_log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                f"{t:.3f}",
                f"{avg_td:.6f}", f"{max(td):.6f}", f"{min(td):.6f}",
                f"{avg_prio:.6f}",
                self.per_diagnostics['priority_updates'],
                buf_size,
                f"{hi_ratio:.4f}",
            ])
        self.logger.info(
            "PER @ %.1fh  td_avg=%.4f  buf=%d  hi_ratio=%.1f%%",
            t / 3600, avg_td, buf_size, hi_ratio * 100,
        )
        self.per_diagnostics.update({
            'td_errors': [], 'avg_priority': [], 'priority_updates': 0,
        })
