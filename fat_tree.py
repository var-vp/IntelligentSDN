#!/usr/bin/env python3
"""
fat_tree.py
===========
Fat-Tree k=4 topology for Mininet with Remote Ryu controller.

Usage:
    sudo python fat_tree.py
"""

from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink
from mininet.topo import Topo
import logging
import os

logging.basicConfig(filename='./fattree.log', level=logging.INFO)
logger = logging.getLogger(__name__)


class Fattree(Topo):
    def __init__(self, k, density):
        self.CoreSwitchList = []
        self.AggSwitchList  = []
        self.EdgeSwitchList = []
        self.HostList       = []
        self.pod            = k
        self.iCoreLayerSwitch = (k // 2) ** 2
        self.iAggLayerSwitch  = k * k // 2
        self.iEdgeLayerSwitch = k * k // 2
        self.density          = density
        self.iHost            = self.iEdgeLayerSwitch * density
        Topo.__init__(self)

    def createTopo(self):
        self.createCoreLayerSwitch(self.iCoreLayerSwitch)
        self.createAggLayerSwitch(self.iAggLayerSwitch)
        self.createEdgeLayerSwitch(self.iEdgeLayerSwitch)
        self.createHost(self.iHost)

    def _addSwitch(self, number, level, switch_list):
        for x in range(1, number + 1):
            PREFIX = str(level) + "00"
            if x >= 10:
                PREFIX = str(level) + "0"
            switch_list.append(self.addSwitch('s' + PREFIX + str(x)))

    def createCoreLayerSwitch(self, NUMBER):
        self._addSwitch(NUMBER, 1, self.CoreSwitchList)

    def createAggLayerSwitch(self, NUMBER):
        self._addSwitch(NUMBER, 2, self.AggSwitchList)

    def createEdgeLayerSwitch(self, NUMBER):
        self._addSwitch(NUMBER, 3, self.EdgeSwitchList)

    def createHost(self, NUMBER):
        for x in range(1, NUMBER + 1):
            PREFIX = "h00"
            if x >= 100:
                PREFIX = "h"
            elif x >= 10:
                PREFIX = "h0"
            self.HostList.append(self.addHost(PREFIX + str(x)))

    def createLink(self, bw_c2a=0.2, bw_a2e=0.1, bw_h2a=0.05):
        end = self.pod // 2

        # Core → Aggregation
        for x in range(0, self.iAggLayerSwitch, end):
            for i in range(0, end):
                for j in range(0, end):
                    self.addLink(
                        self.CoreSwitchList[i * end + j],
                        self.AggSwitchList[x + i],
                        bw=bw_c2a)

        # Aggregation → Edge
        for x in range(0, self.iAggLayerSwitch, end):
            for i in range(0, end):
                for j in range(0, end):
                    self.addLink(
                        self.AggSwitchList[x + i],
                        self.EdgeSwitchList[x + j],
                        bw=bw_a2e)

        # Edge → Host
        for x in range(0, self.iEdgeLayerSwitch):
            for i in range(0, self.density):
                self.addLink(
                    self.EdgeSwitchList[x],
                    self.HostList[self.density * x + i],
                    bw=bw_h2a)

    def set_ovs_protocol_13(self):
        self._set_ovs_protocol_13(self.CoreSwitchList)
        self._set_ovs_protocol_13(self.AggSwitchList)
        self._set_ovs_protocol_13(self.EdgeSwitchList)

    def _set_ovs_protocol_13(self, sw_list):
        for sw in sw_list:
            cmd = "sudo ovs-vsctl set bridge %s protocols=OpenFlow13" % sw
            os.system(cmd)


def _start_iperf_workload(net):
    """
    Launch a realistic mix of elephant and mice flows before dropping
    into the CLI.  Without traffic the RL agent records thousands of
    empty-network transitions that are useless for offline training.

    Elephant flows  — long-lived (120 s), high bandwidth: trigger congestion
    Mice flows      — short-lived (5 s),  low bandwidth:  realistic background
    """
    import random
    import time

    hosts = net.hosts
    if len(hosts) < 2:
        return

    info("*** Starting iperf workload (elephants + mice)\n")

    # ── Start iperf servers on all hosts ────────────────────────────────
    for h in hosts:
        h.cmd('iperf -s -p 5001 &')
    time.sleep(0.5)   # let servers bind

    # ── Elephant flows (3 pairs, 120 s, full bandwidth) ──────────────────
    n_elephants = 3
    pairs = random.sample(hosts, min(n_elephants * 2, len(hosts)))
    for i in range(0, len(pairs) - 1, 2):
        src, dst = pairs[i], pairs[i + 1]
        src.cmd(f'iperf -c {dst.IP()} -p 5001 -t 120 &')
        info(f"  Elephant: {src.name} → {dst.name}\n")

    time.sleep(1)

    # ── Mice flows (10 short bursts, staggered) ───────────────────────────
    for _ in range(10):
        src, dst = random.sample(hosts, 2)
        src.cmd(f'iperf -c {dst.IP()} -p 5001 -t 5 &')
        time.sleep(0.3)

    info("*** Workload running — elephant flows active for 120 s\n")
    info("*** Touching /tmp/start_training.flag to start RL agent\n")
    import os
    os.system('touch /tmp/start_training.flag')


def createTopo(pod, density, ip="127.0.0.1", port=6633,
               bw_c2a=0.2, bw_a2e=0.1, bw_h2a=0.05):
    topo = Fattree(pod, density)
    topo.createTopo()
    topo.createLink(bw_c2a=bw_c2a, bw_a2e=bw_a2e, bw_h2a=bw_h2a)

    net = Mininet(topo=topo, link=TCLink, controller=None, autoSetMacs=True)
    net.addController(
        'controller', controller=RemoteController,
        ip=ip, port=port)
    net.start()
    topo.set_ovs_protocol_13()

    _start_iperf_workload(net)
    CLI(net)
    net.stop()


if __name__ == '__main__':
    setLogLevel('info')
    if os.getuid() != 0:
        logger.error("Must be run as root (sudo)")
    else:
        createTopo(4, 13)
