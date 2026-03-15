#!/usr/bin/env python3
"""
ArpHandler.py
=============
Ryu app context module — handles topology discovery, ARP proxy,
host location learning, and shortest-path flow installation.

Launched automatically as a _CONTEXTS dependency of the main controller:
    ryu-manager controller_extended.py

No standalone usage needed.
"""

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types, ipv4, arp
from ryu.topology import api as topo_api
from ryu.topology.api import get_all_switch, get_link, get_switch
from ryu.lib import hub

import networkx as nx


class ArpHandler(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(ArpHandler, self).__init__(*args, **kwargs)
        self.topology_api_app = self
        self.link_to_port    = {}   # (src_dpid, dst_dpid) -> (src_port, dst_port)
        self.link_delay      = {}   # (src_dpid, dst_dpid) -> delay_ms
        self.access_table    = {}   # (sw, port) -> (host_ip, host_mac)
        self.switch_port_table = {} # dpid -> set(port_no)
        self.access_ports    = {}   # dpid -> set(port_no)  host-facing
        self.interior_ports  = {}   # dpid -> set(port_no)  switch-facing
        self.graph           = nx.DiGraph()
        self.dps             = {}   # dpid -> datapath
        self.switches        = None
        self.discover_thread = hub.spawn(self._discover)

    # ──────────────────────────────────────────────────────────────────────────
    # Topology discovery (runs every 2 s)
    # ──────────────────────────────────────────────────────────────────────────

    def _discover(self):
        while True:
            self.get_topology(None)
            hub.sleep(2)

    def get_topology(self, ev):
        switch_list = get_all_switch(self)
        self.create_port_map(switch_list)
        self.switches = list(self.switch_port_table.keys())
        links = get_link(self.topology_api_app, None)
        self.create_interior_links(links)
        self.create_access_ports()
        self.get_graph()

    def create_port_map(self, switch_list):
        for sw in switch_list:
            dpid = sw.dp.id
            self.graph.add_node(dpid)
            self.dps[dpid] = sw.dp
            self.switch_port_table.setdefault(dpid, set())
            self.interior_ports.setdefault(dpid, set())
            self.access_ports.setdefault(dpid, set())
            for p in sw.ports:
                self.switch_port_table[dpid].add(p.port_no)

    def create_interior_links(self, link_list):
        for link in link_list:
            src, dst = link.src, link.dst
            self.link_to_port[(src.dpid, dst.dpid)] = (src.port_no, dst.port_no)
            if link.src.dpid in self.switches:
                self.interior_ports[link.src.dpid].add(link.src.port_no)
            if link.dst.dpid in self.switches:
                self.interior_ports[link.dst.dpid].add(link.dst.port_no)

    def create_access_ports(self):
        for sw in self.switch_port_table:
            self.access_ports[sw] = (
                self.switch_port_table[sw] - self.interior_ports[sw]
            )

    def get_graph(self):
        link_list = topo_api.get_all_link(self)
        for link in link_list:
            src_dpid  = link.src.dpid
            dst_dpid  = link.dst.dpid
            src_port  = link.src.port_no
            dst_port  = link.dst.port_no
            # Static propagation delay based on switch layer.
            # Fiber length does not change — only queuing delay is dynamic.
            # Randomizing here makes the MDP non-stationary and breaks convergence.
            #
            # Layer detection uses switch_selection_strategies.py convention:
            #   dpid < 2000  → Core   (1ms  — short backplane links)
            #   dpid < 3000  → Aggr   (2ms  — intra-pod links)
            #   dpid >= 3000 → Edge   (5ms  — host-facing links, longer runs)
            #
            # If your fat_tree.py uses sequential dpids (1,2,3,...) instead of
            # range-based ones, fall back to a uniform 1ms to keep the MDP
            # stationary — still correct, just loses layer differentiation.
            if (src_dpid, dst_dpid) not in self.link_delay:
                delay = self._static_link_delay(src_dpid, dst_dpid)
                self.link_delay[(src_dpid, dst_dpid)] = delay
                self.link_delay[(dst_dpid, src_dpid)] = delay
            self.graph.add_edge(src_dpid, dst_dpid,
                                src_port=src_port,
                                dst_port=dst_port,
                                delay=self.link_delay[(src_dpid, dst_dpid)])
        return self.graph

    def _static_link_delay(self, src_dpid, dst_dpid):
        """
        Assign a fixed propagation delay based on switch layer.
        Uses the higher-numbered dpid to classify the link type
        (links always go from higher layer downward in fat-tree).
        """
        higher = max(src_dpid, dst_dpid)
        if higher < 2000:
            return 1    # Core-to-Core (shouldn't exist, safety value)
        elif higher < 3000:
            return 2    # Core-to-Aggr
        else:
            return 5    # Aggr-to-Edge or Edge-to-host

    # ──────────────────────────────────────────────────────────────────────────
    # Packet handlers — called by the main controller's PacketIn handler
    # ──────────────────────────────────────────────────────────────────────────

    def arp_handler(self, datapath, in_port, pkt, arp_pkt):
        """Proxy ARP reply if destination is known; flood otherwise."""
        eth_pkt    = pkt.get_protocol(ethernet.ethernet)
        arp_src_ip = arp_pkt.src_ip
        mac        = arp_pkt.src_mac
        self.register_access_info(datapath.id, in_port, arp_src_ip, mac)

        ofproto = datapath.ofproto
        parser  = datapath.ofproto_parser
        dst_ip  = arp_pkt.dst_ip
        dst_mac = self.get_mac_by_ip(dst_ip)

        if dst_mac:
            # Proxy ARP reply — send directly back on in_port
            reply = packet.Packet()
            reply.add_protocol(ethernet.ethernet(
                ethertype=eth_pkt.ethertype, dst=mac, src=dst_mac))
            reply.add_protocol(arp.arp(
                opcode=arp.ARP_REPLY,
                src_mac=dst_mac, src_ip=dst_ip,
                dst_mac=mac,    dst_ip=arp_src_ip))
            reply.serialize()
            actions = [parser.OFPActionOutput(in_port)]
            out = parser.OFPPacketOut(
                datapath=datapath,
                buffer_id=ofproto.OFP_NO_BUFFER,
                in_port=ofproto.OFPP_CONTROLLER,
                actions=actions,
                data=reply.data)
            datapath.send_msg(out)
        else:
            # Unknown destination — flood
            actions = [parser.OFPActionOutput(ofproto.OFPP_FLOOD)]
            out = parser.OFPPacketOut(
                datapath=datapath,
                buffer_id=ofproto.OFP_NO_BUFFER,
                in_port=in_port,
                actions=actions,
                data=pkt.data)
            datapath.send_msg(out)

    def ip_handler(self, datapath, in_port, pkt, ip_pkt):
        """Learn host location from IP packets and install shortest-path flows."""
        eth_pkt  = pkt.get_protocol(ethernet.ethernet)
        src_ipv4 = ip_pkt.src
        src_mac  = eth_pkt.src
        if src_ipv4 not in ('0.0.0.0', '255.255.255.255'):
            self.register_access_info(datapath.id, in_port, src_ipv4, src_mac)

        dst_ipv4 = ip_pkt.dst
        src_loc  = self.get_host_location(src_ipv4)
        dst_loc  = self.get_host_location(dst_ipv4)

        if src_loc and dst_loc:
            src_dpid, _         = src_loc
            dst_dpid, dst_port  = dst_loc
            match = datapath.ofproto_parser.OFPMatch(
                eth_type=ether_types.ETH_TYPE_IP,
                ipv4_src=src_ipv4,
                ipv4_dst=dst_ipv4,
            )
            out_port = self.set_shortest_path(
                src_ipv4, dst_ipv4,
                datapath.id, dst_dpid, dst_port,
                match,
            )
            if out_port:
                actions = [datapath.ofproto_parser.OFPActionOutput(out_port)]
                out = datapath.ofproto_parser.OFPPacketOut(
                    datapath=datapath,
                    buffer_id=datapath.ofproto.OFP_NO_BUFFER,
                    in_port=in_port,
                    actions=actions,
                    data=pkt.data)
                datapath.send_msg(out)

    # ──────────────────────────────────────────────────────────────────────────
    # Host registration and lookup
    # ──────────────────────────────────────────────────────────────────────────

    def register_access_info(self, dpid, in_port, ip, mac):
        if in_port in self.access_ports.get(dpid, set()):
            if (dpid, in_port) in self.access_table:
                if self.access_table[(dpid, in_port)] == (ip, mac):
                    return
            self.access_table[(dpid, in_port)] = (ip, mac)
            self.logger.info("Host registered: IP=%s MAC=%s sw=%d port=%d",
                             ip, mac, dpid, in_port)

    def get_host_location(self, host_ip):
        """Return (dpid, port) for a host IP, or None."""
        for key, val in self.access_table.items():
            if val[0] == host_ip:
                return key
        return None

    def get_mac_by_ip(self, host_ip):
        """Return MAC for a host IP (for Proxy ARP), or None."""
        for val in self.access_table.values():
            if val[0] == host_ip:
                return val[1]
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Path installation
    # ──────────────────────────────────────────────────────────────────────────

    def set_shortest_path(self, ip_src, ip_dst, src_dpid, dst_dpid,
                          to_port_no, to_dst_match, pre_actions=[]):
        """
        Install shortest-path (by delay) flows across all hops.
        Returns the output port on src_dpid for the first packet-out.
        """
        if not nx.has_path(self.graph, src_dpid, dst_dpid):
            self.logger.info("No path from %d to %d", src_dpid, dst_dpid)
            return 0

        path = nx.shortest_path(self.graph, src_dpid, dst_dpid, weight="delay")

        IDLE_TIMEOUT = 300
        HARD_TIMEOUT = 600

        if len(path) == 1:
            dp      = self.get_datapath(src_dpid)
            actions = [dp.ofproto_parser.OFPActionOutput(to_port_no)]
            self.add_flow(dp, 10, to_dst_match, pre_actions + actions,
                          idle_timeout=IDLE_TIMEOUT, hard_timeout=HARD_TIMEOUT)
            return to_port_no
        else:
            self.install_path(to_dst_match, path, pre_actions,
                              idle_timeout=IDLE_TIMEOUT, hard_timeout=HARD_TIMEOUT)
            dst_dp  = self.get_datapath(dst_dpid)
            actions = [dst_dp.ofproto_parser.OFPActionOutput(to_port_no)]
            self.add_flow(dst_dp, 10, to_dst_match, pre_actions + actions,
                          idle_timeout=IDLE_TIMEOUT, hard_timeout=HARD_TIMEOUT)
            return self.graph[path[0]][path[1]]['src_port']

    def install_path(self, match, path, pre_actions=[], idle_timeout=0, hard_timeout=0):
        for index, dpid in enumerate(path[:-1]):
            port_no = self.graph[path[index]][path[index + 1]]['src_port']
            dp      = self.get_datapath(dpid)
            actions = [dp.ofproto_parser.OFPActionOutput(port_no)]
            self.add_flow(dp, 10, match, pre_actions + actions,
                          idle_timeout=idle_timeout, hard_timeout=hard_timeout)

    def add_flow(self, dp, p, match, actions, idle_timeout=0, hard_timeout=0):
        ofproto = dp.ofproto
        parser  = dp.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod  = parser.OFPFlowMod(datapath=dp, priority=p,
                                 idle_timeout=idle_timeout,
                                 hard_timeout=hard_timeout,
                                 match=match, instructions=inst)
        dp.send_msg(mod)

    # ──────────────────────────────────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────────────────────────────────

    def get_datapath(self, dpid):
        if dpid not in self.dps:
            switch = topo_api.get_switch(self, dpid)[0]
            self.dps[dpid] = switch.dp
        return self.dps[dpid]

    def get_switches(self):
        return self.switches

    def get_links(self):
        return self.link_to_port
