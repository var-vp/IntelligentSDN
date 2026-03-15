#!/usr/bin/env python3
"""
Intelligent Switch Selection Strategies for RL-based AQM

This module provides research-sound methods for selecting which switches
should have RL-based queue management in a Fat-Tree datacenter network.

From a research perspective, strategic selection is crucial because:
1. Limited computational resources (can't run RL on all switches)
2. Different switches have different traffic characteristics
3. Impact varies by position in the network hierarchy

Strategies implemented:
- CORE: Select core layer switches (highest traffic aggregation)
- EDGE: Select edge layer switches (closest to hosts)
- AGG: Select aggregation layer switches (middle tier)
- BOTTLENECK: Select switches with highest observed traffic/drops
- HYBRID: Mix of core + high-traffic switches
- RANDOM: Baseline for comparison
"""

from collections import defaultdict
import random


class IntelligentSwitchSelector:
    """
    Selects which switches should have intelligent (RL-based) queue management.
    
    Fat-Tree naming convention:
    - Core switches: dpid starts with 1 (e.g., 1001, 1002, 1003, 1004)
    - Aggregation switches: dpid starts with 2 (e.g., 2001-2008)
    - Edge switches: dpid starts with 3 (e.g., 3001-3008)
    """
    
    def __init__(self, strategy='CORE', k=3, pod_size=4):
        """
        Args:
            strategy: Selection strategy (CORE, EDGE, AGG, BOTTLENECK, HYBRID, RANDOM)
            k: Number of switches to select
            pod_size: Fat-Tree pod size (typically 4)
        """
        self.strategy = strategy.upper()
        self.k = k
        self.pod_size = pod_size
        
        # For bottleneck detection
        self.traffic_stats = defaultdict(lambda: {'bytes': 0, 'drops': 0, 'samples': 0})
        
    def get_switch_layer(self, dpid):
        """Determine switch layer from DPID"""
        if dpid < 2000:
            return 'CORE'
        elif dpid < 3000:
            return 'AGG'
        else:
            return 'EDGE'
    
    def select_switches(self, available_dpids):
        """
        Select k switches from available switches based on strategy.
        
        Args:
            available_dpids: List of available datapath IDs
            
        Returns:
            set: Selected switch DPIDs
        """
        if not available_dpids:
            return set()
        
        if self.strategy == 'CORE':
            return self._select_core(available_dpids)
        elif self.strategy == 'EDGE':
            return self._select_edge(available_dpids)
        elif self.strategy == 'AGG':
            return self._select_agg(available_dpids)
        elif self.strategy == 'BOTTLENECK':
            return self._select_bottleneck(available_dpids)
        elif self.strategy == 'HYBRID':
            return self._select_hybrid(available_dpids)
        elif self.strategy == 'RANDOM':
            return self._select_random(available_dpids)
        else:
            # Default to CORE
            return self._select_core(available_dpids)
    
    def _select_core(self, dpids):
        """
        Strategy: CORE
        Rationale: Core switches aggregate traffic from all pods, making them
        natural bottlenecks. Controlling queues here has network-wide impact.
        
        Research justification:
        - Highest traffic concentration
        - Affects all pod-to-pod communication
        - Fewer switches to manage (k/2)^2 cores
        """
        core_switches = [d for d in dpids if d < 2000]
        
        if len(core_switches) <= self.k:
            return set(core_switches)
        
        # If more cores than k, take first k (deterministic)
        return set(sorted(core_switches)[:self.k])
    
    def _select_edge(self, dpids):
        """
        Strategy: EDGE
        Rationale: Edge switches directly connect to hosts, allowing fine-grained
        per-flow control and early congestion detection.
        
        Research justification:
        - First point of congestion detection
        - Can prevent congestion from propagating upward
        - Closest to traffic sources/sinks
        """
        edge_switches = [d for d in dpids if d >= 3000]
        
        if len(edge_switches) <= self.k:
            return set(edge_switches)
        
        # Select evenly across pods for coverage
        return set(sorted(edge_switches)[:self.k])
    
    def _select_agg(self, dpids):
        """
        Strategy: AGG (Aggregation)
        Rationale: Aggregation switches balance inter-pod and intra-pod traffic,
        making them strategic control points.
        
        Research justification:
        - Balance between core and edge
        - Handle both upward and downward traffic
        - Pod-level traffic aggregation
        """
        agg_switches = [d for d in dpids if 2000 <= d < 3000]
        
        if len(agg_switches) <= self.k:
            return set(agg_switches)
        
        return set(sorted(agg_switches)[:self.k])
    
    def _select_bottleneck(self, dpids):
        """
        Strategy: BOTTLENECK
        Rationale: Select switches with highest observed traffic or drops,
        dynamically adapting to actual network conditions.
        
        Research justification:
        - Data-driven selection
        - Adapts to actual traffic patterns
        - Focuses resources where congestion occurs
        
        Note: Requires traffic statistics to be updated via update_traffic_stats()
        """
        if not self.traffic_stats:
            # Fallback to CORE if no stats available yet
            return self._select_core(dpids)
        
        # Calculate congestion score for each switch
        scores = []
        for dpid in dpids:
            stats = self.traffic_stats.get(dpid)
            if stats and stats['samples'] > 0:
                # Score = normalized drops + normalized traffic
                avg_drops = stats['drops'] / max(stats['samples'], 1)
                avg_bytes = stats['bytes'] / max(stats['samples'], 1)
                score = avg_drops * 0.7 + avg_bytes * 0.3  # Weight drops more
                scores.append((dpid, score))
            else:
                scores.append((dpid, 0))
        
        # Select top k by score
        scores.sort(key=lambda x: x[1], reverse=True)
        selected = [dpid for dpid, _ in scores[:self.k]]
        return set(selected)
    
    def _select_hybrid(self, dpids):
        """
        Strategy: HYBRID
        Rationale: Combine core switches (strategic) with high-traffic switches
        (reactive) for comprehensive coverage.
        
        Research justification:
        - Best of both worlds: strategic + reactive
        - Core provides baseline coverage
        - Bottleneck detection adds adaptivity
        - More robust to varying traffic patterns
        """
        # Split k: half for core, half for bottlenecks
        k_core = max(1, self.k // 2)
        k_bottleneck = self.k - k_core
        
        # Select core switches
        core_switches = list(self._select_core(dpids))[:k_core]
        
        # Select bottleneck switches (excluding already selected cores)
        remaining = [d for d in dpids if d not in core_switches]
        
        if not self.traffic_stats:
            # If no stats, add some edge switches
            edge_switches = [d for d in remaining if d >= 3000]
            bottleneck_switches = sorted(edge_switches)[:k_bottleneck]
        else:
            # Use bottleneck detection on remaining switches
            old_k = self.k
            self.k = k_bottleneck
            bottleneck_set = self._select_bottleneck(remaining)
            self.k = old_k
            bottleneck_switches = list(bottleneck_set)
        
        return set(core_switches + bottleneck_switches)
    
    def _select_random(self, dpids):
        """
        Strategy: RANDOM
        Rationale: Baseline for research comparison. Random selection allows
        measuring the impact of strategic placement.
        
        Research justification:
        - Unbiased baseline
        - Tests whether strategy matters
        - Control experiment
        """
        if len(dpids) <= self.k:
            return set(dpids)
        
        return set(random.sample(dpids, self.k))
    
    def update_traffic_stats(self, dpid, bytes_delta, drops_delta):
        """
        Update traffic statistics for bottleneck detection.
        
        Args:
            dpid: Datapath ID
            bytes_delta: Bytes transmitted since last update
            drops_delta: Packets dropped since last update
        """
        self.traffic_stats[dpid]['bytes'] += bytes_delta
        self.traffic_stats[dpid]['drops'] += drops_delta
        self.traffic_stats[dpid]['samples'] += 1
    
    def get_strategy_info(self):
        """Return information about current strategy"""
        info = {
            'strategy': self.strategy,
            'k': self.k,
            'pod_size': self.pod_size,
            'description': self._get_strategy_description()
        }
        return info
    
    def _get_strategy_description(self):
        """Get description of current strategy"""
        descriptions = {
            'CORE': 'Core layer switches (highest aggregation)',
            'EDGE': 'Edge layer switches (closest to hosts)',
            'AGG': 'Aggregation layer switches (middle tier)',
            'BOTTLENECK': 'Switches with highest traffic/drops (adaptive)',
            'HYBRID': 'Mix of core switches + high-traffic switches',
            'RANDOM': 'Random selection (baseline)'
        }
        return descriptions.get(self.strategy, 'Unknown strategy')


def get_recommended_strategy(network_characteristics):
    """
    Recommend a strategy based on network characteristics.
    
    Args:
        network_characteristics: dict with keys like 'traffic_pattern', 'primary_goal'
        
    Returns:
        str: Recommended strategy name
    """
    traffic_pattern = network_characteristics.get('traffic_pattern', 'mixed')
    primary_goal = network_characteristics.get('primary_goal', 'balanced')
    
    # Research-based recommendations
    if primary_goal == 'latency':
        # For latency-critical apps, control edge switches (early detection)
        return 'EDGE'
    elif primary_goal == 'throughput':
        # For throughput, control core switches (aggregate bottlenecks)
        return 'CORE'
    elif primary_goal == 'fairness':
        # For fairness, control aggregation layer (pod-level balancing)
        return 'AGG'
    elif traffic_pattern == 'dynamic':
        # For dynamic patterns, use adaptive selection
        return 'BOTTLENECK'
    elif traffic_pattern == 'mixed':
        # For mixed workloads, use hybrid approach
        return 'HYBRID'
    else:
        # Default to core (most impactful)
        return 'CORE'


if __name__ == '__main__':
    # Example usage and testing
    print("Intelligent Switch Selection Strategies\n")
    print("=" * 60)
    
    # Simulate Fat-Tree with k=4
    # Core: 1001-1004, Agg: 2001-2008, Edge: 3001-3008
    all_switches = list(range(1001, 1005)) + list(range(2001, 2009)) + list(range(3001, 3009))
    
    strategies = ['CORE', 'EDGE', 'AGG', 'BOTTLENECK', 'HYBRID', 'RANDOM']
    
    for strategy in strategies:
        selector = IntelligentSwitchSelector(strategy=strategy, k=3, pod_size=4)
        
        # For BOTTLENECK, simulate some traffic stats
        if strategy == 'BOTTLENECK':
            selector.update_traffic_stats(1001, 1000000, 100)  # High traffic core
            selector.update_traffic_stats(3001, 800000, 200)   # High drops edge
            selector.update_traffic_stats(2005, 500000, 50)    # Medium agg
        
        selected = selector.select_switches(all_switches)
        info = selector.get_strategy_info()
        
        print(f"\nStrategy: {strategy}")
        print(f"Description: {info['description']}")
        print(f"Selected switches: {sorted(selected)}")
        print(f"Layers: {[selector.get_switch_layer(d) for d in sorted(selected)]}")
    
    print("\n" + "=" * 60)
    print("\nRecommendations:")
    scenarios = [
        {'traffic_pattern': 'stable', 'primary_goal': 'throughput'},
        {'traffic_pattern': 'bursty', 'primary_goal': 'latency'},
        {'traffic_pattern': 'dynamic', 'primary_goal': 'balanced'},
        {'traffic_pattern': 'mixed', 'primary_goal': 'fairness'},
    ]
    
    for scenario in scenarios:
        rec = get_recommended_strategy(scenario)
        print(f"Scenario: {scenario} → Recommended: {rec}")