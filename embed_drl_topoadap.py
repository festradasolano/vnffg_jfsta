from __future__ import annotations

from embedding import Embedding
from embed_drl import DRLEmbedding, DRLEnv, DRLRun, MAX_HOPS
from substrate import Network
from vnfr import VNFR, MIN_ACCEPTABLE_THROUGHPUT

import math
import numpy as np
from typing import Any


class DRLTopoAdapEmbedding(DRLEmbedding):
    """
    TODO
    """

    def __init__(self, network: Network, vnfr: VNFR, node_pair_paths: dict[tuple, str], max_hops: int = MAX_HOPS, seed: int = None):
        """
        TODO
        """
        super().__init__(network, vnfr, node_pair_paths, max_hops, seed)
        self.usable_vnfs = vnfr.dependants[self.NO_DEPENDENCY].copy()

        self.source_node = vnfr.source_node
        self.ideal_rate_to_alloc = vnfr.data_rate
        self.actual_rate_to_alloc = vnfr.data_rate

    def embed_vnf_node_path(self, vnf_id: str, node_id: str, path: tuple[tuple, int]) -> bool:
        """
        TODO
        """
        # If no VNF, check all VNFS have been embedded and hosting node is target node
        if vnf_id == self.VNFR_TARGET_KEY:
            if len(self.embeded_vnfs) < len(self.vnfr.vnfs):
                # print("No VNF but there are still {vnfs_to_embed} VNFs to be embedded".format(vnfs_to_embed=(len(self.vnfr.vnfs)-len(self.embeded_vnfs))))
                return False
            elif node_id != self.vnfr.target_node:
                # print("No VNF but hosting node {hn} is not target node {tn}".format(hn=node_id, tn=self.vnfr.target_node))
                return False
        # Otherwise, check VNF can be used (wrt VNF dependency graph)
        elif vnf_id not in self.usable_vnfs:
            # print("VNF {v} is not in the usable VNFs {uv}".format(v=vnf_id, uv=self.usable_vnfs))
            return False
        
        # Check hosting node can host VNFs
        if node_id not in self.vnf_capable_node_ids:
            # print("Hosting node {hn} is not in the VNF capable nodes {vcp}".format(hn=node_id, vcp=self.vnf_capable_node_ids))
            return False
        
        # If no path, check source node is the same hosting node
        if (path == None) and (self.source_node != node_id):
            # print("No path but source node {sn} is not the same hosting node {hn}".format(sn=self.source_node, hn=node_id))
            return False
        
        # Get links in the path, if any
        path_links = []  # initialize to no path
        if path != None:
            # Decode path, check is between source and hosting node, and get path links
            (path_node_pair, path_node_pair_idx) = path
            if (self.source_node not in path_node_pair) or (node_id not in path_node_pair):
                # print("Path {p} is not between source node {sn} and hosting node {hn}".format(p=path, sn=self.source_node, hn=node_id))
                return False
            path_links = self.node_pair_paths[path_node_pair][path_node_pair_idx]
        
        # Embed source rate in links
        allocated_rate, tx_delay = self.embed_vnf_path(node_id, path_links, self.network, vnf_id, self.actual_rate_to_alloc)
        self.delay += tx_delay

        # Check allocated source rate
        min_rate_to_alloc = self.ideal_rate_to_alloc * MIN_ACCEPTABLE_THROUGHPUT
        if allocated_rate < min_rate_to_alloc:
            # print("Allocated rate {ar} does not meet the minimum rate to allocate {mr}".format(ar=allocated_rate, mr=min_rate_to_alloc))
            return False

        # Embed VNF, if any
        if vnf_id != self.VNFR_TARGET_KEY:
            # Compute VNF CPU demand, using actual source rate
            vnf = self.vnfr.vnfs[vnf_id]
            vnf_cpu_demand = math.ceil(vnf.cpu_demand_per_bw * allocated_rate / 1_000)

            # Compute available CPU of hosting node, using embedded VNFs, and check it has available CPU
            node_cpu_available = self._compute_node_available_cpu(node_id)
            if node_cpu_available <= 0:
                # print("Unavailable CPU {n_cpu} of hosting node {hn}".format(n_cpu=node_cpu_available, hn=node_id))
                return False
            
            # Check hosting node supports VNF CPU demand
            if node_cpu_available < vnf_cpu_demand:
                # print("Available CPU {n_cpu} of hosting node {hn} is less than VNF CPU demand {v_cpu}".format(n_cpu=node_cpu_available, hn=node_id, v_cpu=vnf_cpu_demand))
                return False
            
            # Embed VNF in hosting node and update VNFFG and list of usable VNFs
            self.embed_vnf_node(vnf_id, node_id, vnf_cpu_demand)
            self.__update_vnffg(vnf_id)
            self.embeded_vnfs.add(vnf_id)
            self.__update_usable_vnfs(vnf_id)

            # Check if node can still host VNFs
            node_cpu_available -= vnf_cpu_demand
            self._update_vnf_capable_nodes(node_id, node_cpu_available)

            # Add VNF processing delay
            self.delay += vnf.processing_delay

            # Update source node and rates
            self.source_node = node_id
            self.ideal_rate_to_alloc *= vnf.ratio_out2in
            self.actual_rate_to_alloc = allocated_rate * vnf.ratio_out2in
        
        # Check embedding delay
        if self.delay > self.vnfr.max_delay:
            # print("Current delay {d} is greater than the maximum delay {md}".format(d=self.delay, md=self.vnfr.max_delay))
            return False
        
        self.throughput = allocated_rate
        return True
    
    def __update_usable_vnfs(self, vnf_id: str):
        """
        TODO
        """
        # Remove VNF from list of usable VNFs
        vnf_idx = self.usable_vnfs.index(vnf_id)
        self.usable_vnfs.pop(vnf_idx)
        # Add VNF dependants to list of usable VNFs, if any
        if vnf_id in self.vnfr.dependants:
            self.usable_vnfs.extend(self.vnfr.dependants[vnf_id])
    
    def __update_vnffg(self, vnf_id: str):
        """
        TODO
        """
        self.vnffg.append(vnf_id)


class DRLTopoAdapEnv(DRLEnv):
    """
    TODO
    """
    MAX_VNF_INSTANCES = 1

    def __init__(self, m1: Embedding, network: Network, vnfr: VNFR, max_vnf_instances: int = MAX_VNF_INSTANCES, max_hops: int = MAX_HOPS, seed: int = None):
        """
        TODO
        """
        super().__init__(m1, network, vnfr, max_vnf_instances, max_hops, seed)
    
    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        """
        TODO
        """
        super().reset()

        # Reset M2 by initializing it
        self.m2 = DRLTopoAdapEmbedding(network=self.network, vnfr=self.vnfr, node_pair_paths=self.node_pair_paths, max_hops=self.max_hops, seed=self.seed)

        # Initialize original auxiliary set with VNF replicas
        # self._auxiliary = np.full(self._len_auxiliary, self._max_vnf_instances)

        # Return observation and info
        return self._get_obs(), self._get_info(action_mask=self.__get_action_mask())

    def step(self, action_idx: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        TODO
        """
        # Initialize reward and flags
        # reward = -math.inf
        # reward = -100
        reward = self.reward - self.reward_coefficient  # M2 crash reward
        terminated = True
        truncated = False

        # Validate action index range
        if action_idx < 0 or action_idx > (self._len_action - 1):
            print("ERROR: Action index {ai} is out of the action range [0, {la}]".format(ai=action_idx, la=self._len_action - 1))
            reward = -math.inf
            truncated = True
            return self._get_obs(), reward, terminated, truncated, self._get_info()
        
        # Decode action index to action space (VNF, node, path)
        action = self._action_idx_to_action_space(action_idx)
        action_vnf_idx = action[0]
        action_node_idx = action[1] + 1
        action_path_idx = action[2]

        # Update M2 by appending action
        if self.i >= self._len_mapped_hosts_paths:
            print("ERROR: Iteration {i} is out of bounds for the mapped hosts and paths; max iteration is {mi}".format(i=self.i, mi=self._len_mapped_hosts_paths - 1))
            reward = -math.inf
            truncated = True
            return self._get_obs(), reward, terminated, truncated, self._get_info()
        elif self.i < self._len_mapped_vnfs:
            self._mapped_vnfs[self.i] = action_vnf_idx
        # self._mapped_vnfs[self.i] = action_vnf_idx
        self._hosting_nodes[self.i] = action_node_idx
        self._inputpaths[self.i] = action_path_idx

        # Update auxiliary set by decreasing remaining replicas of selected VNF, if any
        if action_vnf_idx != 0:
            aux_vnf_idx = action_vnf_idx - 1
            if self._auxiliary[aux_vnf_idx] <= 0:
                # print("Action VNF {v} has {r} replicas in the auxiliary set".format(v=action_vnf_idx, r=self._auxiliary[aux_vnf_idx]))
                return self._get_obs(), reward, terminated, truncated, self._get_info()
            self._auxiliary[aux_vnf_idx] -= 1

        # Decode action space to action VNF, node, and path
        action_vnf = self.m2.VNFR_TARGET_KEY if action_vnf_idx == 0 else "f" + str(action_vnf_idx)
        action_node = "n" + str(action_node_idx)
        if action_path_idx == 0:
            action_path = None
        elif self.m2.source_node == action_node:
            # print("Action path {p} but source node {sn} is the same action node {hn}".format(p=action_path_idx, sn=self.m2.source_node, hn=action_node))
            return self._get_obs(), reward, terminated, truncated, self._get_info()
        else:
            node_pair = (self.m2.source_node, action_node)
            len_node_pair_paths = len(self.node_pair_paths[node_pair])
            if action_path_idx > len_node_pair_paths:
                # print("Action path {p} does not exist for node pair {np}; max action path is {l}".format(p=action_path_idx, np=node_pair, l=len_node_pair_paths))
                return self._get_obs(), reward, terminated, truncated, self._get_info()
            action_path = (node_pair, action_path_idx - 1)
        
        # Embed VNF, node, and path
        if not self.m2.embed_vnf_node_path(action_vnf, action_node, action_path):
            return self._get_obs(), reward, terminated, truncated, self._get_info()
        
        # Calculate reward using modification cost from M1 to M2
        cost = self.m2.compute_modification_cost(self.m1, self.m1_resource_cost, self.network)
        self.reward = -cost

        # If completed M2, end episode and report M2 as candidate
        if action_vnf == self.m2.VNFR_TARGET_KEY:
            reward = self.reward + self.reward_coefficient  # M2 candidate reward
            return self._get_obs(), reward, terminated, truncated, self._get_info(is_candidate_m2=True, modification_cost=cost)
        
        # Validate one action exists in the mask, if any
        action_mask = self.__get_action_mask()
        if action_mask is not None:
            len_valid_actions = len(np.nonzero(action_mask)[0])
            if len_valid_actions <= 0:
                # print("Action mask provided {a} valid actions for the next state".format(a=len_valid_actions))
                return self._get_obs(), reward, terminated, truncated, self._get_info()

        # Otherwise, end iteration and return observation, reward, and action mask
        reward = self.reward
        terminated = False
        self.i += 1
        return self._get_obs(), reward, terminated, truncated, self._get_info(action_mask=action_mask)

    def __get_action_mask(self) -> np.ndarray:
        """
        TODO
        """
        source_node = self.m2.source_node

        # Initialize action mask to zeros
        action_mask = np.full(self._len_action, 0, dtype=np.int8)

        # If VNFG is complete, update action mask by adding paths to target node (no VNF)
        if len(self.m2.vnffg) >= self._len_auxiliary:
            action_vnf_idx = 0
            target_node = self.vnfr.target_node
            action_node_idx = self.get_idx(target_node) - 1

            # If source and target nodes are the same, update action mask by adding target node and no path
            if source_node == target_node:
                action_path_idx = 0
                action_idx = self._action_space_to_action_idx(action_vnf_idx, action_node_idx, action_path_idx)
                action_mask[action_idx] = 1
                return action_mask
            
            # Otherwise, update action mask by adding target node and paths from source node
            node_pair = (source_node, target_node)
            len_node_pair_paths = len(self.node_pair_paths[node_pair])
            for action_path_idx in range(1, len_node_pair_paths + 1):
                action_idx = self._action_space_to_action_idx(action_vnf_idx, action_node_idx, action_path_idx)
                action_mask[action_idx] = 1
            return action_mask

        # Update action mask by adding VNF capable nodes reachable from source node and the paths from source node, for each usable VNF
        for node_id in self.m2.vnf_capable_node_ids:
            action_node_idx = self.get_idx(node_id) - 1
            
            # If source node is VNF capable, update action mask by adding source node and no path
            if node_id == source_node:
                action_path_idx = 0
                # Update for each usable VNF
                self.__update_action_mask_for_usable_VNFs(action_mask, action_node_idx, action_path_idx)
            
            # For each node neighbor of source node, update action mask by adding node and paths from source node
            elif node_id in self.neighbors_per_node[source_node]:
                node_pair = (source_node, node_id)
                len_node_pair_paths = len(self.node_pair_paths[node_pair])
                for action_path_idx in range(1, len_node_pair_paths + 1):
                    # Update for each usable VNF
                    self.__update_action_mask_for_usable_VNFs(action_mask, action_node_idx, action_path_idx)
        
        return action_mask
    
    def __update_action_mask_for_usable_VNFs(self, action_mask: np.ndarray, action_node_idx: int, action_path_idx: int):
        """
        TODO
        Update action mask for each usable VNF
        """
        for vnf_id in self.m2.usable_vnfs:
            action_vnf_idx = self.get_idx(vnf_id)
            action_idx = self._action_space_to_action_idx(action_vnf_idx, action_node_idx, action_path_idx)
            action_mask[action_idx] = 1


class DRLTopoAdapRun(DRLRun):
    """
    TODO
    """

    def __init__(self, m1: Embedding, network: Network, vnfr: VNFR, seed: int = None, experiment: int = 0):
        """
        TODO
        """
        # Initialize environment
        env = DRLTopoAdapEnv(m1=m1, network=network, vnfr=vnfr, seed=seed)

        # Build name of file to write results
        filename = "{0}_{1}vnfs_seed{2}_exp{3}.csv".format(self.__class__.__name__, str(len(vnfr.vnfs)), str(seed), str(experiment))

        # Call superclass constructor
        super().__init__(env, filename, seed)

        # # Initialize environment to obtain number of states and actions
        # self.env = DRLTopoAdapEnv(m1=m1, network=network, vnfr=vnfr, seed=seed)
        # len_state = len(self.env.observation_space.nvec)
        # len_action = np.prod(self.env.action_space.nvec)

        # # Build Q network and Target network
        # self.qnet = DQN(len_state, len_action, seed=seed)
        # self.target_net = DQN(len_state, len_action, seed=seed)
        # self.target_net.copy_weights(self.qnet)

        # # Build name of file to write results
        # self.results_filename = "{0}_{1}vnfs_seed{2}.csv".format(self.__class__.__name__, str(len(vnfr.vnfs)), str(seed))
