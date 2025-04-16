from __future__ import annotations

from embedding import Embedding, EmbeddingRate
from embed_drl import DRLEmbedding, DRLEnv, DRLRun, MAX_HOPS
from substrate import Network
from vnfr import VNFR

import math
import numpy as np
from typing import Any


class DRLFuncScalingEmbedding(DRLEmbedding):
    """
    TODO
    """

    def __init__(self, network: Network, vnfr: VNFR, node_pair_paths: dict[tuple, str], vnffg: list, max_hops: int = MAX_HOPS, seed: int = None):
        """
        TODO
        """
        super().__init__(network, vnfr, node_pair_paths, max_hops, seed)
        self.vnffg = vnffg
        self.vnffg_idx = 0
        self.usable_vnfs = [self.vnffg[self.vnffg_idx]]

        self.source_nodes: list[str] = [vnfr.source_node]
        self.source_rates: dict[str, EmbeddingRate] = {
            vnfr.source_node: EmbeddingRate(vnfr.data_rate)
        }
        self.next_src_nodes: list[str] = list()
        self.next_src_rates: dict[str, EmbeddingRate] = dict()
        self.ongoing_vnf_id = None
        self.ongoing_max_tx_delay = 0
        self.target_embeds_counter = 0

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
        
        # If no path, check hosting node is in the source nodes
        if (path == None) and (node_id not in self.source_nodes):
            # print("No path but hosting node {hn} is not in the source nodes {sn}".format(hn=node_id, sn=self.source_nodes))
            return False
        
        # Get links and source in the path, if any
        path_links = []  # initialize to no path
        path_node_src = node_id  # initialize to hosting node
        if path != None:
            # Decode path, check is between hosting node and a source node, and get path links
            path_node_pair, path_node_pair_idx = path
            if (path_node_pair[0] not in self.source_nodes) or (path_node_pair[1] != node_id):
                # print("Path {p} is not between hosting node {hn} and one of the source nodes {sn}".format(p=path, hn=node_id, sn=self.source_nodes))
                return False
            path_links = self.node_pair_paths[path_node_pair][path_node_pair_idx]
            path_node_src = path_node_pair[0]
        
        # If embedding same VNF, check no duplicated source node and hosting node
        if (vnf_id == self.ongoing_vnf_id) and (self.source_rates[path_node_src].allocated_rate > 0) and (node_id in self.next_src_nodes):
            # print("Pair source node {sn} and hosting node {hn} have been already used for embedding VNF {v}".format(sn=path_node_src, hn=node_id, v=vnf_id))
            return False
        
        # Embed source rate in links, update source allocated rate, and compare transmission delay
        rate_to_allocate = self.source_rates[path_node_src].get_missing_rate_to_alloc()
        allocated_rate, tx_delay = self.embed_vnf_path(node_id, path_links, self.network, vnf_id, rate_to_allocate)
        self.source_rates[path_node_src].add_allocated_rate(allocated_rate)
        if tx_delay > self.ongoing_max_tx_delay:
            self.ongoing_max_tx_delay = tx_delay  # TODO add when finished for target

        # Embed VNF, if any
        vnf_processing_delay = 0
        if vnf_id != self.VNFR_TARGET_KEY:
            # Compute total allocated rate and VNF allocated CPU by using previous resource allocation, if any
            total_allocated_rate = allocated_rate
            vnf_allocated_node_cpu = 0
            if node_id in self.next_src_nodes:
                total_allocated_rate += self.next_src_rates[node_id].rate_to_alloc
                vnf_allocated_node_cpu = self.allocated_node_cpus[node_id][vnf_id]

            # Compute VNF CPU demand, using total allocated rate and VNF allocated CPU
            vnf = self.vnfr.vnfs[vnf_id]
            vnf_cpu_demand = math.ceil(vnf.cpu_demand_per_bw * total_allocated_rate / 1_000) - vnf_allocated_node_cpu
            vnf_processing_delay = vnf.processing_delay

            # Check if any VNF CPU demand
            # if vnf_cpu_demand > 0:
            #     pass

            # Compute available CPU of hosting node, using embedded VNFs, and check it has available CPU
            node_cpu_available = self._compute_node_available_cpu(node_id)
            if node_cpu_available <= 0:
                # print("Unavailable CPU {n_cpu} of hosting node {hn}".format(n_cpu=node_cpu_available, hn=node_id))
                return False
            
            # Check hosting node supports VNF CPU demand
            if node_cpu_available < vnf_cpu_demand:
                # print("Available CPU {n_cpu} of hosting node {hn} is less than VNF CPU demand {v_cpu}".format(n_cpu=node_cpu_available, hn=node_id, v_cpu=vnf_cpu_demand))
                return False
            
            # Embed VNF in hosting node and update VNFFG
            self.embed_vnf_node(vnf_id, node_id, vnf_cpu_demand)

            # Check if node can still host VNFs
            node_cpu_available -= vnf_cpu_demand
            self._update_vnf_capable_nodes(node_id, node_cpu_available)

        # If target embedding, check allocated rate, count embeddings, and update throughput
        else:
            if allocated_rate < rate_to_allocate:
                # print("Allocated rate {ar} to target node does not meet the minimum rate to allocate {mr}".format(ar=allocated_rate, mr=rate_to_allocate))
                return False
            self.target_embeds_counter += 1
            self.throughput += allocated_rate
        
        # Add allocated rate to next layer of source nodes and rates
        if node_id not in self.next_src_nodes:
            self.next_src_nodes.append(node_id)
            self.next_src_rates[node_id] = EmbeddingRate(allocated_rate)
        else:
            self.next_src_rates[node_id].add_rate_to_alloc(allocated_rate)

        # If different ongoing VNF, update list of usable VNFs, ongoing VNF, and VNF processing delay
        if vnf_id != self.ongoing_vnf_id:
            self.embeded_vnfs.add(vnf_id)
            self.__update_usable_vnfs(vnf_id)
            self.ongoing_vnf_id = vnf_id
            self.delay += vnf_processing_delay
        
        # Check embedding delay
        if (self.delay + self.ongoing_max_tx_delay) > self.vnfr.max_delay:
            # print("Current delay {d} is greater than the maximum delay {md}".format(d=self.delay, md=self.vnfr.max_delay))
            return False
        
        return True
    
    def end_vnf_embedding(self) -> bool:
        """
        TODO
        """
        # Check allocation of minimum rate from current source nodes
        for source_node in self.source_nodes:
            if not self.source_rates[source_node].is_rate_allocated():
                source_rate = self.source_rates[source_node]
                # print("Rate to allocate {ra} from source node {sn} has not been allocated: is allocated {ia} and allocated rate {ar}".format(ra=source_rate.rate_to_alloc, sn=source_node, ia=source_rate.is_allocated, ar=source_rate.allocated_rate))
                return False
        
        # Add ongoing max transmission delay and reset it
        self.delay += self.ongoing_max_tx_delay
        self.ongoing_max_tx_delay = 0

        return True

    def next_vnf_embedding(self) -> bool:
        """
        TODO
        """
        # Validate current source nodes
        if not self.end_vnf_embedding():
            return False
        
        # Apply VNF data rate ratio to next source rates
        ongoing_vnf = self.vnfr.vnfs[self.ongoing_vnf_id]
        for next_src_node in self.next_src_nodes:
            self.next_src_rates[next_src_node].apply_ratio_out2in(ongoing_vnf.ratio_out2in)
        
        # Move to next layer of source nodes and rates
        self.source_nodes = self.next_src_nodes.copy()
        self.source_rates = self.next_src_rates.copy()
        self.next_src_nodes = list()
        self.next_src_rates = dict()

        return True

    def __update_usable_vnfs(self, vnf_id: str):
        """
        TODO
        """
        # Remove ongoing VNF, if not None
        if self.ongoing_vnf_id is not None:
            ongoing_vnf_idx = self.usable_vnfs.index(self.ongoing_vnf_id)
            self.usable_vnfs.pop(ongoing_vnf_idx)
        
        # Add next VNF to list of usable VNFs, if any
        self.vnffg_idx += 1
        if self.vnffg_idx < len(self.vnffg):
            self.usable_vnfs.append(self.vnffg[self.vnffg_idx])


class DRLFuncScalingEnv(DRLEnv):
    """
    TODO
    """
    MAX_VNF_INSTANCES = 3

    def __init__(self, m1: Embedding, network: Network, vnfr: VNFR, max_vnf_instances: int = MAX_VNF_INSTANCES, max_hops: int = MAX_HOPS, seed: int = None):
        """
        TODO
        """
        super().__init__(m1, network, vnfr, max_vnf_instances, max_hops, seed)
        # self.reward_coefficient = VNF_LICENSE_COST
    
    # def action_sample(self, mask: np.ndarray = None) -> int:
    #     """
    #     -------------------------------------------------------------------------------
    #     TODO DELETE
    #     -------------------------------------------------------------------------------
    #     """
    #     # TODO delete
    #     # actions = [
    #     #     (1, 8, 1),
    #     #     (1, 6, 1),
    #     #     (2, 5, 2),
    #     #     (2, 5, 9),
    #     #     (3, 5, 0),
    #     #     (0, 1, 2)
    #     # ]
    #     # actions = [
    #     #     (1, 7, 0),
    #     #     (2, 7, 0),
    #     #     (3, 6, 1),
    #     #     (3, 5, 1),
    #     #     (0, 1, 7),
    #     #     (0, 1, 9)
    #     # ]
    #     actions = [
    #         (1, 7, 0),
    #         (2, 7, 0),
    #         (3, 7, 0),
    #         (0, 1, 3)
    #     ]
    #     # vnf_idx, node_idx, path_idx = actions[0]
    #     vnf_idx, node_idx, path_idx = actions[self.i]
    #     return self._action_space_to_action_idx(vnf_idx, node_idx, path_idx)
    
    # def reset(self, vnf_instances: int) -> tuple[np.ndarray, dict[str, Any]]:
    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        """
        TODO
        """
        super().reset()

        # Reset M2 by initializing it
        self.m2 = DRLFuncScalingEmbedding(network=self.network, vnfr=self.vnfr, node_pair_paths=self.node_pair_paths, vnffg=self.m1.vnffg, max_hops=self.max_hops, seed=self.seed)

        # Return observation and info
        return self._get_obs(), self._get_info(action_mask=self.__get_action_mask())

    def step(self, action_idx: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        TODO
        """
        # Initialize reward and flags
        # reward = -1 * VNF_LICENSE_COST * (self._len_auxiliary * self._max_vnf_instances)
        # reward = -1.5 * VNF_LICENSE_COST * np.count_nonzero(self._mapped_vnfs)
        reward = self.reward - (2 * self.reward_coefficient)  # M2 crash reward
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

        # Update auxiliary set with selected VNF, if any
        if action_vnf_idx != 0:
            # Decrease remaining replicas of selected VNF
            aux_vnf_idx = action_vnf_idx - 1
            if self._auxiliary[aux_vnf_idx] <= 0:
                # print("Action VNF {v} has {r} replicas in the auxiliary set".format(v=action_vnf_idx, r=self._auxiliary[aux_vnf_idx]))
                return self._get_obs(), reward, terminated, truncated, self._get_info()
            self._auxiliary[aux_vnf_idx] -= 1
        
        # Decode action VNF and TODO
        action_vnf = self.m2.VNFR_TARGET_KEY if action_vnf_idx == 0 else "f" + str(action_vnf_idx)
        if (self.m2.ongoing_vnf_id is not None) and (action_vnf != self.m2.ongoing_vnf_id):
            ongoing_vnf_idx = self.get_idx(self.m2.ongoing_vnf_id) - 1
            self._auxiliary[ongoing_vnf_idx] = 0
            if not self.m2.next_vnf_embedding():
                return self._get_obs(), reward, terminated, truncated, self._get_info()
        
        # Decode action node and path
        action_node = "n" + str(action_node_idx)
        if action_path_idx == 0:
            action_path = None
        else:
            # Decode path source node index and path index
            path_src_node_idx = (action_path_idx - 1) // self.max_paths_per_node_pair
            node_pair_path_idx = (action_path_idx - 1) % self.max_paths_per_node_pair

            # Check path source node index exists in list of source nodes
            if path_src_node_idx > (len(self.m2.source_nodes) - 1):
                # print("Action path source node index {sni} is out of bounds for source nodes {sn}".format(sni=path_src_node_idx, sn=self.m2.source_nodes))
                return self._get_obs(), reward, terminated, truncated, self._get_info()
            
            # Get path source node and check is not the same hosting node
            path_src_node = self.m2.source_nodes[path_src_node_idx]
            if path_src_node == action_node:
                # print("Action path {p} but source node {sn} is the same action node {hn}".format(p=action_path_idx, sn=path_src_node, hn=action_node))
                return self._get_obs(), reward, terminated, truncated, self._get_info()
            
            # Get node pair path, if exists
            node_pair = (path_src_node, action_node)
            max_node_pair_path_idx = len(self.node_pair_paths[node_pair]) - 1
            if node_pair_path_idx > max_node_pair_path_idx:
                # print("Action path index {p} does not exist for node pair {np}; max node pair path index is {mi}".format(p=node_pair_path_idx, np=node_pair, mi=max_node_pair_path_idx))
                return self._get_obs(), reward, terminated, truncated, self._get_info()
            action_path = (node_pair, node_pair_path_idx)

        
        # Embed VNF, node, and path
        if not self.m2.embed_vnf_node_path(action_vnf, action_node, action_path):
            return self._get_obs(), reward, terminated, truncated, self._get_info()
        
        # Calculate reward using modification cost from M1 to M2
        cost = self.m2.compute_modification_cost(self.m1, self.m1_resource_cost, self.network)
        self.reward = -cost

        # If completed embeddings to target node, check M2 is correcly connected
        if self.m2.target_embeds_counter >= len(self.m2.source_nodes):
            if not self.m2.end_vnf_embedding():
                return self._get_obs(), reward, terminated, truncated, self._get_info()
            
            # If correcly completed M2, end episode and return reward and M2 as candidate
            reward = self.reward + (np.count_nonzero(self._mapped_vnfs==0) * self.reward_coefficient)  # M2 candidate reward
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
        # Initialize action mask to zeros
        action_mask = np.full(self._len_action, 0, dtype=np.int8)

        # If TODO
        if self.m2.ongoing_vnf_id == self.m2.VNFR_TARGET_KEY:
            self.__update_action_mask_for_target_node(action_mask, self.m2.source_nodes, self.m2.source_rates)
            return action_mask
        
        if len(self.m2.vnffg) >= self._len_auxiliary:
            self.__update_action_mask_for_target_node(action_mask, self.m2.next_src_nodes, self.m2.next_src_rates)
        
        for vnf_id in self.m2.usable_vnfs:
            action_vnf_idx = self.get_idx(vnf_id)
            
            # Skip if no more instances in auxiliary
            if self._auxiliary[action_vnf_idx - 1] <= 0:
                continue

            # Source nodes if same ongoing or None; next source nodes if different ongoing
            source_nodes = self.m2.source_nodes
            is_ongoing_vnf = True
            if self.m2.ongoing_vnf_id is not None and self.m2.ongoing_vnf_id != vnf_id:
                source_nodes = self.m2.next_src_nodes
                is_ongoing_vnf = False
            
            for node_id in self.m2.vnf_capable_node_ids:
                action_node_idx = self.get_idx(node_id) - 1

                for src_node_idx in range(len(source_nodes)):
                    source_node = source_nodes[src_node_idx]
                    if is_ongoing_vnf and (self.m2.source_rates[source_node].is_allocated) and (node_id in self.m2.next_src_nodes):
                        continue
                    self.__update_action_mask_for_node_pair(action_mask, action_vnf_idx, source_node, src_node_idx, node_id, action_node_idx)
        
        return action_mask
    
    def __update_action_mask_for_node_pair(self, action_mask: np.ndarray, action_vnf_idx: int, source_node: str, src_node_idx: int, action_node: str, action_node_idx: int):
        """
        TODO
        """
        if action_node == source_node:
            action_path_idx = 0
            action_idx = self._action_space_to_action_idx(action_vnf_idx, action_node_idx, action_path_idx)
            action_mask[action_idx] = 1
        else:
            node_pair = (source_node, action_node)
            len_node_pair_paths = len(self.node_pair_paths[node_pair])
            for node_pair_path_idx in range(1, len_node_pair_paths + 1):
                action_path_idx = (src_node_idx * self.max_paths_per_node_pair) + node_pair_path_idx
                action_idx = self._action_space_to_action_idx(action_vnf_idx, action_node_idx, action_path_idx)
                action_mask[action_idx] = 1
    
    def __update_action_mask_for_target_node(self, action_mask: np.ndarray, source_nodes: list, source_rates: dict[str, EmbeddingRate]):
        """
        TODO
        """
        action_vnf_idx = 0
        target_node = self.vnfr.target_node
        target_node_idx = self.get_idx(target_node) - 1

        for src_node_idx in range(len(source_nodes)):
            source_node = source_nodes[src_node_idx]
            if source_rates[source_node].is_allocated:
                continue
            self.__update_action_mask_for_node_pair(action_mask, action_vnf_idx, source_node, src_node_idx, target_node, target_node_idx)


class DRLFuncScalingRun(DRLRun):
    """
    TODO
    """
    MAX_VNF_INSTANCES = 3

    # def __init__(self, m1: Embedding, network: Network, vnfr: VNFR, max_vnf_instances = MAX_VNF_INSTANCES, seed: int = None):
    def __init__(self, m1: Embedding, network: Network, vnfr: VNFR, seed: int = None, experiment: int = 0):
        """
        TODO
        """
        # Initialize environment
        env = DRLFuncScalingEnv(m1=m1, network=network, vnfr=vnfr, seed=seed)

        # Build name of file to write results
        filename = "{0}_{1}vnfs_seed{2}_exp{3}.csv".format(self.__class__.__name__, str(len(vnfr.vnfs)), str(seed), str(experiment))

        # Call superclass constructor
        # super().__init__(env, filename, max_vnf_instances, seed)
        super().__init__(env, filename, seed)
