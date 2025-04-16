from embedding import Embedding
from substrate import Network
from vnfr import VNFR

import math

class RandomEmbedding(Embedding):
    """
    TODO
    """
    MAX_HOPS = 3

    def __init__(self, vnfs: list, vnf_capable_nodes: list, links: list, max_hops: int = MAX_HOPS, seed: int = None):
        """
        TODO
        """
        super().__init__(vnfs, vnf_capable_nodes, links, seed)
        self.max_hops = max_hops
        self.node_pair_paths = None

    def build_embbeding(self, network: Network, vnfr: VNFR) -> bool:
        """
        TODO
        """
        # Build a random VNF Forwarding Graph (VNF-FG)
        self.build_vnffg(vnfr)

        # Get paths between node pairs
        self.node_pair_paths, _ = network.get_node_pair_paths(self.max_hops)

        # Randomly embed each VNF
        source_node = vnfr.source_node
        ideal_rate_to_alloc = vnfr.data_rate
        actual_rate_to_alloc = vnfr.data_rate
        vnf_capable_nodes = network.vnf_capable_nodes.copy()
        for vnf_id in self.vnffg:
            # Compute VNF CPU demand, using ideal data rate
            vnf = vnfr.vnfs[vnf_id]
            vnf_cpu_demand = math.ceil(vnf.cpu_demand_per_bw * ideal_rate_to_alloc / 1_000)

            # Embed VNF in a random capable node
            embedded = False
            node = None
            tries = 0
            while not embedded:
                # Select a random node and compute available CPU using embedded VNFs
                i_node = self.rs.randint(len(vnf_capable_nodes))
                node = vnf_capable_nodes[i_node]
                node_cpu_available = node.cpu_capacity - self.allocated_node_cpus[node.id][self.TOTAL_KEY]

                # Remove node with no available CPU
                if node_cpu_available == 0:
                    vnf_capable_nodes.pop(i_node)
                    continue

                # Validate if node supports VNF demand
                if node_cpu_available >= vnf_cpu_demand:
                    # Embed VNF in node
                    self.embed_vnf_node(vnf_id, node.id, vnf_cpu_demand)
                    embedded = True
                else:
                    # Count number of tries and break to avoid infinite loop
                    tries += 1
                    if tries > 2*len(vnf_capable_nodes):
                        return False
            
            # Embed random path between VNF source node and embedding node, using actual data rate
            path = self.__get_random_path(source_node, node.id)
            if path is None:
                return False
            allocated_rate, tx_delay = self.embed_vnf_path(node.id, path, network, vnf_id, actual_rate_to_alloc)
            self.delay += tx_delay

            # Update values: source node, source rate, and transmission delay
            source_node = node.id
            ideal_rate_to_alloc *= vnf.ratio_out2in
            actual_rate_to_alloc = allocated_rate * vnf.ratio_out2in

        # Embed random path between last VNF embedding node and VNFR target node, using actual data rate
        path = self.__get_random_path(source_node, vnfr.target_node)
        if path is None:
            return False
        self.throughput, tx_delay = self.embed_vnf_path(vnfr.target_node, path, network, self.VNFR_TARGET_KEY, actual_rate_to_alloc)
        self.delay += tx_delay + vnfr.vnfs_delay
        return True
    
    def build_vnffg(self, vnfr: VNFR):
        """
        Builds a random VNF Forwarding Graph (VNF-FG)

        TODO
        """
        self.vnffg = []
        usable_vnfs = vnfr.dependants[self.NO_DEPENDENCY].copy()
        while len(usable_vnfs) > 0:
            # Randomly add a VNF from the list
            i_vnf = self.rs.randint(len(usable_vnfs))
            vnf_id = usable_vnfs.pop(i_vnf)
            self.vnffg.append(vnf_id)

            # Add VNF dependants to list of usable VNFs, if any
            if vnf_id in vnfr.dependants:
                usable_vnfs.extend(vnfr.dependants[vnf_id])
    
    def get_overloaded_vnfs(self) -> list[str]:
        """
        TODO
        """
        ovld_vnfs = list()
        # If no lowest throughput node, return empty list
        if len(self.min_throughput_node) == 0:
            return ovld_vnfs
        
        # Build list with VNFs in lowest throughput node
        for vnf in self.allocated_node_cpus[self.min_throughput_node]:
            if vnf != self.TOTAL_KEY:  # skip total key
                ovld_vnfs.append(vnf)
        return ovld_vnfs
    
    def __get_random_path(self, source_node: str, target_node: str) -> list:
        """
        TODO
        """
        # Return empty path if same source and target
        if source_node == target_node:
            return []
        
        # Select a random path between source and target, if exists
        node_pair = (source_node, target_node)
        len_node_pair_paths = len(self.node_pair_paths[node_pair])
        if len_node_pair_paths <= 0:
            return None
        path_idx = self.rs.randint(len_node_pair_paths)
        path = self.node_pair_paths[node_pair][path_idx]
        return path
