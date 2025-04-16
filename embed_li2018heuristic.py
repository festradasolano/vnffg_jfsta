from embedding import Embedding, NodeEmbed
from substrate import Network, Node
from vnfr import VNFR

import math

class Li2018Embedding(Embedding):
    """
    TODO
    """
    MAX_HOPS = 3

    def __init__(self, vnfs: list, vnf_capable_nodes: list, links: list, seed: int = None, max_hops: int = MAX_HOPS, enable_split: bool = False):
        """
        TODO
        """
        super().__init__(vnfs, vnf_capable_nodes, links, seed)
        self.max_hops = max_hops
        self.enable_split = enable_split

    def build_embbeding(self, network: Network, vnfr: VNFR) -> bool:
        """
        TODO
        """
        # Build VNF Forwarding Graph (VNF-FG)
        self.build_vnffg(vnfr)

        if not self.enable_split:
            return self.__build_embedding(network, vnfr)
        return self.__build_embedding_split(network, vnfr)
    
    def build_vnffg(self, vnfr: VNFR):
        """
        TODO
        """
        self.vnffg = []
        
        # Get VNFs without dependency, in ascending order by ratio
        vnfs_no_dependency = vnfr.dependants[self.NO_DEPENDENCY]
        usable_vnfs = self.__insert_vnfs_sort_asc_by_ratio(vnfs_no_dependency, [], vnfr.vnfs)

        while len(usable_vnfs) > 0:
            # Add usable VNF with lowest ratio
            vnf_id = usable_vnfs.pop(0)
            self.vnffg.append(vnf_id)

            # Add VNF dependants as usable, if any, in ascending order by ratio
            if vnf_id in vnfr.dependants:
                usable_vnfs = self.__insert_vnfs_sort_asc_by_ratio(vnfr.dependants[vnf_id], usable_vnfs, vnfr.vnfs)
    
    def __build_embedding(self, network: Network, vnfr: VNFR) -> bool:
        """
        TODO
        """
        source_node = vnfr.source_node
        source_rate = SourceRate(vnfr.data_rate)
        for vnf_id in self.vnffg:
            # Compute VNF CPU demand, using ideal data rate
            vnf = vnfr.vnfs[vnf_id]
            vnf_cpu_demand = math.ceil(vnf.cpu_demand_per_bw * source_rate.ideal / 1_000)

            # Find an embedding node, increasing hops
            node_embed = self.__find_embedding_node(network, source_node, vnf_cpu_demand)
            
            # Unable to find a node to embed a VNF
            if node_embed is None:
                return False
            
            # Embed VNF in node
            self.embed_vnf_node(vnf_id, node_embed.id, vnf_cpu_demand)
            
            # Embed shortest path between VNF source node and embedding node, using actual data rate
            source_rate.actual = self.embed_vnf_shortest_path(source_node, node_embed.id, network, vnf_id, source_rate.actual)

            # Update source node and source rate
            source_node = node_embed.id
            source_rate.update(vnf.ratio_out2in)

        # Embed shortest path between last VNF embedding node and VNFR target node, using actual data rate
        source_rate.actual = self.embed_vnf_shortest_path(source_node, vnfr.target_node, network, self.VNFR_TARGET_KEY, source_rate.actual)
        self.throughput = source_rate.actual
        self.delay += vnfr.vnfs_delay
        return True
    
    def __build_embedding_split(self, network: Network, vnfr: VNFR) -> bool:
        source_node = vnfr.source_node
        source_node_split = None
        source_rate = SourceRate(vnfr.data_rate)
        for vnf_id in self.vnffg:
            # Compute VNF CPU demand, using ideal data rate
            vnf = vnfr.vnfs[vnf_id]
            vnf_cpu_demand = math.ceil(vnf.cpu_demand_per_bw * source_rate.ideal_total / 1_000)

            # Find an embedding node, increasing hops
            node_embed = self.__find_embedding_node(network, source_node, vnf_cpu_demand)

            # Embed VNF in node if found; otherwise, split VNF
            if node_embed is not None:
                self.embed_vnf_node(vnf_id, node_embed.id, vnf_cpu_demand)

                # Embed shortest path between VNF source node and embedding node, using actual rate
                source_rate.actual = self.embed_vnf_shortest_path(source_node, node_embed.id, network, vnf_id, source_rate.actual)
                
                # If split source, embed shortest path between VNF split source node and embedding node, using actual split rate
                if source_node_split is not None:
                    source_rate.actual_split = self.embed_vnf_shortest_path(source_node_split, node_embed.id, network, vnf_id, source_rate.actual_split)
                
                # Update source node and source rate
                source_node = node_embed.id
                source_node_split = None
                source_rate.update(vnf.ratio_out2in)
                source_rate.join()
            
            else:
                # Unable to split CPU demand shorther than 1
                if vnf_cpu_demand <= 1:
                    return False
                
                # Find two nodes to embed splitted VNF
                if source_node_split is None:
                    # Find two nodes for no split source
                    node_embed, node_embed_split = self.__find_split_embedding_nodes(network, source_node, vnf_cpu_demand)
                    if (node_embed is None) or (node_embed_split is None):
                        return False
                    
                    # Split CPU and source rate
                    vnf_cpu = node_embed.cpu_available
                    vnf_cpu_split = vnf_cpu_demand - vnf_cpu
                    split_portion = vnf_cpu_split / vnf_cpu_demand
                    source_rate.split(split_portion)
                else:
                    # Compute CPU demand based on split source rate
                    vnf_cpu = math.ceil(vnf_cpu_demand * (1 - source_rate.split_portion))
                    vnf_cpu_split = math.ceil(vnf_cpu_demand * source_rate.split_portion)

                    # Find two nodes for split source
                    node_embed, node_embed_split = self.__find_split_embedding_nodes_split_cpu(network, source_node, vnf_cpu, vnf_cpu_split)
                    if (node_embed is None) or (node_embed_split is None):
                        return False
                
                # Embed VNF in split nodes
                self.embed_vnf_node(vnf_id, node_embed.id, vnf_cpu)
                self.embed_vnf_node(vnf_id, node_embed_split.id, vnf_cpu_split)

                # Embed paths
                source_rate.actual = self.embed_vnf_shortest_path(source_node, node_embed.id, network, vnf_id, source_rate.actual)
                if source_node_split is None:
                    source_rate.actual_split = self.embed_vnf_shortest_path(source_node, node_embed_split.id, network, vnf_id, source_rate.actual_split)
                else:
                    source_rate.actual_split = self.embed_vnf_shortest_path(source_node_split, node_embed_split.id, network, vnf_id, source_rate.actual_split)
                
                # Update source node and source rate
                source_node = node_embed.id
                source_node_split = node_embed_split.id
                source_rate.update(vnf.ratio_out2in)

        # Embed shortest path between last VNF embedding node(s) and VNFR target node, using actual data rate
        source_rate.actual = self.embed_vnf_shortest_path(source_node, vnfr.target_node, network, self.VNFR_TARGET_KEY, source_rate.actual)
        if source_node_split is not None:
            source_rate.actual_split = self.embed_vnf_shortest_path(source_node_split, vnfr.target_node, network, self.VNFR_TARGET_KEY, source_rate.actual_split)
        source_rate.update()
        source_rate.join()
        self.throughput = source_rate.actual_total
        self.delay += vnfr.vnfs_delay
        return True
    
    # def _embed_node_vnf(self, vnf_id: str, node_id: str, vnf_cpu_demand: int):
    #     """
    #     TODO
    #     """
    #     self.embedding_vnfs[vnf_id][node_id] = []
    #     self.allocated_node_cpus[node_id][vnf_id] = vnf_cpu_demand
    #     self.allocated_node_cpus[node_id][self.TOTAL_KEY] += vnf_cpu_demand
    
    def __find_embedding_node(self, network: Network, source_node: str, vnf_cpu_demand: int) -> NodeEmbed:
        """
        TODO Finds an embedding node, increasing maximum hops
        """
        for hops in range(self.max_hops + 1):
            vnf_capable_nodes = network.get_vnf_capable_nodes_from_source(source_node, hops, vnf_cpu_demand)
            node_embed = self.__get_node_max_cpu_available(vnf_capable_nodes)
            if node_embed is not None and node_embed.cpu_available >= vnf_cpu_demand:
                return node_embed
        return None
    
    def __find_split_embedding_nodes(self, network: Network, source_node: str, vnf_cpu_demand: int) -> "tuple[NodeEmbed, NodeEmbed]":
        """
        TODO
        """
        for hops in range(self.max_hops):
            vnf_capable_nodes = network.get_vnf_capable_nodes_from_source(source_node, hops)
            vnf_capable_nodes.extend(network.get_vnf_capable_nodes_from_source(source_node, hops+1))
            node_embed, node_embed_split = self.__get_node_pair_max_cpu_available(vnf_capable_nodes)
            if node_embed is not None and node_embed_split is not None:
                total_cpu = node_embed.cpu_available + node_embed_split.cpu_available
                if total_cpu >= vnf_cpu_demand:
                    return (node_embed, node_embed_split)
        return (None, None)
    
    def __find_split_embedding_nodes_split_cpu(self, network: Network, source_node: str, vnf_cpu: int, vnf_cpu_split: int) -> "tuple[NodeEmbed, NodeEmbed]":
        """
        TODO
        """
        for hops in range(self.max_hops):
            vnf_capable_nodes = network.get_vnf_capable_nodes_from_source(source_node, hops)
            vnf_capable_nodes.extend(network.get_vnf_capable_nodes_from_source(source_node, hops+1))
            node_embed, node_embed_split = self.__get_node_pair_max_cpu_available(vnf_capable_nodes)
            if ((node_embed is not None and node_embed_split is not None) and
                (node_embed.cpu_available >= vnf_cpu and node_embed_split.cpu_available >= vnf_cpu_split)):
                return (node_embed, node_embed_split)
        return (None, None)
    
    def __get_node_max_cpu_available(self, nodes: "list[Node]") -> NodeEmbed:
        """
        TODO
        """
        # If empty list, return None
        if len(nodes) == 0:
            return None
        
        # If only one node, return its ID and available CPU
        max_node = nodes[0]
        max_node_cpu = max_node.cpu_capacity - self.allocated_node_cpus[max_node.id][self.TOTAL_KEY]
        if len(nodes) == 1:
            return NodeEmbed(max_node.id, max_node_cpu)
        
        # Find node with highest available CPU and return it
        for node in nodes[1:]:
            node_cpu = node.cpu_capacity - self.allocated_node_cpus[node.id][self.TOTAL_KEY]
            if node_cpu > max_node_cpu:
                max_node = node
                max_node_cpu = node_cpu
        return NodeEmbed(max_node.id, max_node_cpu)
    
    def __get_node_pair_max_cpu_available(self, nodes: "list[Node]") -> "tuple[NodeEmbed, NodeEmbed]":
        """
        TODO
        """
        # If less than two nodes, return None
        if len(nodes) < 2:
            return (None, None)
        
        # Sort list of nodes in descending order by available CPU
        sorted_nodes: list[NodeEmbed] = list()
        for node in nodes:
            node_cpu = node.cpu_capacity - self.allocated_node_cpus[node.id][self.TOTAL_KEY]
            idx = len(sorted_nodes)
            is_same_node = False
            for i in len(sorted_nodes):
                compare_node = sorted_nodes[i]
                if node.id == compare_node.id:
                    is_same_node = True
                    break
                elif node_cpu > compare_node.cpu_available:
                    idx = i
                    break
            # Add node to sorted list, if not added before
            if not is_same_node:
                sorted_nodes.insert(idx, NodeEmbed(node.id, node_cpu))
        
        # If less than two nodes in sorted list, return None
        if len(sorted_nodes) < 2:
            return (None, None)
        
        # Return the two nodes with the highest available CPU
        return (sorted_nodes[0], sorted_nodes[1])

    def __insert_vnfs_sort_asc_by_ratio(self, unsorted_vnfs: list, sorted_vnfs: list, vnfs: dict) -> list:
        """
        TODO
        """
        for vnf_id in unsorted_vnfs:
            vnf_ratio = vnfs[vnf_id].ratio_out2in
            idx = len(sorted_vnfs)
            for i in range(len(sorted_vnfs)):
                compare_vnf_id = sorted_vnfs[i]
                compare_ratio = vnfs[compare_vnf_id].ratio_out2in
                if vnf_ratio <= compare_ratio:
                    idx = i
                    break
            sorted_vnfs.insert(idx, vnf_id)
        return sorted_vnfs
   

class SourceRate():
    """
    TODO
    """

    def __init__(self, source_rate: float):
        """
        TODO
        """
        self.ideal = source_rate
        self.actual = source_rate
        self.is_split = False
        self.split_portion = 0
        self.ideal_split = 0
        self.actual_split = 0
        self.ideal_total = source_rate
        self.actual_total = source_rate
    
    def join(self):
        """TODO
        """
        self.ideal = self.ideal_total
        self.actual = self.actual_total
        self.ideal_split = 0
        self.actual_split = 0
        self.split_portion = 0
        self.split = False
    
    def split(self, split_portion: float):
        """TODO
        """
        self.split_portion = split_portion
        self.ideal_split = math.floor(self.ideal_total * self.split_portion)
        self.actual_split = math.floor(self.actual_total * self.split_portion)
        self.ideal = self.ideal_total - self.ideal_split
        self.actual = self.actual_total - self.actual_split
        self.split = True
    
    def update(self, vnf_ratio_out2in: float = 1):
        """TODO
        """
        self.ideal *= vnf_ratio_out2in
        self.actual *= vnf_ratio_out2in
        self.ideal_split *= vnf_ratio_out2in
        self.actual_split *= vnf_ratio_out2in
        self.ideal_total = self.ideal + self.ideal_split
        self.actual_total = self.actual + self.actual_split
