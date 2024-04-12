from embedding import Embedding
from substrate import Network
from vnfr import VNFR

import math

class RandomEmbedding(Embedding):
    """
    TODO
    """

    def build_embbeding(self, network: Network, vnfr: VNFR) -> bool:
        """
        TODO
        """
        # Build a random VNF Forwarding Graph (VNF-FG)
        self.build_vnffg(vnfr)

        # Randomly embed each VNF
        source_node = vnfr.source_node
        ideal_source_rate = vnfr.data_rate
        actual_source_rate = vnfr.data_rate
        vnf_capable_nodes = network.vnf_capable_nodes.copy()
        for vnf_id in self.vnffg:
            # Compute VNF CPU demand, using ideal data rate
            vnf = vnfr.vnfs[vnf_id]
            vnf_cpu_demand = math.ceil(vnf.cpu_demand_per_bw * ideal_source_rate / 1_000)

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
                    self.embedding_vnfs[vnf_id][node.id] = []
                    self.allocated_node_cpus[node.id][vnf_id] = vnf_cpu_demand
                    self.allocated_node_cpus[node.id][self.TOTAL_KEY] += vnf_cpu_demand
                    embedded = True
                else:
                    # Count number of tries and break to avoid infinite loop
                    tries += 1
                    if tries > 2*len(vnf_capable_nodes):
                        return False
            
            # Embed shortest path between VNF source node and embedding node, using actual data rate
            actual_source_rate = self._embed_shortest_path(source_node, node.id, network, vnf_id, actual_source_rate)

            # Update values: source node, source rate, and transmission delay
            source_node = node.id
            ideal_source_rate *= vnf.ratio_out2in
            actual_source_rate *= vnf.ratio_out2in

        # Embed shortest path between last VNF embedding node and VNFR target node, using actual data rate
        actual_source_rate = self._embed_shortest_path(source_node, vnfr.target_node, network, self.VNFR_TARGET_KEY, actual_source_rate)
        self.throughput = actual_source_rate
        self.delay += vnfr.vnfs_delay
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
