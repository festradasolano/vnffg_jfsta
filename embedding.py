from substrate import Network
from vnfr import VNFR

from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

import math

# N. T. Jahromi, S. Kianpisheh, and R. H. Glitho, “Online VNF placement and chaining
# for value-added services in content delivery networks,” in Proc. IEEE International
# Symposium on Local and Metropolitan Area Networks (LANMAN), 2018, pp. 19–24.

# CONSTANTS
VNF_LICENSE_COST = 100  # (Jahromi et al., 2018)
COST_PER_CPU = 5  # (Jahromi et al., 2018)
COST_ACTIVE_NODE = 1  #
VM_RAM_PER_CPU = 2  # General purpose VMs from Azure and AWS, in GB
VM_DISK_PER_CPU = 10  # General purpose VMs from Azure and AWS, in GB
# VM_RAM = 4  # (Jahromi et al., 2018)
# VM_DISK = 40  # (Jahromi et al., 2018)


class Embedding:
    """
    TODO
    """
    NO_DEPENDENCY = ""
    VNFR_TARGET_KEY = "target"
    TOTAL_KEY = "total"

    def __init__(self, vnfs: list, vnf_capable_nodes: list, links: list, seed: int = None):
        """
        TODO
        """
        self.rs = RandomState(MT19937(SeedSequence(seed)))
        self.vnffg = list()
        self.throughput = 0
        self.delay = 0

        self.embedding_vnfs = dict()  # {f1: {n1: [l1], n2: [l2, l3]}, f2: {n1: []}}
        for vnf_id in vnfs:
            self.embedding_vnfs[vnf_id] = dict()
        self.embedding_vnfs[self.VNFR_TARGET_KEY] = dict()
        
        self.allocated_node_cpus = dict()  # {n1: {f1: c1, f2: c2}, n2: {f3: c3}}
        for node in vnf_capable_nodes:
            # self.allocated_node_cpus[node.id] = dict()
            self.allocated_node_cpus[node.id] = {
                self.TOTAL_KEY: 0
            }
        
        self.allocated_link_bws = dict()  # {l1: {f1: r1, f2: r2}], l2: {f1: r1}}
        for link_id in links:
            # self.allocated_link_bws[link_id] = dict()
            self.allocated_link_bws[link_id] = {
                self.TOTAL_KEY: 0
            }
    
    def build_embbeding(self, network: Network, vnfr: VNFR) -> bool:
        """
        TODO
        """
        raise NotImplementedError("Subclass must implement this method")
    
    def build_vnffg(self, vnfr: VNFR):
        """
        TODO
        """
        raise NotImplementedError("Subclass must implement this method")

    def compute_resource_cost(self, network: Network):
        """
        TODO
        """
        # Compute cost of CPU and active nodes
        total_cpus = 0
        active_nodes = 0
        for node_id in self.allocated_node_cpus:
            node_cpus = self.allocated_node_cpus[node_id][self.TOTAL_KEY]
            if node_cpus > 0:
                total_cpus += node_cpus
                active_nodes += 1
        cpu_cost = total_cpus * COST_PER_CPU
        active_cost = active_nodes * COST_ACTIVE_NODE
        
        # Compute cost of bandwidth
        bw_cost = 0
        for link_id in self.allocated_link_bws:
            link_bw_in_gb = self.allocated_link_bws[link_id][self.TOTAL_KEY] / 1_000
            if link_bw_in_gb > 0:
                link_bw_cost = network.links[link_id].cost_per_gb * link_bw_in_gb
                bw_cost += link_bw_cost
        
        # Return sum of costs
        return cpu_cost + active_cost + bw_cost
    
    def _embed_shortest_path(self, source_node: str, target_node: str, network: Network, vnf_id: str, rate_to_alloc: float) -> float:
        """
        TODO
        """
        # Embed VNF in shortest path between source and target nodes
        path = network.get_shortest_path_between_nodes(source_node, target_node, rs=self.rs)
        self.embedding_vnfs[vnf_id][target_node] = path

        # Allocate rate to links in shortest path
        for l in path:
            # Get link and compute available bandwidth using embedded VNFs
            link_id = frozenset(l)
            link = network.links[link_id]
            link_bw_available = link.bw_capacity - self.allocated_link_bws[link_id][self.TOTAL_KEY]
            
            # Embed VNF, limiting incoming rate to available link bandwidth
            if link_bw_available < rate_to_alloc:
                rate_to_alloc = link_bw_available
            self.allocated_link_bws[link_id][vnf_id] = rate_to_alloc
            self.allocated_link_bws[link_id][self.TOTAL_KEY] += rate_to_alloc
            
            # Compute tx delay
            self.delay += link.tx_delay
        return rate_to_alloc


class NodeEmbed:
    """
    TODO
    """

    def __init__(self, id: str, cpu_available: int):
        """
        TODO
        """
        self.id = id
        self.cpu_available = cpu_available


def compute_modification_cost(original_embed: Embedding, modified_embed: Embedding, network: Network) -> float:
    """TODO
    """
    # Compute cost difference of resource consumption
    orig_res_cost = original_embed.compute_resource_cost(network)
    mod_res_cost = modified_embed.compute_resource_cost(network)
    res_cost_diff = mod_res_cost - orig_res_cost

    # Compute costs of VNF migration and count number of VNF instances
    orig_count_vnfs = 0
    mod_count_vnfs = 0
    migration_cost = 0
    for vnf in modified_embed.embedding_vnfs:
        if vnf == Embedding.VNFR_TARGET_KEY:
            continue

        orig_node_id = next(iter(original_embed.embedding_vnfs[vnf]))
        for mod_node_id in modified_embed.embedding_vnfs[vnf]:
            # If VNF was migrated to another node(s), compute cost
            if mod_node_id != orig_node_id:
                # Calculate VM size for VNF
                vnf_cpu = original_embed.allocated_node_cpus[orig_node_id][vnf]
                vnf_vm_size = (vnf_cpu * VM_DISK_PER_CPU) + (vnf_cpu * VM_RAM_PER_CPU)
                
                # Compute migration cost between nodes
                node_mig_cost = 0
                migration_path = network.get_shortest_path_between_nodes(orig_node_id, mod_node_id)
                for l in migration_path:
                    link_id = frozenset(l)
                    node_mig_cost = node_mig_cost + (vnf_vm_size * network.links[link_id].cost_per_gb)
                migration_cost += node_mig_cost

        # Compute number of instantiated VNFs
        orig_count_vnfs += len(original_embed.embedding_vnfs[vnf])
        mod_count_vnfs += len(modified_embed.embedding_vnfs[vnf])
    
    # Compute cost of new VNF instances and total cost
    extra_vnfs_cost = (mod_count_vnfs - orig_count_vnfs) * VNF_LICENSE_COST
    return res_cost_diff + extra_vnfs_cost + migration_cost