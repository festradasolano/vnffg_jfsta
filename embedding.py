from __future__ import annotations

from substrate import Network, NUM_NODES, LINK_MAX_BW_CAPACITY, LINK_MAX_COST_PER_GB, LINK_TUPLES, MAX_HOPS
from vnfr import VNFR, MAX_NUM_VNFS, VNF_MAX_CPU_DEMAND_PER_BW, MAX_DATA_RATE, VNF_MAX_RATIO_OUT2IN

from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

import math

# N. T. Jahromi, S. Kianpisheh, and R. H. Glitho, “Online VNF placement and chaining
# for value-added services in content delivery networks,” in Proc. IEEE International
# Symposium on Local and Metropolitan Area Networks (LANMAN), 2018, pp. 19–24.

# CONSTANTS
VNF_LICENSE_COST = 100  # (Jahromi et al., 2018)
# VNF_LICENSE_COST = 10  # (Jahromi et al., 2018)
COST_PER_CPU = 5  # (Jahromi et al., 2018)
VM_RAM_PER_CPU = 2  # General purpose VMs from Azure and AWS, in GB
VM_DISK_PER_CPU = 10  # General purpose VMs from Azure and AWS, in GB
# VM_RAM = 4  # (Jahromi et al., 2018)
# VM_DISK = 40  # (Jahromi et al., 2018)
COST_ACTIVE_NODE = 1  #
COST_ACTIVE_LINK = 1  #

MAX_NUM_INSTANCES_PER_VNF = 3

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
    
    def compute_modification_cost(self, m1: Embedding, m1_resource_cost: float, network: Network) -> float:
        """
        TODO
        """
        # Compute cost difference of resource consumption
        # m1_resource_cost = m1.compute_resource_cost(network)
        # print("***m1_resource_cost", m1_resource_cost)
        m2_resource_cost = self.compute_resource_cost(network)
        # print("***m2_resource_cost", m2_resource_cost)
        resource_cost_diff = m2_resource_cost - m1_resource_cost

        # Compute costs of VNF migration and count number of extra VNF instances
        migration_cost = 0
        extra_vnfs = 0
        for vnf in self.embedding_vnfs:
            if vnf == Embedding.VNFR_TARGET_KEY:
                continue

            m1_node_id = next(iter(m1.embedding_vnfs[vnf]))
            for m2_node_id in self.embedding_vnfs[vnf]:
                # If VNF was migrated to another node(s), compute cost
                if m2_node_id != m1_node_id:
                    # Calculate VM size for VNF
                    vnf_cpu = m1.allocated_node_cpus[m1_node_id][vnf]
                    vnf_vm_size = (vnf_cpu * VM_DISK_PER_CPU) + (vnf_cpu * VM_RAM_PER_CPU)
                    
                    # Compute migration cost between nodes
                    node_mig_cost = 0
                    migration_path = network.get_shortest_path_between_nodes(m1_node_id, m2_node_id, rs=self.rs)
                    for l in migration_path:
                        link_id = frozenset(l)
                        node_mig_cost = node_mig_cost + (vnf_vm_size * network.links[link_id].cost_per_gb)
                    migration_cost += node_mig_cost

            # Compute number of instantiated VNFs
            if len(self.embedding_vnfs[vnf]) > len(m1.embedding_vnfs[vnf]):
                extra_vnfs += len(self.embedding_vnfs[vnf]) - len(m1.embedding_vnfs[vnf])
        
        # Compute cost of new VNF instances
        extra_vnfs_cost = extra_vnfs * VNF_LICENSE_COST

        # Compute and return total cost
        # print("***resource_cost_diff", resource_cost_diff)
        # print("***extra_vnfs_cost", extra_vnfs_cost)
        # print("***migration_cost", migration_cost)
        return resource_cost_diff + extra_vnfs_cost + migration_cost
    
    def compute_resource_cost(self, network: Network) -> float:
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
        
        # Compute cost of bandwidth and active links
        bw_cost = 0
        active_links = 0
        for link_id in self.allocated_link_bws:
            link_bw_in_gb = self.allocated_link_bws[link_id][self.TOTAL_KEY] / 1_000
            if link_bw_in_gb > 0:
                link_bw_cost = network.links[link_id].cost_per_gb * link_bw_in_gb
                bw_cost += link_bw_cost
            if len(self.allocated_link_bws[link_id]) > 1:
                active_links += 1
        
        # Compute cost of active resources and return sum of costs
        active_cost = (active_nodes * COST_ACTIVE_NODE) + (active_links * COST_ACTIVE_LINK)
        return cpu_cost + active_cost + bw_cost
    
    def embed_vnf_node(self, vnf_id: str, node_id: str, vnf_cpu_demand: int):
        """
        TODO
        """
        if vnf_id in self.allocated_node_cpus[node_id]:
            self.allocated_node_cpus[node_id][vnf_id] += vnf_cpu_demand
        else:
            self.allocated_node_cpus[node_id][vnf_id] = vnf_cpu_demand
        self.allocated_node_cpus[node_id][self.TOTAL_KEY] += vnf_cpu_demand
    
    def embed_vnf_path(self, node_id: str, path: list, network: Network, vnf_id: str, rate_to_alloc: float) -> tuple[float, int]:
        """
        TODO
        """
        # Embed VNF in entering path of target node
        if node_id not in self.embedding_vnfs[vnf_id]:
            self.embedding_vnfs[vnf_id][node_id] = list()
        self.embedding_vnfs[vnf_id][node_id].append(path)
        # self.embedding_vnfs[vnf_id][node_id] = path

        # Allocate rate to links in path
        tx_delay = 0
        for l in path:
            # Get link and compute available bandwidth using embedded VNFs
            link_id = frozenset(l)
            link = network.links[link_id]
            link_bw_available = link.bw_capacity - self.allocated_link_bws[link_id][self.TOTAL_KEY]

            # Embed VNF, limiting incoming rate to available link bandwidth
            if link_bw_available < rate_to_alloc:
                rate_to_alloc = link_bw_available
            
            if vnf_id in self.allocated_link_bws[link_id]:
                self.allocated_link_bws[link_id][vnf_id] += rate_to_alloc
            else:
                self.allocated_link_bws[link_id][vnf_id] = rate_to_alloc
            self.allocated_link_bws[link_id][self.TOTAL_KEY] += rate_to_alloc

            # Compute transmission delay
            tx_delay += link.tx_delay
        return rate_to_alloc, tx_delay
    
    def embed_vnf_shortest_path(self, source_node: str, target_node: str, network: Network, vnf_id: str, rate_to_alloc: float) -> tuple[float, int]:
        """
        TODO
        """
        # Embed VNF in shortest path between source and target nodes
        path = network.get_shortest_path_between_nodes(source_node, target_node, rs=self.rs)
        return self.embed_vnf_path(target_node, path, network, vnf_id, rate_to_alloc)


class EmbeddingRate():
    """
    TODO
    """

    def __init__(self, rate_to_alloc: float, allocated_rate: float = 0, is_allocated: bool = False):
        """
        TODO
        """
        self.rate_to_alloc = rate_to_alloc
        self.allocated_rate = allocated_rate
        self.is_allocated = is_allocated
    
    def add_rate_to_alloc(self, rate_to_alloc: float):
        """
        TODO
        """
        self.rate_to_alloc += rate_to_alloc
    
    def add_allocated_rate(self, allocated_rate: float):
        """
        TODO
        """
        self.allocated_rate += allocated_rate
        self.is_allocated = True
    
    def apply_ratio_out2in(self, ratio_out2in: float):
        """
        TODO
        """
        self.rate_to_alloc *= ratio_out2in
        self.allocated_rate *= ratio_out2in
    
    def get_missing_rate_to_alloc(self) -> float:
        """
        TODO
        """
        return max(0, self.rate_to_alloc - self.allocated_rate)
    
    def is_rate_allocated(self, min_proportion: float = 1) -> bool:
        """
        TODO
        """
        min_rate_to_alloc = self.rate_to_alloc * min_proportion
        if self.allocated_rate < min_rate_to_alloc:
            return False
        return self.is_allocated


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


def compute_max_modification_cost() -> float:
    # Cost of CPUs
    max_rate = MAX_DATA_RATE * (VNF_MAX_RATIO_OUT2IN ** MAX_NUM_VNFS)
    max_vnf_cpu = math.ceil(VNF_MAX_CPU_DEMAND_PER_BW * max_rate / 1_000)
    max_total_cpus = max_vnf_cpu * MAX_NUM_VNFS
    max_cpu_cost = max_total_cpus * COST_PER_CPU

    # Cost of active nodes
    max_active_cost = NUM_NODES * COST_ACTIVE_NODE

    # Cost of link bandwidth
    max_link_bw_in_gb = LINK_MAX_BW_CAPACITY / 1_000
    max_link_bw_cost = max_link_bw_in_gb * LINK_MAX_COST_PER_GB
    max_bw_cost = max_link_bw_cost * len(LINK_TUPLES)

    # Compute cost of new VNF instances and total cost
    max_extra_vnfs = MAX_NUM_VNFS * MAX_NUM_INSTANCES_PER_VNF
    max_extra_vnfs_cost = max_extra_vnfs * VNF_LICENSE_COST
    
    # Cost of migration
    max_vnf_vm_size = (max_vnf_cpu * VM_DISK_PER_CPU) + (max_vnf_cpu * VM_RAM_PER_CPU)
    max_node_mig_cost = (max_vnf_vm_size * LINK_MAX_COST_PER_GB) * MAX_HOPS
    max_migration_cost = max_node_mig_cost * MAX_NUM_VNFS
    
    return math.ceil(max_cpu_cost + max_active_cost + max_bw_cost + max_extra_vnfs_cost + max_migration_cost)
