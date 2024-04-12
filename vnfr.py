from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

# REFERENCES
# 
# TODO (Kodirov et al., 2018) VNF Chain Allocation and Management at Data Center Scale
#
# TODO (Li et al., 2018) Online Joint VNF Chain Composition and Embedding for 5G Networks
#
# TODO (Alleg et al., 2017) Delay-aware VNF Placement and Chaining based on a Flexible Resource Allocation Approach
#
# N. T. Jahromi, S. Kianpisheh, and R. H. Glitho, “Online VNF placement and chaining
# for value-added services in content delivery networks,” in Proc. IEEE International
# Symposium on Local and Metropolitan Area Networks (LANMAN), 2018, pp. 19–24.

# CONSTANTS
MIN_NUM_VNFS = 2
MAX_NUM_VNFS = 10  # (Kodirov et al., 2018)
VNF_MIN_RATIO_OUT2IN = 0.5
VNF_MAX_RATIO_OUT2IN = 1.5  # (Li et al., 2018)
VNF_MIN_CPU_DEMAND_PER_BW = 1
VNF_MAX_CPU_DEMAND_PER_BW = 4  # (Kodirov et al., 2018)
VNF_MIN_PROCESSING_DELAY = 10
VNF_MAX_PROCESSING_DELAY = 30  # (Alleg et al., 2017)
VNF_MIN_MEMORY = 0.5
VNF_MAX_MEMORY = 2  # (Kodirov et al., 2018)
MIN_DATA_RATE = 100
MAX_DATA_RATE = 1000  # ¿?
MIN_ACCEPTABLE_DELAY = 150
MAX_ACCEPTABLE_DELAY = 600  # (Alleg et al., 2017)
# MIN_ACCEPTABLE_DELAY = 1800
# MAX_ACCEPTABLE_DELAY = 2000  # (Jahromi et al., 2018)
MIN_ACCEPTABLE_THROUGHPUT = 0.8


class VNF:
    """
    A class used to represent a VNF

    ...

    Attributes
    ----------
    id : str
        The ID of the VNF
    ratio_out2in : float
        The ratio of the outgoing data rate to the incoming data rate of the VNF
    cpu_demand_per_bw: Node
        The CPU demand, in computing units, per bandwith unit (1 Gbps)
    processing_delay : int
        The processing delay of the VNF, in ms

    Methods
    -------
    """

    def __init__(self, id: str, ratio_out2in: float, cpu_demand_per_bw: int, processing_delay: int):
        """
        Parameters
        ----------
        id : str
            The ID of the VNF
        ratio_out2in : float
            The ratio of the outgoing data rate to the incoming data rate of the VNF
        cpu_demand_per_bw: Node
            The CPU demand, in computing units, per bandwith unit (1 Gbps)
        processing_delay : int
            The processing delay of the VNF, in ms
        """
        self.id = id
        self.ratio_out2in = round(ratio_out2in, 2)
        self.cpu_demand_per_bw = cpu_demand_per_bw
        self.processing_delay = processing_delay

class VNFR:

    def __init__(self, source_node: str, target_node: str, vnfs: "dict[str, VNF]", dependencies: "dict[str, str]", data_rate: int, max_delay: int):
        self.source_node = source_node
        self.target_node = target_node
        self.vnfs: dict[str, VNF] = vnfs
        self.dependencies: dict[str, str] = dependencies
        self.data_rate = data_rate
        self.max_delay = max_delay

        # Build collection of dependants
        self.dependants: dict[str, list] = dict()
        for v in dependencies.values():
            self.dependants[v] = []
        for k, v in dependencies.items():
            self.dependants[v].append(k)
        
        # Compute throughput and delay of the VNFR
        throughput = data_rate
        self.vnfs_delay = 0
        for vnf in vnfs.values():
            throughput *= vnf.ratio_out2in
            self.vnfs_delay += vnf.processing_delay
        print("****", throughput)
        self.min_throughput = throughput * MIN_ACCEPTABLE_THROUGHPUT
    
    def info(self):
        pass


def build_random_vnfr(nodes_id: list, seed: int = None) -> VNFR:
    """Builds a random VNF request (VNFR).

    Parameters
    ----------
    nodes_id: list
        The list of Node IDs in the substrate network; used to randomly select
        the ingress and egress nodes of the VNFR
    seed : int, optional
        The seed for the random generator
    
    Returns
    -------
    VNFR:
        A random VNFR
    """
    # Create new seeded random generator
    rs = RandomState(MT19937(SeedSequence(seed)))

    # Randomly select the ingress and egress nodes
    source_node, target_node = rs.choice(nodes_id, size=2, replace=False)

    # Randomly build set of VNFs
    num_vnfs = rs.randint(MIN_NUM_VNFS, MAX_NUM_VNFS+1)
    vnfs: dict[str, VNF] = dict()
    for i in range(num_vnfs):
        vnf_id = "f" + str(i)
        vnf_ratio_out2in = rs.uniform(VNF_MIN_RATIO_OUT2IN, VNF_MAX_RATIO_OUT2IN)
        vnf_cpu_demand_per_bw = rs.randint(VNF_MIN_CPU_DEMAND_PER_BW, VNF_MAX_CPU_DEMAND_PER_BW+1)
        vnf_processing_delay = rs.randint(VNF_MIN_PROCESSING_DELAY, VNF_MAX_PROCESSING_DELAY+1)
        vnf = VNF(vnf_id, vnf_ratio_out2in, vnf_cpu_demand_per_bw, vnf_processing_delay)
        vnfs[vnf_id] = vnf
    
    # Randomly build dependencies
    dependencies: dict[str, str] = dict()
    vnfs_in_d = [-1]
    for i in range(num_vnfs):
        vnf_id = "f" + str(i)
        vnf_depends_on = rs.choice(vnfs_in_d)
        if vnf_depends_on == -1:
            dependencies[vnf_id] = ""
        else:
            dependencies[vnf_id] = "f" + str(vnf_depends_on)
        vnfs_in_d.append(i)
    
    # Randomly select ingress data rate
    data_rate = rs.randint(MIN_DATA_RATE, MAX_DATA_RATE+1)

    # Randomly select maximum acceptable delay
    max_delay = rs.randint(MIN_ACCEPTABLE_DELAY, MAX_ACCEPTABLE_DELAY+1)

    # Build and return VNFR
    return VNFR(source_node, target_node, vnfs, dependencies, data_rate, max_delay)

# build_random(None, 1000)