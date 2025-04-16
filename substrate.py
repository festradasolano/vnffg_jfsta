from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

# REFERENCES
#
# S. Agarwal, M. Kodialam, and T. Lakshman, “Traffic engineering in software
# defined networks,” in Proc. IEEE Conference on Computer Communications
# (INFOCOM), 2013, pp. 2211–2219.
#
# G. Wang, G. Feng, T. Q. Quek, S. Qin, R. Wen, and W. Tan, “Reconfiguration in
# network slicing—optimizing the profit and performance,” IEEE Transactions on
# Network and Service Management, vol. 16, no. 2, pp. 591–605, 2019.
#
# TODO (Kodirov et al., 2018) VNF Chain Allocation and Management at Data Center Scale
#
# N. T. Jahromi, S. Kianpisheh, and R. H. Glitho, “Online VNF placement and chaining
# for value-added services in content delivery networks,” in Proc. IEEE International
# Symposium on Local and Metropolitan Area Networks (LANMAN), 2018, pp. 19–24.
#
# TODO (Alleg et al., 2017) Delay-aware VNF Placement and Chaining based on a Flexible Resource Allocation Approach

# CONSTANTS
NUM_NODES = 15  # (Agarwal et al., 2013)
MIN_NUM_VNF_CAPABLE_NODES = 6  # (Wang et al., 2019)
NODE_MIN_CPU_CAPACITY = 0
NODE_MAX_CPU_CAPACITY = 20  # (Kodirov et al., 2018)
LINK_MIN_BW_CAPACITY = 100
LINK_MAX_BW_CAPACITY = 1_000  # (Wang et al., 2019)
# LINK_MIN_BW_CAPACITY = 1_000
# LINK_MAX_BW_CAPACITY = 10_000  # (Jahromi et al., 2018)
LINK_MIN_DELAY = 10
LINK_MAX_DELAY = 10  # (Alleg et al., 2017)
# LINK_MIN_DELAY = 4
# LINK_MAX_DELAY = 50  # (Jahromi et al., 2018)
LINK_MIN_COST_PER_GB = 0.09
LINK_MAX_COST_PER_GB = 0.115  # (Jahromi et al., 2018)
LINK_TUPLES = [
    (1, 2), (1, 3), (1, 4), (2, 3),
    (2, 5), (2, 11), (3, 4), (3, 6),
    (3, 7), (4, 8), (4, 9), (5, 12),
    (6, 7), (6, 10), (6, 11), (7, 9),
    (7, 10), (8, 9), (9, 10), (9, 15),
    (10, 11), (10, 13), (10, 14), (11, 12),
    (11, 13), (12, 13), (13, 14), (14, 15)
]  # (Agarwal et al., 2013)
MAX_HOPS = 4

# -----------------------------------------------------------------------------
# TODO REMOVE
# NUM_NODES = 5
# MIN_NUM_VNF_CAPABLE_NODES = 0
# LINK_TUPLES = [
#     (1, 2), (1, 3), (1, 4), (2, 3), (2, 5)
# ]
# MAX_HOPS = 2
# -----------------------------------------------------------------------------


class Node:
    """
    A class used to represent a Node.

    ...

    Attributes
    ----------
    id : str
        The ID of the node
    cpu_capacity : int
        The available CPU capacity of the node, in computing units
    is_vnf_capable : bool
        Specifies if the node is capable of hosting VNFs

    Methods
    -------
    """

    def __init__(self, id: str, cpu_capacity: int, is_vnf_capable: bool):
        """
        Parameters
        ----------
        id : str
            The ID of the node
        cpu_capacity : int
            The available CPU capacity of the node, in computing units
        is_vnf_capable : bool
            Specifies if the node is capable of hosting VNFs
        """
        self.id = id
        self.cpu_capacity = cpu_capacity
        self.is_vnf_capable = is_vnf_capable


class Link:
    """
    A class used to represent a Link between a pair of Nodes.

    ...

    Attributes
    ----------
    id : set
        The ID of the link represented with a set of two Node IDs
    source_node_id : str
        The source Node ID of the link
    target_node_id: str
        The target Node ID of the link
    bw_capacity : int
        The available bandwidth capacity of the link, in Mbps
    tx_delay : int
        The transmission delay of the link, in ms
    cost_per_gb : float
        The cost per GB of the link, in dollars

    Methods
    -------
    """

    def __init__(self, id: set, source_node_id: str, target_node_id: str, bw_capacity: int, tx_delay: int, cost_per_gb: float):
        """
        Parameters
        ----------
        id : set
            The ID of the link represented with a set of two Node IDs
        source_node_id : str
            The source Node ID of the link
        target_node_id: str
            The target Node ID of the link
        bw_capacity : int
            The available bandwidth capacity of the link, in Mbps
        tx_delay : int
            The transmission delay of the link, in ms
        cost_per_gb : float
            The cost per GB of the link, in dollars
        """
        self.id = id
        self.source_node_id = source_node_id
        self.target_node_id = target_node_id
        self.bw_capacity = bw_capacity
        self.tx_delay = tx_delay
        self.cost_per_gb = cost_per_gb


class Network:
    """
    A class used to represent a Network of Nodes and Links.

    ...

    Attributes
    ----------
    nodes : dict
        A collection of key:value pairs, where the key is the Node ID and the
        value is the corresponding Node object
    links : dict
        A collection of key:value pairs, where the key is the Link ID and the
        value is the corresponding Link object
    neighbors: dict
        A collection of key:value pairs, where the key is the Node ID and the
        value is a list of its neighboring Node IDs

    Methods
    -------
    get_nodes_id():
        Returns a list of all the Nodes ID
    
    get_vnf_capable_nodes():
        Returns a list of the Nodes that can host VNFs
    
    print():
        Prints the information of the Network, including Nodes and Links
    """

    def __init__(self, nodes: "dict[str, Node]", links: "dict[str, Link]"):
        """
        Parameters
        ----------
        nodes : dict
            A collection of key:value pairs, where the key is the Node ID and
            the value is the corresponding Node object
        links : dict
            A collection of key:value pairs, where the key is the Link ID and
            the value is the corresponding Link object
        """
        self.nodes: dict[str, Node] = nodes
        self.links: dict[str, Link] = links

        # Build list of VNF capable nodes
        self.vnf_capable_nodes: list[Node] = []
        for node_id in self.nodes:
            node = self.nodes[node_id]
            if node.is_vnf_capable and node.cpu_capacity > NODE_MIN_CPU_CAPACITY:
                self.vnf_capable_nodes.append(node)

        # Build map of neighbors, which associates each Node ID with a list of
        # neighbors Node ID
        self.neighbors: dict[str, list[str]] = dict()
        for node_id in self.nodes:
            self.neighbors[node_id] =[]
        for link in self.links.values():
            self.neighbors[link.source_node_id].append(link.target_node_id)
            self.neighbors[link.target_node_id].append(link.source_node_id)
        
        # Map of node neighbors per number of hops for each node, which
        # associates the Node ID with a map that associates the number of hops
        # with the list of reachable neighbors Node ID
        self.nodes_per_hops: dict[str, dict[int, list]] = dict()

        # Map of paths between a pair of nodes, which associates a tuple
        # (Node ID, Node ID) with the list of paths between them
        self.node_pair_paths: dict[tuple[str, str], list] = dict()

        # Build the maps of neighbors per hops and paths between nodes
        self.__build_hops_paths()

    def __build_hops_paths(self, max_hops: int = MAX_HOPS):
        """
        TODO
        """
        # Map to track visited nodes
        visited = {}
        for n in self.nodes:
            visited[n] = False
        
        # Initialize map of paths between nodes
        for u in self.nodes:
            for v in self.nodes:
                if u != v:
                    self.node_pair_paths[(u, v)] = []
        
        for node_id in self.nodes:
            # Initialize map of neighbors per hops
            self.nodes_per_hops[node_id] = dict()
            for i in range(1, max_hops+1):
                self.nodes_per_hops[node_id][i] = []
            
            # Call recursive function
            self.__build_hops_paths_util(node_id, node_id, visited, 0, max_hops, [])

    def __build_hops_paths_util(self, source: str, current: str, visited: dict[str, bool], hop: int, max_hops: int, path: list):
        """
        TODO
        """
        # Condition to stop recursive function
        hop += 1
        if hop > max_hops:
            return
        
        # Mark current node as visited
        visited[current] = True

        for neighbor in self.neighbors[current]:
            # Check if neighbor has been already visited
            if visited[neighbor] == False:
                # Add neighbor to the list of current number of hops
                self.nodes_per_hops[source][hop].append(neighbor)

                # Add link to path and add path to the pair of nodes
                path.append({current, neighbor})
                self.node_pair_paths[(source, neighbor)].append(path.copy())

                # Call recursive function
                self.__build_hops_paths_util(source, neighbor, visited, hop, max_hops, path)

                # Remove last link from path
                path.pop()

        # Mark current node as not visited
        visited[current] = False

    def get_neighbors_per_node(self, max_hops: int = MAX_HOPS):
        """
        TODO
        """
        neighbors_per_node: dict[str, set[str]] = dict()
        for node_id in self.nodes_per_hops:
            neighbors = set()
            for hop in range(1, max_hops+1):
                for neighbor_id in self.nodes_per_hops[node_id][hop]:
                    neighbors.add(neighbor_id)
            neighbors_per_node[node_id] = neighbors
        return neighbors_per_node
    
    def get_nodes_id(self) -> "list[str]":
        """
        Returns a list of all the Nodes ID.

        Parameters
        ----------
        None

        Returns
        -------
        list:
            A list of the Nodes ID
        """
        return list(self.nodes.keys())
    
    def get_node_pair_paths(self, max_hops: int = MAX_HOPS) -> tuple[dict[tuple[str, str], list], int]:
        """
        TODO
        """
        max_paths_per_node_pair = 0
        node_pair_paths = dict()
        for node_pair in self.node_pair_paths:
            paths_per_node_pair = list()
            for path in self.node_pair_paths[node_pair]:
                if len(path) <= max_hops:
                    paths_per_node_pair.append(path)
            node_pair_paths[node_pair] = paths_per_node_pair

            if len(paths_per_node_pair) > max_paths_per_node_pair:
                max_paths_per_node_pair = len(paths_per_node_pair)
        
        return node_pair_paths, max_paths_per_node_pair
    
    def get_paths(self, max_hops: int = MAX_HOPS) -> tuple[list[tuple[tuple, int]], int]:
        """
        TODO
        """
        paths: list[tuple[tuple, int]] = list()
        max_node_pair_paths = 0
        for node_pair in self.node_pair_paths:
            len_node_pair_paths = len(self.node_pair_paths[node_pair])
            for i in range(len_node_pair_paths):
                path = self.node_pair_paths[node_pair][i]
                if len(path) <= max_hops:
                    paths.append((node_pair, i))
            if len_node_pair_paths > max_node_pair_paths:
                max_node_pair_paths = len_node_pair_paths
        return paths, max_node_pair_paths

    def get_shortest_path_between_nodes(self, source_node: str, target_node: str, rs: RandomState = None, seed: int = None) -> list:
        """
        TODO
        """
        # Create new seeded random generator, if non passed
        if rs == None:
            rs = RandomState(MT19937(SeedSequence(seed)))

        # Return empty path if same source and target
        if source_node == target_node:
            return []
        
        # Find shortest path between source and target
        paths = self.node_pair_paths[source_node, target_node]
        path = paths[0]
        for p in paths:
            if len(path) == 1:
                return path
            elif len(p) < len(path):
                path = p
            elif len(p) == len(path):
                rand_i = rs.randint(2)
                if rand_i == 1:
                    path = p
        return path
    
    def get_vnf_capable_nodes_from_source(self, source_node: str, hops: int, vnf_cpu_demand: int = 1) -> "list[Node]":
        """
        TODO
        """
        # Validate number of hops
        if hops < 0 or hops > MAX_HOPS:
            raise ValueError("Invalid number of hops: {}".format(hops))
        
        # If no hops, return source node if it can host requested VNF demand
        if hops == 0:
            node = self.nodes[source_node]
            if node.is_vnf_capable and node.cpu_capacity >= vnf_cpu_demand:
                return [node]
            return []
        
        # Return neighboring nodes of source node within hops that can host requested VNF demand
        node_neighbors = self.nodes_per_hops[source_node][hops]
        vnf_capable_nodes = []
        for node_id in node_neighbors:
            node = self.nodes[node_id]
            if node.is_vnf_capable and node.cpu_capacity >= vnf_cpu_demand:
                vnf_capable_nodes.append(node)
        return vnf_capable_nodes
    
    def info(self):
        """
        Prints the information of the Network, including Nodes and Links.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        print("-"*80)
        print("Network:")
        print("  Nodes:")
        for node in self.nodes.values():
            print("    {}: CPU = {}, Support VNF = {}".format(node.id, node.cpu_capacity, node.is_vnf_capable))
        print("  Links:")
        for link in self.links.values():
            print("    {}: BW = {}, Delay = {}".format(link.id, link.bw_capacity, link.tx_delay))


def build_random_network(seed: int = None) -> Network:
    """Builds a random substrate Network.

    Parameters
    ----------
    seed : int, optional
        The seed for the random generator
    
    Returns
    -------
    Network:
        A random substrate Network
    """
    # Create new seeded random generator
    rs = RandomState(MT19937(SeedSequence(seed)))

    # Randomly select VNF capable nodes
    num_vnf_capable_nodes = rs.randint(MIN_NUM_VNF_CAPABLE_NODES, NUM_NODES+1)
    nodes_idx = range(1, NUM_NODES+1)
    vnf_capable_nodes = rs.choice(nodes_idx, size=num_vnf_capable_nodes, replace=False)

    # Build map of nodes
    nodes = {}
    for i in nodes_idx:
        # Node ID and CPU
        node_id = "n" + str(i)
        node_cpu_capacity = rs.randint(NODE_MIN_CPU_CAPACITY, NODE_MAX_CPU_CAPACITY+1)

        # Check if node is VNF capable
        is_vnf_capable = False
        if i in vnf_capable_nodes:
            is_vnf_capable = True
        
        node = Node(node_id, node_cpu_capacity, is_vnf_capable)
        nodes[node_id] = node
    
    # Build list of edges
    links = {}
    for t in LINK_TUPLES:
        # Build ID of source and target nodes
        source_id = "n" + str(t[0])
        target_id = "n" + str(t[1])

        # Link ID, bandwidth, delay and cost
        # TODO add costs
        link_id = {source_id, target_id}
        link_bw_capacity = rs.randint(LINK_MIN_BW_CAPACITY, LINK_MAX_BW_CAPACITY+1)
        link_tx_delay = rs.randint(LINK_MIN_DELAY, LINK_MAX_DELAY+1)
        link_cost_per_gb = rs.randint((LINK_MIN_COST_PER_GB * 1_000), (LINK_MAX_COST_PER_GB * 1_000)+1) / 1_000
        link = Link(link_id, source_id, target_id, link_bw_capacity, link_tx_delay, link_cost_per_gb)
        links[frozenset(link_id)] = link
    
    # Build and return network
    return Network(nodes, links)
