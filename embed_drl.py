from __future__ import annotations

from drl import DQN
from embedding import Embedding, VNF_LICENSE_COST
from substrate import Network, NUM_NODES
from vnfr import VNFR

import gymnasium as gym
import math
import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import os
from pathlib import Path
import re
import tensorflow as tf
from typing import Any


# Constants
MAX_HOPS = 3
ACTION_MASK_KEY = "action_mask"
IS_CANDIDATE_M2_KEY = "is_candidate_m2"
MODIFICATION_COST_KEY = "modification_cost"


class DRLEmbedding(Embedding):
    """
    TODO
    """

    def __init__(self, network: Network, vnfr: VNFR, node_pair_paths: dict[tuple, str], max_hops: int = MAX_HOPS, seed: int = None):
        """
        TODO
        """
        super().__init__(list(vnfr.vnfs.keys()), network.vnf_capable_nodes, list(network.links.keys()), seed)
        self.network = network
        self.vnfr = vnfr
        self.max_hops = max_hops
        self.node_pair_paths = node_pair_paths

        self.embeded_vnfs = set()
        # self.usable_vnfs = vnfr.dependants[self.NO_DEPENDENCY].copy()
        self.vnf_capable_node_ids = [node.id for node in network.vnf_capable_nodes]

    def embed_vnf_node_path(self, vnf_id: str, node_id: str, path: tuple[tuple, int]) -> bool:
        """
        TODO
        """
        raise NotImplementedError("Subclass must implement this method")
    
    def _compute_node_available_cpu(self, node_id: str) -> int:
        """
        TODO
        """
        node = self.network.nodes[node_id]
        node_cpu_available = node.cpu_capacity - self.allocated_node_cpus[node_id][self.TOTAL_KEY]
        self._update_vnf_capable_nodes(node_id, node_cpu_available)
        return node_cpu_available
    
    def _update_vnf_capable_nodes(self, node_id: str, node_cpu_available: int):
        if node_cpu_available <= 0:
            node_idx = self.vnf_capable_node_ids.index(node_id)
            self.vnf_capable_node_ids.pop(node_idx)


class DRLEnv(gym.Env):
    """
    TODO
    """

    def __init__(self, m1: Embedding, network: Network, vnfr: VNFR, max_vnf_instances: int, max_hops: int = MAX_HOPS, seed: int = None):
        """
        TODO
        """
        # Construction parameters
        self.seed = seed
        self.rs = RandomState(MT19937(SeedSequence(seed)))
        self.m1 = m1
        self.network = network
        self.vnfr = vnfr
        self._max_vnf_instances = max_vnf_instances
        self.max_hops = max_hops
        
        # Initialize embedding parameters
        self.neighbors_per_node = network.get_neighbors_per_node(self.max_hops)
        self.node_pair_paths, self.max_paths_per_node_pair = network.get_node_pair_paths(self.max_hops)
        self.m2 = None
        self.m1_resource_cost = self.m1.compute_resource_cost(self.network)
        self.reward_coefficient = VNF_LICENSE_COST

        # Initialize counters
        # self.iter = 0
        self.i = 0

        # OBSERVATION AND ACTION SPACES

        # Auxiliary set (number of replicas per VNF to be mapped)
        self._len_vnfs = len(self.vnfr.vnfs)
        self._len_auxiliary = self._len_vnfs
        obs_auxiliary_space = np.full(self._len_auxiliary, self._max_vnf_instances + 1)  # 0 for no replica of VNF

        # Modified mapped forwarding graph (M2):
        # Hosted VNFs
        # self._len_mapped_vnfs = (self._max_vnf_instances * self._len_vnfs) + self._max_vnf_instances
        self._len_mapped_vnfs = self._max_vnf_instances * self._len_vnfs
        obs_mapped_vnfs_space = np.full(self._len_mapped_vnfs, self._len_vnfs + 1)  # 0 for no VNF
        # Hosting nodes
        self._len_mapped_hosts_paths = self._len_mapped_vnfs + self._max_vnf_instances
        self._len_hosting_nodes = NUM_NODES
        # obs_hosting_nodes_space = np.full(self._len_mapped_vnfs, self._len_hosting_nodes + 1)  # 0 for no VNF hosting node
        obs_hosting_nodes_space = np.full(self._len_mapped_hosts_paths, self._len_hosting_nodes + 1)  # 0 for no VNF hosting node
        # Entering paths
        self._len_paths = self.max_paths_per_node_pair
        # obs_inputpaths_space = np.full(self._len_mapped_vnfs, self._len_paths + 1)  # 0 for no entering path, including same hosting node
        obs_inputpaths_space = np.full(self._len_mapped_hosts_paths, self._len_paths + 1)  # 0 for no entering path, including same hosting node

        # Observation space = auxiliary set + hosted VNFs + hosting nodes + entering paths
        obs_space = np.concatenate((obs_auxiliary_space, obs_mapped_vnfs_space, obs_hosting_nodes_space, obs_inputpaths_space), axis=0)
        self.observation_space = gym.spaces.MultiDiscrete(obs_space, seed=seed)

        # Action space = VNF + hosting node + entering path of hosting node
        self._len_action_vnfs = self._len_vnfs + 1   # 0 for no VNF to embed (i.e., path to target node)
        self._len_action_nodes = NUM_NODES
        self._len_action_paths = (self.max_paths_per_node_pair * self._max_vnf_instances) + 1  # 0 for no path (i.e., same hosting node)
        act_space = [self._len_action_vnfs, self._len_action_nodes, self._len_action_paths]
        self.action_space = gym.spaces.MultiDiscrete(act_space, seed=seed)
        self._len_action = np.prod(self.action_space.nvec)
    
    def action_sample(self, mask: np.ndarray = None) -> int:
        """
        TODO
        """
        # If no mask, sample any value from action space
        if mask is None:
            action = self.action_space.sample()
            return self._action_space_to_action_idx(action[0], action[1], action[2])
        
        # Sample valid action from action mask and parse to action space
        valid_actions = np.nonzero(mask)[0]
        return self.rs.choice(valid_actions)

    # def reset(self, vnf_instances: int):
    def reset(self):
        """
        TODO
        """
        # We need the following line to seed self.np_random
        super().reset(seed=self.seed)

        # Increment episode and initialize iteration
        # self.iter += 1
        self.i = 0

        # Initialize original auxiliary set with VNF replicas
        self._auxiliary = np.full(self._len_auxiliary, self._max_vnf_instances)
        # self._auxiliary = np.full(self._len_auxiliary, vnf_instances)

        # Reset observation space and reward
        self._mapped_vnfs = np.full(self._len_mapped_vnfs, 0)
        self._hosting_nodes = np.full(self._len_mapped_hosts_paths, 0)
        self._inputpaths = np.full(self._len_mapped_hosts_paths, 0)
        self.reward = 0

    def step(self, action_idx: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        TODO
        """
        raise NotImplementedError("Subclass must implement this method")

    def _action_idx_to_action_space(self, action_idx: int) -> tuple[int, int, int]:
        """
        TODO
        """
        action_vnf_divisor = self._len_action_nodes * self._len_action_paths
        action_node_divisor = self._len_action_paths
        action_vnf_idx = action_idx // action_vnf_divisor
        action_node_idx = (action_idx % action_vnf_divisor) // action_node_divisor
        action_path_idx = (action_idx % action_vnf_divisor) % action_node_divisor
        return (action_vnf_idx, action_node_idx, action_path_idx)
    
    def _action_space_to_action_idx(self, action_vnf_idx: int, action_node_idx: int, action_path_idx: int) -> int:
        """
        TODO
        """
        action_vnf_mutiplier = self._len_action_nodes * self._len_action_paths
        action_node_multiplier = self._len_action_paths
        return (action_vnf_idx * action_vnf_mutiplier) + (action_node_idx * action_node_multiplier) + action_path_idx

    def _get_info(self, action_mask: np.ndarray = None, is_candidate_m2: bool = False, modification_cost: float = 0) -> dict[str, Any]:
        """
        TODO
        """
        # if action_mask is None:
        #     action_mask = np.full(self._len_action, 0, dtype=np.int8)
        return {
            ACTION_MASK_KEY: action_mask,
            IS_CANDIDATE_M2_KEY: is_candidate_m2,
            MODIFICATION_COST_KEY: modification_cost
            }

    def _get_obs(self) -> np.ndarray:
        """
        TODO
        """
        # auxiliary = self.scale(self._auxiliary, self._max_vnf_instances)
        # mapped_vnfs = self.scale(self._mapped_vnfs, self._len_vnfs)
        # hosting_nodes = self.scale(self._hosting_nodes, self._len_hosting_nodes)
        # input_paths = self.scale(self._inputpaths, self._len_paths)
        # return np.concatenate((auxiliary, mapped_vnfs, hosting_nodes, input_paths), axis=0)
        # print(self._mapped_vnfs)
        return np.concatenate((self._auxiliary, self._mapped_vnfs, self._hosting_nodes, self._inputpaths), axis=0)
    
    @staticmethod
    def get_idx(id: str) -> int:
        """
        TODO
        """
        if isinstance(id, int):
            return id
        elif not isinstance(id, str):
            return 0
        
        has_idx = re.search(r"[0-9]+", id)
        if has_idx:
            return int(has_idx.group())
        return 0
    
    @staticmethod
    def scale(data: np.ndarray, max_value: float, min_value: float = 0):
        # std = (data - min_value) / (max_value - min_value)
        return (2 * (data - min_value) / (max_value - min_value)) - 1


class DRLRun():
    """
    TODO
    """
    # CONSTANTS
    RESULTS_DIR_NAME = "jfstaf_results"
    EPISODE_KEY = "episode"
    LOSS_KEY = "loss"
    AVG_REWARD_KEY = "avg_reward"
    SUCCESS_RATE_KEY = "success_rate"
    AVG_SUCCESS_COST_KEY = "avg_success_cost"
    EPSILON_KEY = "epsilon"
    MAX_AVG_REWARD_KEY = "max_avg_reward"
    MAX_SUCCESS_RATE_KEY = "max_success_rate"
    MILESTONE_EPISODE = 100

    # PARAMETERS
    MAX_EPISODES = 25_000
    EPSILON = 1
    MIN_EPSILON = 0.01
    MAX_TRAINING_PATIENCE = 10
    MIN_TRAINING_PATIENCE_SUCCESS = 0.9
    TESTING_EPISODES = 30

    # HYPERPARAMETERS
    EPSILON_DECAY = 0.9998
    TARGET_NET_UPDATE_RATE = 400

    # def __init__(self, env: DRLEnv, filename: str, max_vnf_instances: int = MAX_VNF_INSTANCES, seed: int = None):
    def __init__(self, env: DRLEnv, filename: str, seed: int = None):
        """
        TODO
        """
        self.rs = RandomState(MT19937(SeedSequence(seed)))
        self.seed = seed
        self.env = env
        self.filename = filename

        # Obtain length of states and actions from environment
        len_state = len(self.env.observation_space.nvec)
        len_action = np.prod(self.env.action_space.nvec)

        # Build Q network and Target network
        self.qnet = DQN(len_state, len_action, seed=seed)
        self.target_net = DQN(len_state, len_action, seed=seed)
        self.target_net.copy_weights(self.qnet)

        #
        # self._max_vnf_instances = max_vnf_instances
        # self._rewards_vnf_instances = np.full(max_vnf_instances, -math.inf)

    def run(self, max_episodes: int = MAX_EPISODES, epsilon: float = EPSILON, epsilon_decay: float = EPSILON_DECAY, min_epsilon: float = MIN_EPSILON, target_net_update_rate: int = TARGET_NET_UPDATE_RATE, testing_episodes: int = TESTING_EPISODES) -> bool:
        """
        TODO
        """
        # Open file to write results and write column names
        path_results_dir = "{0}/{1}".format(str(Path.home()), self.RESULTS_DIR_NAME)
        if not os.path.exists(path_results_dir):
            os.makedirs(path_results_dir)
        path_file = "{0}/{1}/{2}".format(str(Path.home()), self.RESULTS_DIR_NAME, self.filename)
        file = open(path_file, "w")
        file.write("{0},{1},{2},{3},{4},{5}\n".format(self.EPISODE_KEY, self.LOSS_KEY, self.AVG_REWARD_KEY, self.SUCCESS_RATE_KEY, self.AVG_SUCCESS_COST_KEY, self.EPSILON_KEY))

        # Train the DDQN model
        self.__run_training(max_episodes, epsilon, epsilon_decay, min_epsilon, target_net_update_rate, file)

        # Check if using trained DDQN model to run testing
        if testing_episodes <= 0:
            # Close results file and exit
            file.close()
            return
        
        # Test the trained DDQN model
        self.__run_testing(testing_episodes, file)
        
        # Close results file and exit
        file.close()
    
    def run_tunning(self, hparams: list[dict[str, Any]]) -> dict[str, Any]:
        """
        TODO
        """
        # Get hyperparameters
        n_hidden_layers = hparams["hidden_layers"]
        hidden_neurons = 2 ** hparams["hidden_neurons_pow2"]
        learning_rate = 10 ** hparams["learning_rate_pow10"]
        discount_factor = hparams["discount_factor"]
        batch_size = 2 ** hparams["batch_size_pow2"]
        epsilon_decay = hparams["epsilon_decay"]
        target_net_update_rate = 50 * (2 ** hparams["target_net_update_rate_pow2"])

        # Build hidden layers
        hidden_layers = []
        for i in range(n_hidden_layers):
            hidden_layers.append(hidden_neurons)
        
        print(
            "\tHidden layers:", hidden_layers,
            ", Learning rate:", learning_rate,
            ", Discount factor:", discount_factor,
            ", Batch size:", batch_size,
            ", Epsilon decay:", epsilon_decay,
            ", TargetNet update rate:", target_net_update_rate
        )

        # Obtain length of states and actions from environment
        len_state = len(self.env.observation_space.nvec)
        len_action = np.prod(self.env.action_space.nvec)

        # Rebuild Q network and Target network
        self.qnet = DQN(
            len_state=len_state,
            len_action=len_action,
            hidden_layers=hidden_layers,
            discount_factor=discount_factor,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=self.seed
        )
        self.target_net = DQN(
            len_state=len_state,
            len_action=len_action,
            hidden_layers=hidden_layers,
            discount_factor=discount_factor,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=self.seed
        )
        self.target_net.copy_weights(self.qnet)

        # Run training and return results
        results = self.__run_training(
            max_episodes=self.MAX_EPISODES,
            epsilon=self.EPSILON,
            epsilon_decay=epsilon_decay,
            min_epsilon=self.MIN_EPSILON,
            target_net_update_rate=target_net_update_rate
        )
        print("\tResults:", results)
        return results
    
    def __run_training(self, max_episodes: int, epsilon: float, epsilon_decay: float, min_epsilon: float, target_net_update_rate: int, file: Any = None, max_patience: int = MAX_TRAINING_PATIENCE, min_patience_success: float = MIN_TRAINING_PATIENCE_SUCCESS) -> dict[str, Any]:
        """
        TODO
        """
        # Initialize counter and collectors
        steps = 0
        sum_rewards = 0
        sum_success = 0
        sum_success_cost = 0
        max_avg_reward = -math.inf
        max_success_rate = 0
        patience_counter = 0

        # while self.env.iter < max_episodes:
        for episode in range(1, max_episodes + 1):
            # Use epsilon-greedy to select number of instances per VNF
            # vnf_instances_idx = np.argmax(self._rewards_vnf_instances)
            # if self.rs.random() < epsilon:
            #     vnf_instances_idx = self.rs.randint(self._max_vnf_instances)
            # vnf_instances = vnf_instances_idx + 1

            # Initialize environment using number of instances per VNF
            # current_state, info = self.env.reset(vnf_instances)
            current_state, info = self.env.reset()
            action_mask = info[ACTION_MASK_KEY]
            terminated = False

            while not terminated:
                # Use epsilon-greedy to explore and exploit the actions
                if self.rs.random() < epsilon:
                    # Exploration: sample action from environment
                    action_idx = self.env.action_sample(mask=action_mask)
                else:
                    # Exploitation: get best action from Q network
                    action_idx = self.qnet.get_action(state=current_state, action_mask=action_mask)
                
                # Apply action and increase step
                next_state, reward, terminated, truncated, info = self.env.step(action_idx)
                steps += 1

                # If truncated, there was an error
                if truncated:
                    print("An ERROR occured while applying the action {a} to the state {s}".format(a=action_idx, s=current_state))
                    return False
                
                # Add experience and train Q network
                self.qnet.add_experience(current_state, action_idx, reward, next_state, terminated)
                loss = self.qnet.train(self.target_net)

                # Copy Q network to Target network depending on update rate
                if steps % target_net_update_rate == 0:
                    self.target_net.copy_weights(self.qnet)

                # Get environment info and update current state
                action_mask = info[ACTION_MASK_KEY]
                is_candidate_m2 = info[IS_CANDIDATE_M2_KEY]
                cost = info[MODIFICATION_COST_KEY]
                current_state = next_state
            
            # Collect reward and success, if M2 candidate
            sum_rewards += reward
            if is_candidate_m2:
                sum_success += 1
                sum_success_cost += cost
                
                # Update maximum reward for number of instances per VNF
                # if self._rewards_vnf_instances[vnf_instances_idx] == -math.inf:
                #     self._rewards_vnf_instances[vnf_instances_idx] = reward
                # else:
                #     self._rewards_vnf_instances[vnf_instances_idx] += reward
                #     self._rewards_vnf_instances[vnf_instances_idx] /= 2
            
            # Every milestone episode, compute and write results
            if episode % self.MILESTONE_EPISODE == 0:
                average_reward, success_rate = self.__write_results(
                    episode=episode,
                    sum_rewards=sum_rewards,
                    sum_success=sum_success,
                    sum_success_cost=sum_success_cost,
                    num_episodes=self.MILESTONE_EPISODE,
                    loss=loss,
                    epsilon=epsilon,
                    file=file
                )

                # Reset collectors
                sum_rewards = 0
                sum_success = 0
                sum_success_cost = 0

                # Check maximum average reward and update patience counter for early stop
                if average_reward > max_avg_reward:
                    max_avg_reward = average_reward
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Check maximum success rate
                if success_rate > max_success_rate:
                    max_success_rate = success_rate
                
                # Check for early stop
                if (max_success_rate >= min_patience_success) and (patience_counter >= max_patience):
                    break
            
            # Update epsilon
            epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        return {
            self.MAX_AVG_REWARD_KEY: max_avg_reward,
            self.MAX_SUCCESS_RATE_KEY: max_success_rate,
            self.EPISODE_KEY: episode
            }
    
    def __run_testing(self, testing_episodes: int, file: Any = None):
        """
        TODO
        """
        # Initialize collectors
        sum_rewards = 0
        sum_success = 0
        sum_success_cost = 0

        for episode in range(testing_episodes):
            # Exploit the best number of instances per VNF
            # vnf_instances = np.argmax(self._rewards_vnf_instances) + 1

            # Initialize environment, using the best number of instances per VNF
            # current_state, info = self.env.reset(vnf_instances)
            current_state, info = self.env.reset()
            action_mask = info[ACTION_MASK_KEY]
            terminated = False

            while not terminated:
                # Exploit and apply best action from Q network
                action_idx = self.qnet.get_action(state=current_state, action_mask=action_mask)
                next_state, reward, terminated, truncated, info = self.env.step(action_idx)

                # If truncated, there was an error
                if truncated:
                    print("An ERROR occured while applying the action {a} to the state {s}".format(a=action_idx, s=current_state))
                    return False
                
                # Get environment info and update current state
                action_mask = info[ACTION_MASK_KEY]
                is_candidate_m2 = info[IS_CANDIDATE_M2_KEY]
                cost = info[MODIFICATION_COST_KEY]
                current_state = next_state
            
            # Collect reward and success, if any
            sum_rewards += reward
            if is_candidate_m2:
                sum_success += 1
                sum_success_cost += cost
        
        # Compute and write results
        self.__write_results(
            episode=episode,
            sum_rewards=sum_rewards,
            sum_success=sum_success,
            sum_success_cost=sum_success_cost,
            num_episodes=testing_episodes,
            file=file
        )

        # average_reward = sum_rewards / self.TESTING_EPISODES
        # success_rate = sum_success / self.TESTING_EPISODES
        # average_candidates_cost = math.inf if sum_success == 0 else sum_success_cost / sum_success
        # print("Testing episodes {e}: average reward = {r}, success rate = {s}, success reward rate = {sr}"
        #         .format(e=self.TESTING_EPISODES, r=average_reward, s=success_rate, sr=average_candidates_cost))

    def __write_results(self, episode: int, sum_rewards: float, sum_success: int, sum_success_cost: float, num_episodes: int, loss: float = 0, epsilon: float = 0, file: Any = None) -> tuple[float, float]:
        """
        TODO
        """
        # Compute results
        average_reward = sum_rewards / num_episodes
        success_rate = sum_success / num_episodes
        average_success_cost = math.inf if sum_success == 0 else sum_success_cost / sum_success

        # Print and write results
        print("Episode {i}: loss = {l}, average reward = {avgr}, success rate = {sr}, average success cost = {avgsc}, epsilon = {e}"
                .format(i=episode, l=loss, avgr=average_reward, sr=success_rate, avgsc=average_success_cost, e=epsilon))
        if file is not None:
            file.write("{0},{1},{2},{3},{4},{5}\n".format(episode, loss, average_reward, success_rate, average_success_cost, epsilon))
        
        return average_reward, success_rate
