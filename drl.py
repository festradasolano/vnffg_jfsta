from __future__ import annotations

import math
import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import tensorflow as tf
from typing import Any

class DLModel(tf.keras.Model):
    """
    TODO
    """

    def __init__(self, input_units: int, output_units: int, hidden_layers: "list[int]"):
        """
        TODO
        """
        super(DLModel, self).__init__()
        # Input layer
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(input_units,))
        # Hidden layers
        self.hidden_layers = []
        for hidden_units in hidden_layers:
            self.hidden_layers.append(tf.keras.layers.Dense(hidden_units, activation="tanh"))
        # Output layer
        self.output_layer = tf.keras.layers.Dense(output_units, activation="linear")
    
    @tf.function
    def call(self, inputs):
        """
        Implements the model's forward pass.
        TODO
        """
        x = self.input_layer(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

class DQN:
    """
    TODO
    """
    # CONSTANTS
    STATE_KEY = "state"
    ACTION_KEY = "action"
    REWARD_KEY = "reward"
    NEXT_STATE_KEY = "next_state"
    NEXT_ACTION_MASK_KEY = "next_action_mask"
    DONE_KEY = "done"

    # HYPERPARAMETERS
    HIDDEN_LAYERS = [128, 128]
    DISCOUNT_FACTOR = 0.95
    MAX_EXPERIENCES = 10_000
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3

    def __init__(self, len_state: int, len_action: int, hidden_layers: "list[int]" = HIDDEN_LAYERS, discount_factor: float = DISCOUNT_FACTOR, max_experiences: int = MAX_EXPERIENCES, batch_size: int = BATCH_SIZE, learning_rate: float = LEARNING_RATE, seed: int = None):
        """
        TODO
        """
        self.rs = RandomState(MT19937(SeedSequence(seed)))

        # Build DL model
        self.model = DLModel(len_state, len_action, hidden_layers)
        self.len_action = len_action

        # Initialize DQN parameters
        self.discount_factor = discount_factor
        self.max_experiences = max_experiences
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(learning_rate)
        self.replay_data: list[dict[str, Any]] = []
    
    def add_experience(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        TODO
        """
        # Build experience sample to add to replay data
        experience = {
            self.STATE_KEY: state,
            self.ACTION_KEY: action,
            self.REWARD_KEY: reward,
            self.NEXT_STATE_KEY: next_state,
            self.DONE_KEY: done
        }
        
        # If more than max experiences, remove oldest one
        if len(self.replay_data) >= self.max_experiences:
            self.replay_data.pop(0)
        self.replay_data.append(experience)

    def copy_weights(self, from_model: DQN):
        """
        TODO
        """
        vars_model = self.model.trainable_variables
        vars_model_from = from_model.model.trainable_variables
        for var_model, var_model_from in zip(vars_model, vars_model_from):
            var_model.assign(var_model_from.numpy())
    
    def get_action(self, state: np.ndarray, action_mask: np.ndarray = None) -> int:
        """
        Returns the action that maximizes the Q-value.

        TODO
        """
        # Predict Q-values for every possible action
        q_values = self.predict(state)[0].numpy()

        # Select the action that maximizes the Q-Value (apply action mask, if any)
        if action_mask is not None:
            q_values[np.where(action_mask == 0)] = -math.inf
        return np.argmax(q_values)
    
    def predict(self, inputs):
        """
        TODO
        """
        return self.model(np.atleast_2d(inputs.astype("float32")))
    
    def train(self, target_net: DQN) -> float:
        """
        TODO
        Based on https://medium.com/@aniket.tcdav/deep-q-learning-with-tensorflow-2-686b700c868b
        """
        # Skip training if not enough experiences
        if len(self.replay_data) < self.batch_size:
            return 0
        
        # Get random batch of experiences
        batch = self.rs.choice(self.replay_data, size=self.batch_size, replace=False)
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = list(), list(), list(), list(), list()
        for sample in batch:
            batch_states.append(sample[self.STATE_KEY])
            batch_actions.append(sample[self.ACTION_KEY])
            batch_rewards.append(sample[self.REWARD_KEY])
            batch_next_states.append(sample[self.NEXT_STATE_KEY])
            batch_dones.append(sample[self.DONE_KEY])        
        
        # Convert batch to numpy array
        states = np.array(batch_states)
        actions = np.array(batch_actions)
        rewards = np.array(batch_rewards)
        # rewards = rewards / 10
        next_states = np.array(batch_next_states)
        dones = np.array(batch_dones)

        # Get target Q values, applying action mask
        q_values_next = np.max(target_net.predict(next_states), axis = 1)
        target_q_values = np.where(dones, rewards, rewards + self.discount_factor * q_values_next)
        target_q_values = tf.convert_to_tensor(target_q_values, dtype = "float32")

        # Predict Q values and compute MSE loss
        with tf.GradientTape() as tape:
            q_values = tf.math.reduce_sum(self.predict(states) * tf.one_hot(actions, self.len_action), axis=1)
            loss = tf.math.reduce_mean(tf.square(target_q_values - q_values))

        # Train Q network and return MSE loss
        variables = self.model.trainable_variables
        tape.watch(variables)
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss.numpy()
    
    # @staticmethod
    # def convert2_scalar(actions: "np.ndarray"):
    #     scalars = []
    #     for action in actions:
    #         if np.isscalar(action):
    #             scalar = action
    #         else:
    #             a = action[0]
    #             b = action[1]
    #             scalar = a*15 + b
    #         scalars.append(scalar)
    #     return scalars
    
    # @staticmethod
    # def scale(data: np.ndarray, max_value: float, min_value: float = 0):
    #     # data_std = (data - min_value) / (max_value - min_value)
    #     # return data_std * 10 - 5
    #     return (100 * (data - min_value) / (max_value - min_value)) - 50