import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network, network
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
import numpy as np
from tf_agents.networks import sequential

class CustomQNetwork(network.Network):
    def __init__(self, input_tensor_spec, action_spec, name='CustomQNetwork'):
        super(CustomQNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)

        self.action_spec = action_spec
        
        self.dense_layers = sequential.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(action_spec.maximum - action_spec.minimum + 1)
        ])

    def call(self, observation, step_type=None, network_state=()):
        del step_type  # unused
        x = tf.cast(observation, tf.float32)
        logits = self.dense_layers(x)
        return logits, network_state

class RLAgent:
    def __init__(self, env):
        self.env = env
        
        self.q_net = CustomQNetwork(
            env.observation_spec(),
            env.action_spec()
        )
        
        self.agent = dqn_agent.DqnAgent(
            env.time_step_spec(),
            env.action_spec(),
            q_network=self.q_net,
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-3),
            td_errors_loss_fn=common.element_wise_squared_loss
        )
        self.agent.initialize()
        
        # Initialize replay buffer
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=1,
            max_length=1000)

    def train(self, num_iterations):
        for _ in range(num_iterations):
            time_step = self.env.reset()
            while not time_step.is_last():
                action_step = self.agent.policy.action(time_step)
                next_time_step = self.env.step(action_step.action)
                traj = trajectory.from_transition(time_step, action_step, next_time_step)
                self.replay_buffer.add_batch(traj)
                time_step = next_time_step

            experience = self.replay_buffer.gather_all()
            train_loss = self.agent.train(experience)
            self.replay_buffer.clear()

    def act(self, observation):
        time_step = self.env.reset()
        observation = tf.convert_to_tensor([observation], dtype=tf.float32)
        action_step = self.agent.policy.action(time_step._replace(observation=observation))
        return action_step.action.numpy()[0]

    def update(self, new_data):
        self.env._data = new_data
        self.train(num_iterations=1)

    def save_model(self, path):
        tf.saved_model.save(self.agent, path)

    def load_model(self, path):
        self.agent = tf.saved_model.load(path)

class TradingEnvironment(py_environment.PyEnvironment):
    def __init__(self, data):
        super(TradingEnvironment, self).__init__()
        self._data = data
        self._episode_ended = False
        
        if self._data is None or self._data.empty:
            raise ValueError("Invalid or empty data provided to TradingEnvironment")
        
        self._state = self._get_initial_state()

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(5,), dtype=np.float32, minimum=-np.inf, maximum=np.inf, name='observation')

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def time_step_spec(self):
        return ts.time_step_spec(self.observation_spec())

    def _get_initial_state(self):
        return self._data.iloc[0].values[:5].astype(np.float32)

    def _reset(self):
        self._state = self._get_initial_state()
        self._episode_ended = False
        return ts.restart(self._state)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        # Implement trading logic and state transition
        # Return reward and next state
        # This is a simplified example
        reward = 0
        if action == 1:  # Buy
            reward = self._data.iloc[1]['Close'] - self._data.iloc[0]['Close']
        elif action == 2:  # Sell
            reward = self._data.iloc[0]['Close'] - self._data.iloc[1]['Close']

        self._state = self._data.iloc[1].values[:5].astype(np.float32)
        
        if len(self._data) <= 2:
            self._episode_ended = True
            return ts.termination(self._state, reward)
        else:
            self._data = self._data.iloc[1:]
            return ts.transition(self._state, reward, discount=0.99)
