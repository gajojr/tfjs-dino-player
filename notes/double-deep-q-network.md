# DDQN (Double Deep Q-Network)

DDQN is an extension of the Deep Q-Network (DQN) algorithm, designed to address the overestimation of Q-values inherent in traditional Q-learning algorithms.

## Key Components
- **Two Neural Networks**: DDQN employs two distinct deep neural networks; the online network and the target network.
    - **Online Network**: Updated at every iteration during the training process, this network is utilized to select the action with the highest Q-value for the current state.
    - **Target Network**: Updated less frequently, this network generates the Q-values for the next state.
- **Experience Replay**: This technique stores the agent's experiences at each time step, which includes the current state, selected action, and received reward into a replay buffer. Training batches are randomly sampled from this replay buffer to update the online network, reducing correlation between consecutive samples and stabilizing the learning process.

## Training Process
1. **Initialization**: Initialize the replay buffer with a fixed capacity and both the online and target networks with random weights.
2. **Observation**: Observe the current state of the environment.
3. **Action Selection**: Select a random action with a certain probability; otherwise, select the action with the highest Q-value as predicted by the online network.
4. **Execution**: Execute the selected action, observing the reward and the next state.
5. **Experience Storage**: Store the experience in the replay buffer.
6. **Batch Sampling**: Sample a batch of experiences from the replay buffer.
7. **Preprocessing**: Preprocess the state of each experience by converting it to grayscale, cropping it, and stacking it with the previous three frames to provide better context about the state of the environment.
8. **Q-value Prediction**: Utilize the online network to predict the Q-values for the current state and the target network to predict the Q-values for the next state.
9. **Temporal Difference Error Calculation**: Calculate the temporal difference error between the predicted Q-value and the actual reward plus the discounted maximum Q-value of the next state.
10. **Backpropagation**: Update the weights of the online network to minimize the temporal difference error using backpropagation.
11. **Target Network Update**: Update the target network with the weights of the online network every N iterations.

## Advantages
- The dual network structure and experience replay mechanism in DDQN significantly reduce the overestimation of Q-values, leading to a more stable training process.
- Comparative studies have demonstrated that DDQN tends to outperform DQN across various reinforcement learning tasks, showcasing its enhanced performance capabilities.