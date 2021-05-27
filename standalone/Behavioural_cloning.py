"""
A baseline solution using behavioural cloning in the research track of NeurIPS 2021 MineRL Diamond competition.
With default parameters it trains in 15-20 mins on a machine with a GeForce RTX 2080 Ti GPU.
It uses less than 16GB RAM, achieves an average reward of 1.8 and sometimes obtains cobblestone (35 total reward).
You can adjust RAM usage to fit your specifications by changing the DATA_SAMPLES parameter below.
"""

import random

from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
import torch as th
from torch import nn
import gym
import minerl


# Parameters:
DATA_DIR = "data"  # path to MineRL dataset (should contain "MineRLObtainIronPickaxeVectorObf-v0" directory).
EPOCHS = 2  # how many times we train over dataset.
LEARNING_RATE = 0.0001  # learning rate for the neural network.
BATCH_SIZE = 32
NUM_ACTION_CENTROIDS = 100  # number of KMeans centroids used to cluster the data.

# Adjust DATA_SAMPLES to fit your RAM, extra 100k samples is about 1.2 GB RAM.
# Example RAM usage and training time for training with different DATA_SAMPLES on a mid-range PC:
# (using the default parameters)
# |----------------------------------------------|
# | DATA_SAMPLES | RAM Usage, MB | Time, minutes |
# |------------------------------|---------------|
# |      100,000 |         3,854 |           1.9 |
# |      200,000 |         5,135 |           3.6 |
# |      500,000 |         8,741 |           8.1 |
# |    1,000,000 |        14,833 |          17.0 |
# |    1,528,808 |        21,411 |          28.1 | <- full MineRLObtainIronPickaxeVectorObf-v0 dataset
# |----------------------------------------------|
DATA_SAMPLES = 1000000

TRAIN_MODEL_NAME = 'research_potato.pth'  # name to use when saving the trained agent.
TEST_MODEL_NAME = 'research_potato.pth'  # name to use when loading the trained agent.
TRAIN_KMEANS_MODEL_NAME = 'centroids_for_research_potato.npy'  # name to use when saving the KMeans model.
TEST_KMEANS_MODEL_NAME = 'centroids_for_research_potato.npy'  # name to use when loading the KMeans model.

TEST_EPISODES = 10  # number of episodes to test the agent for.
MAX_TEST_EPISODE_LEN = 18000  # 18k is the default for MineRLObtainDiamondVectorObf.


class NatureCNN(nn.Module):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    Nicked from stable-baselines3:
        https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/torch_layers.py

    :param input_shape: A three-item tuple telling image dimensions in (C, H, W)
    :param output_dim: Dimensionality of the output vector
    """

    def __init__(self, input_shape, output_dim):
        super().__init__()
        n_input_channels = input_shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.zeros(1, *input_shape)).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


def train():
    # For demonstration purposes, we will only use ObtainPickaxe data which is smaller,
    # but has the similar steps as ObtainDiamond in the beginning.
    # "VectorObf" stands for vectorized (vector observation and action), where there is no
    # clear mapping between original actions and the vectors (i.e. you need to learn it)
    data = minerl.data.make("MineRLObtainIronPickaxeVectorObf-v0",  data_dir=DATA_DIR, num_workers=1)

    # First, use k-means to find actions that represent most of them.
    # This proved to be a strong approach in the MineRL 2020 competition.
    # See the following for more analysis:
    # https://github.com/GJuceviciute/MineRL-2020

    # Go over the dataset once and collect all actions and the observations (the "pov" image).
    # We do this to later on have uniform sampling of the dataset and to avoid high memory use spikes.
    all_actions = []
    all_pov_obs = []

    print("Loading data")
    trajectory_names = data.get_trajectory_names()
    random.shuffle(trajectory_names)

    # Add trajectories to the data until we reach the required DATA_SAMPLES.
    for trajectory_name in trajectory_names:
        trajectory = data.load_data(trajectory_name, skip_interval=0, include_metadata=False)
        for dataset_observation, dataset_action, _, _, _ in trajectory:
            all_actions.append(dataset_action["vector"])
            all_pov_obs.append(dataset_observation["pov"])
        if len(all_actions) >= DATA_SAMPLES:
            break

    all_actions = np.array(all_actions)
    all_pov_obs = np.array(all_pov_obs)

    # Run k-means clustering using scikit-learn.
    print("Running KMeans on the action vectors")
    kmeans = KMeans(n_clusters=NUM_ACTION_CENTROIDS)
    kmeans.fit(all_actions)
    action_centroids = kmeans.cluster_centers_
    print("KMeans done")

    # Now onto behavioural cloning itself.
    # Much like with intro track, we do behavioural cloning on the discrete actions,
    # where we turn the original vectors into discrete choices by mapping them to the closest
    # centroid (based on Euclidian distance).

    network = NatureCNN((3, 64, 64), NUM_ACTION_CENTROIDS).cuda()
    optimizer = th.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()

    num_samples = all_actions.shape[0]
    update_count = 0
    losses = []
    # We have the data loaded up already in all_actions and all_pov_obs arrays.
    # Let's do a manual training loop
    print("Training")
    for _ in range(EPOCHS):
        # Randomize the order in which we go over the samples
        epoch_indices = np.arange(num_samples)
        np.random.shuffle(epoch_indices)
        for batch_i in range(0, num_samples, BATCH_SIZE):
            # NOTE: this will cut off incomplete batches from end of the random indices
            batch_indices = epoch_indices[batch_i:batch_i + BATCH_SIZE]

            # Load the inputs and preprocess
            obs = all_pov_obs[batch_indices].astype(np.float32)
            # Transpose observations to be channel-first (BCHW instead of BHWC)
            obs = obs.transpose(0, 3, 1, 2)
            # Normalize observations. Do this here to avoid using too much memory (images are uint8 by default)
            obs /= 255.0

            # Map actions to their closest centroids
            action_vectors = all_actions[batch_indices]
            # Use numpy broadcasting to compute the distance between all
            # actions and centroids at once.
            # "None" in indexing adds a new dimension that allows the broadcasting
            distances = np.sum((action_vectors - action_centroids[:, None]) ** 2, axis=2)
            # Get the index of the closest centroid to each action.
            # This is an array of (batch_size,)
            actions = np.argmin(distances, axis=0)

            # Obtain logits of each action
            logits = network(th.from_numpy(obs).float().cuda())

            # Minimize cross-entropy with target labels.
            # We could also compute the probability of demonstration actions and
            # maximize them.
            loss = loss_function(logits, th.from_numpy(actions).long().cuda())

            # Standard PyTorch update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_count += 1
            losses.append(loss.item())
            if (update_count % 1000) == 0:
                mean_loss = sum(losses) / len(losses)
                tqdm.write("Iteration {}. Loss {:<10.3f}".format(update_count, mean_loss))
                losses.clear()
    print("Training done")

    # Save network and the centroids into separate files
    np.save(TRAIN_KMEANS_MODEL_NAME, action_centroids)
    th.save(network, TRAIN_MODEL_NAME)
    del data


def test():
    print("Running episodes")
    action_centroids = np.load(TEST_KMEANS_MODEL_NAME)
    network = th.load(TEST_MODEL_NAME).cuda()

    env = gym.make('MineRLObtainDiamondVectorObf-v0')

    num_actions = action_centroids.shape[0]
    action_list = np.arange(num_actions)

    for episode in range(TEST_EPISODES):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            # Process the action:
            #   - Add/remove batch dimensions
            #   - Transpose image (needs to be channels-last)
            #   - Normalize image
            obs = th.from_numpy(obs['pov'].transpose(2, 0, 1)[None].astype(np.float32) / 255).cuda()
            # Turn logits into probabilities
            probabilities = th.softmax(network(obs), dim=1)[0]
            # Into numpy
            probabilities = probabilities.detach().cpu().numpy()
            # Sample action according to the probabilities
            discrete_action = np.random.choice(action_list, p=probabilities)

            # Map the discrete action to the corresponding action centroid (vector)
            action = action_centroids[discrete_action]
            minerl_action = {"vector": action}

            obs, reward, done, info = env.step(minerl_action)
            total_reward += reward
            steps += 1
            if steps >= MAX_TEST_EPISODE_LEN:
                break

        print(f'Episode reward: {total_reward}, episode length: {steps}')

    env.close()


if __name__ == "__main__":
    # train()
    test()
