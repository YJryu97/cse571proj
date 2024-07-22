import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from replay_buffer import ReplayBuffer
from velodyne_env import GazeboEnv

import os
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Train Q value
def evaluate(network, epoch, eval_episodes=10):
    avg_reward = 0.0
    col = 0
    for _ in range(eval_episodes):
        count = 0
        state = env.reset()
        done = False
        while not done and count < 501:
            action = network.get_action(np.array(state))
            a_in = [(action[0] + 1) / 2, action[1]]
            state, reward, done, _ = env.step(a_in)
            avg_reward += reward
            count += 1
            if reward < -90:
                col += 1
            print("Current avg_reward:", avg_reward/count)
    avg_reward /= eval_episodes
    avg_col = col / eval_episodes
    print("..............................................")
    print(
        "Average Reward over %i Evaluation Episodes, Epoch %i: %f, %f"
        % (eval_episodes, epoch, avg_reward, avg_col)
    )
    print("..............................................")
    return avg_reward

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 800)
        self.fc2 = nn.Linear(800, 600)
        self.fc3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()
    
    def forward(self, state):
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))
        action = self.tanh(self.fc3(state))
        return action
    
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()
        # input_v = state_dim + action_dim
        # self.action_dim = action_dim
        # num_outputs = action_dim.shape[0]
        # Layer 1
        self.fc1 = nn.Linear(state_dim, 800)
        self.ln1 = nn.LayerNorm(800)
        # Action Layer 2
        self.fc2_a = nn.Linear(action_dim, 600)
        self.ln2_a = nn.LayerNorm(600)
        # State Layer 2
        self.fc2 = nn.Linear(800, 600)
        self.ln2 = nn.LayerNorm(600)
        self.fc3 = nn.Linear(600, 1)
        

    def forward(self, state, action):
        #print("State shape:", state.shape)
        #print("Action shape:", action.shape)
        #s = torch.cat([state, action], 1)
        # print("Concatenated shape:", s.shape)
        # Layer 1
        s1 = self.fc1(state)
        # s1 = self.ln1(s1)
        s1 = F.relu(s1)
        self.fc2(s1)
        self.fc2_a(action)
        # s = F.relu(self.fc1(state))
        # Layer 2
        s11 = torch.mm(s1, self.fc2.weight.data.t()) # Matrix multiplication
        # s11 = self.ln2(s11)
        s12 = torch.mm(action, self.fc2_a.weight.data.t())
        # s12 = self.ln2_a(s12)
        s = F.relu(s11 + s12 + self.fc2_a.bias.data)
        q = self.fc3(s)
        # Output
        #q = self.fc3(s)
        
        return q
        # s2 = F.relu(self.fc4(state))
        # self.fc5_s_a(s2)
        # self.fc5_a(action)
        # s21 = torch.mm(s2, self.fc5_s_a.weight.data.t()) # Matrix multiplication
        #s22 = torch.mm(action, self.fc5_a.weight.data.t())
        # s1 = F.relu(s21 + s22 + self.fc5_a.bias.data)
        # q2 = self.fc6(s2)
        
    
class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        # Initialize actor network
        self.actor = ActorNetwork(state_dim, action_dim).to(device)
        self.actor_target = ActorNetwork(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        
        # Initialize Critic Network
        self.critic = CriticNetwork(state_dim, action_dim).to(device)
        self.critic_target = CriticNetwork(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action
        self.writer = SummaryWriter()
        self.iter_count = 0
    
    def get_action(self, state):
        # Function to get the action from the actor
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    # Training
    def train(self,
              replay_buffer, # where to get samples for training
              iterations, 
              batch_size=100, # how many samples to sample at each iteration
              discount=1, # discount factor in BE
              tau = 0.005, 
              policy_noise = 0.2,
              noise_clip = 0.5,
              policy_freq=2,
    ):
        av_Q = 0
        max_Q = -inf
        av_loss = 0
        for it in range(iterations):
            # Sample a batch of transitions from the replay buffer
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
            ) = replay_buffer.sample_batch(batch_size)
            
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)
            
            # Obtain the estimated action from the next state by using the actor-target
            next_action = self.actor_target(next_state)
            
            # Add noise to the action
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
            
            #print(next_state.)
            target_Q = self.critic_target(next_state, next_action)
            av_Q += torch.mean(target_Q)
            # Calculate the final Q value from the target network parameters by using Bellman equation
            target_Q = reward + ((1 - done) * discount * target_Q).detach()
            max_Q = max(max_Q, torch.max(target_Q))
            
            # Get the Q values of the basis networks with the current parameters
            current_Q = self.critic(state, action)
            
            # Calculate the loss between the current Q value and the target Q value
            loss = F.mse_loss(current_Q, target_Q) 
            
            # Update policy by one step of gradient descent (critic update)
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step() # Perfome single optimization step
            
            # Update Actor 
            if it % policy_freq == 0:
            # Maximize the actor output value by performing gradient descent on negative Q values
            # (essentially perform gradient ascent)
                policy_grad = self.critic(state, self.actor(state))
                policy_grad = -policy_grad.mean()
                self.actor_optimizer.zero_grad()
                policy_grad.backward()
                self.actor_optimizer.step()
                
            # Use soft update to update the actor-target network parameters by
            # infusing small amount of current parameters  
            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
                ):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            # Use soft update to update the critic-target network parameters by
            # infusing small amount of current parameters  
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            av_loss += loss
        self.iter_count += 1
        
        # Write new values for tensorboard
        self.writer.add_scalar("loss", av_loss / iterations, self.iter_count)
        self.writer.add_scalar("Av. Q", av_Q / iterations, self.iter_count)
        self.writer.add_scalar("Max. Q", max_Q, self.iter_count)
    
    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))            
    
    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )
        self.critic.load_state_dict(
            torch.load("%s%s_critic.pth" % (directory, filename))
        )   
    
# set the parameters for the implementation
seed = 0 # random seed number
eval_freq = 5e3 # After how many steps to perform the evaluation
max_ep = 500 # maximum number of steps per episode
eval_ep = 10 # number of episodes for evaluation 
max_timesteps = 5e5 # Maximum number of steps to perform
expl_noise = 1 # Initial exploration noise starting value in range [expl_min, ... 1]
expl_decay_steps = (500000) # Number of steps over which the initial exploration noise will decay over
expl_min = 0.1  # Exploration noise after the decay in range [0...expl_noise]
batch_size = 40  # Size of the mini-batch
discount = 0.9999  # Discount factor to calculate the discounted future reward (should be close to 1)         
tau = 0.005  # Soft target update variable (should be close to 0)        
policy_noise = 0.2  # Added noise for exploration
noise_clip = 0.5  # Maximum clamping values of the noise
policy_freq = 3  # Frequency of Actor network updates
buffer_size = 1e6  # Maximum size of the buffer
file_name = "DDPG_velodyne"  # name of the file to store the policy
save_model = True  # Wether to save the model or not
load_model = False  # Wether to load a stored model
random_near_obstacle = True  # To take random actions near obstacles or not
        
# Create the network storage folders
if not os.path.exists("./results3"):
    os.makedirs("./results3")
if save_model and not os.path.exists("./pytorch_models3"):
    os.makedirs("./pytorch_models3")

# Create the training environment
environment_dim = 20
robot_dim = 4 # polar coordinate features + previous action features (2).
env = GazeboEnv("robot_scenario.launch", environment_dim)
time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = environment_dim + robot_dim
action_dim = 2
max_action = 1

# The network
network = DDPG(state_dim, action_dim, max_action)
# Create a replay buffer
replay_buffer = ReplayBuffer(buffer_size, seed)
if load_model:
    try:
        network.load(file_name, "./pytorch_models3")
    except:
        print("Could not load the stored model parameters, initializing training with random parameters")

# Create evaluation Data store
evaluations = [] 

timestep = 0
timesteps_since_eval = 0
episode_num = 0
done = True
epoch = 1

count_rand_actions = 0
random_action = []

# Begin 
while timestep < max_timesteps:
    
    # On termination of episode
    if done:
        if timestep != 0:
            network.train(
                replay_buffer, # where to get samples for training
                episode_timesteps, 
                batch_size, # how many samples to sample at each iteration
                discount, # discount factor in BE
                tau, 
                policy_noise,
                noise_clip,
                policy_freq,
            )
        
        if timesteps_since_eval >= eval_freq:
            print("Validating")
            timesteps_since_eval %= eval_freq
            evaluations.append(
                evaluate(network=network, epoch=epoch, eval_episodes = eval_ep)
            )
            network.save(file_name, directory="./pytorch_models3")
            np.save("./results3/%s" % (file_name), evaluations)
            epoch += 1
            
        state = env.reset()
        done = False
        
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1
    
    # add some exploration noise 
    # Greedy strategy but also exploitation
    if expl_noise > expl_min:
        expl_noise = expl_noise - ((1 - expl_min) / expl_decay_steps)
        
    action = network.get_action(np.array(state))
    action - (action + np.random.normal(0, expl_noise, size=action_dim)).clip(-max_action, 
                                                                              max_action)
    # If the robot is facing an obstacle, randomly force it to take a consistent random action.
    # This is done to increase exploration in situations near obstacles.
    # Training can also be performed without it
    if random_near_obstacle:
        if (
            np.random.uniform(0, 1) > 0.85
            and min(state[4:-8]) < 0.6
            and count_rand_actions < 1
        ):
            count_rand_actions = np.random.randint(8, 15)
            random_action = np.random.uniform(-1, 1, 2)

        if count_rand_actions > 0:
            count_rand_actions -= 1
            action = random_action
            action[0] = -1

    # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
    #a_in = [(action[0] + 1) / 2, action[1]]
    a_in = [(action[0] + 1)/2 , action[1]]
    next_state, reward, done, target = env.step(a_in)
    done_bool = 0 if episode_timesteps + 1 == max_ep else int(done)
    done = 1 if episode_timesteps + 1 == max_ep else int(done)
    episode_reward += reward

    # Save the tuple in replay buffer
    replay_buffer.add(state, action, reward, done_bool, next_state)

    # Update the counters
    state = next_state
    episode_timesteps += 1
    timestep += 1
    timesteps_since_eval += 1

# After the training is done, evaluate the network and save it
evaluations.append(evaluate(network=network, epoch=epoch, eval_episodes=eval_ep))
if save_model:
    network.save("%s" % file_name, directory="./models")
np.save("./results3/%s" % file_name, evaluations)
