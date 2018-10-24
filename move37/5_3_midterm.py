import os
import numpy as np
import gym
from joblib import parallel_backend, Parallel, delayed
import time

np.random.seed(1337)
env_name='BipedalWalker-v2'
env = gym.make(env_name)

# Hyperparameters
episode_length=2000
lr = 0.03 # learning rate / how much noise is applied for mutations
frac_mut = 0.1 # fraction of mutations
n_policy = 100 # number of policies for in generation
n_generations = 100

# used to save the visualization
def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

videos_dir = mkdir('.', 'videos')
monitor_dir = mkdir(videos_dir, env_name)

def gen_random_policy():
    ''' generates a random policy '''
    return np.random.randn(env.observation_space.shape[0], env.action_space.shape[0])
             
def crossover(policy1, policy2):
    ''' crossover for evolutionary algorithm '''
    new_policy = policy1.copy()
    for i in range(env.observation_space.shape[0]):
        for j in range(env.action_space.shape[0]):
            rand = np.random.uniform()
            if rand > 0.5:
                new_policy[i, j] = policy2[i, j]
    return new_policy

def mutation(policy, p=frac_mut):
    ''' mutation for evolutionary algorithm '''
    new_policy = policy.copy()
    for i in range(env.observation_space.shape[0]):
        for j in range(env.action_space.shape[0]):
            rand = np.random.uniform()
            if rand < p:
                new_policy[i, j] = new_policy[i, j] + lr*np.random.randn()
    return new_policy

def state_to_action(sensor_input, policy):
    ''' maps from state space (sensor values) to action space (motor torques) by a linear function '''
    sensor_input = np.atleast_2d(sensor_input)
    outp = np.dot(sensor_input, policy)[0]
    return outp

def evaluate_policy( policy ):
    ''' runs one epoch with a given policy to evaluate the reward '''
    state = env.reset()
    done = False
    sum_rewards = 0.0
    num_plays = 0.0
    while not done and num_plays < episode_length:
        action = state_to_action(state, policy)
        state, reward, done, _ = env.step(action)
        sum_rewards += reward
        num_plays += 1
    sum_rewards += 300 # get positive rewards to get rid of sign errors
    return sum_rewards

rec_vid = False # dont record die video during training since it slows down the process
should_record = lambda i: rec_vid
env = gym.wrappers.Monitor(env, monitor_dir, video_callable=should_record, force=True)
policy_pop = [gen_random_policy() for _ in range(n_policy)] # start with a random population of policies
a = time.time()

for i in range(n_generations):
    policy_scores = Parallel(n_jobs=4)( delayed(evaluate_policy)(p) for p in policy_pop )        
    print('Generation %d : max score = %d' %(i+1,  max(policy_scores) - 300.0))
    policy_ranks = list(reversed(np.argsort(policy_scores)))
    elite_set = [policy_pop[x] for x in policy_ranks[:5]]
    select_probs = np.array(policy_scores) / np.sum(policy_scores)
    child_set = [crossover(
        policy_pop[np.random.choice(range(n_policy), p=select_probs)],
        policy_pop[np.random.choice(range(n_policy), p=select_probs)])
        for _ in range(n_policy - 5) ]
    mutated_list = [mutation(p) for p in child_set] 
    policy_pop = elite_set
    policy_pop += mutated_list
policy_score = [evaluate_policy(p) for p in policy_pop]
optimal_policy = policy_pop[np.argmax(policy_score)]

b = time.time()
print('used time: ', b-a)

# show the visualization
rec_vid = True
evaluate_policy(optimal_policy)
rec_vid = False

env.env.close()
