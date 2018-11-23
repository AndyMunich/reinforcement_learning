# based on https://github.com/wwxFromTju/maddpg-tf
import tensorflow as tf
import numpy as np
import make_env
from agent_model_maddpg import MADDPG
from memory import Memory
from collections import deque

# hyperparams
gpu_fraction = 0.8
len_replay = 80000 
vis_freq = len_replay*0.1 # visualize every 10% of the length of the replay 
print_freq = len_replay*0.05 # visualize every 5% of the length of the replay

def create_init_update(oneline_name, target_name, tau=0.99):
    online_var = [i for i in tf.trainable_variables() if oneline_name in i.name]
    target_var = [i for i in tf.trainable_variables() if target_name in i.name]

    target_init = [tf.assign(target, online) for online, target in zip(online_var, target_var)]
    target_update = [tf.assign(target, (1 - tau) * online + tau * target) for online, target in zip(online_var, target_var)]

    return target_init, target_update


agent1_ddpg = MADDPG('agent1')
agent1_ddpg_target = MADDPG('agent1_target')
agent1_actor_target_init, agent1_actor_target_update = create_init_update('agent1_actor', 'agent1_target_actor')
agent1_critic_target_init, agent1_critic_target_update = create_init_update('agent1_critic', 'agent1_target_critic')

agent2_ddpg = MADDPG('agent2')
agent2_ddpg_target = MADDPG('agent2_target')
agent2_actor_target_init, agent2_actor_target_update = create_init_update('agent2_actor', 'agent2_target_actor')
agent2_critic_target_init, agent2_critic_target_update = create_init_update('agent2_critic', 'agent2_target_critic')

agent3_ddpg = MADDPG('agent3')
agent3_ddpg_target = MADDPG('agent3_target')
agent3_actor_target_init, agent3_actor_target_update = create_init_update('agent3_actor', 'agent3_target_actor')
agent3_critic_target_init, agent3_critic_target_update = create_init_update('agent3_critic', 'agent3_target_critic')


def get_agents_action(obs, sess, noise=0):
    agent1_action = agent1_ddpg.action(state=[obs[0]], sess=sess) + np.random.randn(2) * noise
    agent2_action = agent2_ddpg.action(state=[obs[1]], sess=sess) + np.random.randn(2) * noise
    agent3_action = agent3_ddpg.action(state=[obs[2]], sess=sess) + np.random.randn(2) * noise
    return agent1_action, agent2_action, agent3_action


def train_agent(agent_ddpg, agent_ddpg_target, agent_memory, agent_actor_target_update, agent_critic_target_update, sess, other_actors):
    total_obs_batch, total_act_batch, rew_batch, total_next_obs_batch, __mask = agent_memory.sample(32)

    act_batch = total_act_batch[:, 0, :]
    other_act_batch = np.hstack([total_act_batch[:, 1, :], total_act_batch[:, 2, :]])

    obs_batch = total_obs_batch[:, 0, :]

    next_obs_batch = total_next_obs_batch[:, 0, :]
    next_other_actor1_o = total_next_obs_batch[:, 1, :]
    next_other_actor2_o = total_next_obs_batch[:, 2, :]

    next_other_action = np.hstack([other_actors[0].action(next_other_actor1_o, sess), other_actors[1].action(next_other_actor2_o, sess)])
    target = rew_batch.reshape(-1, 1) + 0.9999 * agent_ddpg_target.Q(state=next_obs_batch, action=agent_ddpg.action(next_obs_batch, sess),
                                                                     other_action=next_other_action, sess=sess)
    agent_ddpg.train_actor(state=obs_batch, other_action=other_act_batch, sess=sess)
    agent_ddpg.train_critic(state=obs_batch, action=act_batch, other_action=other_act_batch, target=target, sess=sess)

    sess.run([agent_actor_target_update, agent_critic_target_update])

def vis_policy(env, sess):
    obs_t = env.reset()
    for i in range(2000):
        env.render()            
        agent1_action, agent2_action, agent3_action = get_agents_action(obs_t, sess, noise=0.2)

        a_t = [[0, i[0][0], 0, i[0][1], 0] for i in [agent1_action, agent2_action, agent3_action]]
        a_t.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])

        obs_next_t, reward_t, done, __t = env.step(a_t)
        obs_t = obs_next_t


if __name__ == '__main__':
    env = make_env.make_env('simple_tag')
    obs = env.reset()

    agent_reward_v = [tf.Variable(0, dtype=tf.float32) for i in range(3)]
    agent_reward_op = [tf.summary.scalar('agent' + str(i) + '_reward', agent_reward_v[i]) for i in range(3)]

    agent_a1 = [tf.Variable(0, dtype=tf.float32) for i in range(3)]
    agent_a1_op = [tf.summary.scalar('agent' + str(i) + '_action_1', agent_a1[i]) for i in range(3)]

    agent_a2 = [tf.Variable(0, dtype=tf.float32) for i in range(3)]
    agent_a2_op = [tf.summary.scalar('agent' + str(i) + '_action_2', agent_a2[i]) for i in range(3)]
    
    reward_100 = [tf.Variable(0, dtype=tf.float32) for i in range(3)]
    reward_100_op = [tf.summary.scalar('agent' + str(i) + '_reward_l100_mean', reward_100[i]) for i in range(3)]

    reward_1000 = [tf.Variable(0, dtype=tf.float32) for i in range(3)]
    reward_1000_op = [tf.summary.scalar('agent' + str(i) + '_reward_l1000_mean', reward_1000[i]) for i in range(3)]

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction

    sess = tf.Session(config=config)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run([agent1_actor_target_init, agent1_critic_target_init,
              agent2_actor_target_init, agent2_critic_target_init,
              agent3_actor_target_init, agent3_critic_target_init])

    agent1_memory = Memory(len_replay)
    agent2_memory = Memory(len_replay)
    agent3_memory = Memory(len_replay)
    
    reward_100_list = [[], [], []]
    ##########################################################################################
    ########## first half of the replay is used to only save samples for training ############
    # in the second half of "len_replay" the agents are trained and new samples are added  ###
    #### every 10000 steps the training is visualized to see how the policies improve ########
    ##########################################################################################
    for i in range(len_replay):
        # add samples to agend_memory
        if i % print_freq == 0:
            print('epoch=', i)
            
        agent1_action, agent2_action, agent3_action = get_agents_action(obs, sess, noise=0.2)

        a = [[0, i[0][0], 0, i[0][1], 0] for i in [agent1_action, agent2_action, agent3_action]]
        a.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])

        obs_next, reward, done, _ = env.step(a)

        for agent_index in range(3):
            reward_100_list[agent_index].append(reward[agent_index])
            reward_100_list[agent_index] = reward_100_list[agent_index][-1000:]

        agent1_memory.add(np.vstack([obs[0], obs[1], obs[2]]), np.vstack([agent1_action[0], agent2_action[0], agent3_action[0]]),
                          reward[0], np.vstack([obs_next[0], obs_next[1], obs_next[2]]), False)

        agent2_memory.add(np.vstack([obs[1], obs[2], obs[0]]), np.vstack([agent2_action[0], agent3_action[0], agent1_action[0]]),
                          reward[1], np.vstack([obs_next[1], obs_next[2], obs_next[0]]), False)

        agent3_memory.add(np.vstack([obs[2], obs[0], obs[1]]), np.vstack([agent3_action[0], agent1_action[0], agent2_action[0]]),
                          reward[2], np.vstack([obs_next[2], obs_next[0], obs_next[1]]), False)

        # train the agents and visualize periodically
        if i > len_replay/2:
            if(i % vis_freq == 0):
                vis_policy(env, sess)
                obs = env.reset()

            train_agent(agent1_ddpg, agent1_ddpg_target, agent1_memory, agent1_actor_target_update,
                        agent1_critic_target_update, sess, [agent2_ddpg_target, agent3_ddpg_target])

            train_agent(agent2_ddpg, agent2_ddpg_target, agent2_memory, agent2_actor_target_update,
                        agent2_critic_target_update, sess, [agent3_ddpg_target, agent1_ddpg_target])

            train_agent(agent3_ddpg, agent3_ddpg_target, agent3_memory, agent3_actor_target_update,
                        agent3_critic_target_update, sess, [agent1_ddpg_target, agent2_ddpg_target])

        obs = obs_next
        
    sess.close()
