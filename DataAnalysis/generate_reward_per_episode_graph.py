import pickle

import numpy as np
from matplotlib import pyplot as plt

def open_pickle(filename):
    with open(filename + '.pickle', 'rb') as handle:
        return pickle.load(handle)
def run(filenames=[],label_names = [],window_size=[]):
    all_reward = []
    all_done = []

    plt.rcParams.update({'font.size': 13})

    max_value = 10500
    window_reward = 200
    window_done = 1500
    different_window_sizes = True

    if window_size == []:
        different_window_sizes = False
        for f in filenames:
            window_size.append(window_reward)
    plt.figure(figsize=(9, 5))

    for filename,window in zip(filenames,window_size):
        done_array = open_pickle(f'done/{filename}_done')
        rewards = open_pickle(f'rewards/{filename}_total_reward')

        average = []
        stopping_value = min(len(done_array),max_value)
        last_index_done = 0

        for ind in range(stopping_value,0+window-1,-1):
            x = np.mean(rewards[ind - window:ind])
            last_index_done = ind
            average.append(x)

        last_average = average[-1]
        for ind in range(last_index_done, 0,-1):
            x = np.mean(rewards[:ind])

            average.append(x)
        # for i in range(window):
        #     average.append(0)
        average.reverse()
        all_reward.append(average)
    # for ind in range(len(all_episodes_easy_sum) - window + 1):
    #     average_easy_y.append(np.mean(all_episodes_easy_sum[ind:ind + window]))
    colours = ['b','g','r','c','m','y','k','w']
    for label,reward, colour in zip(label_names,all_reward,colours):
        plt.plot(reward,label=label,color= colour)
    # plt.plot(average_easy_y,label='average_easy_y ')

    # plt.plot(all_point_difficult_rewards,label='all_point_difficult_rewards')
    # plt.plot(all_line_difficult_rewards,label='all_line_difficult_rewards')
    if not different_window_sizes:
        plt.ylabel(f'Episode reward averaged over \n a {window_reward} episode sliding window')
    else:
        plt.ylabel(f'Episode reward averaged over \n different episode sliding window sizes')

    plt.xlabel('Number of episodes')

    # plt.xticks(np.arange(0, 8000 + 1, 500))
    plt.grid()
    plt.legend(loc='lower right')
    plt.show()

    # plt.figure(figsize=(10, 4))
    # for filename in filenames:
    #     done_array = open_pickle(f'done/{filename}_done')
    #     average = []
    #     stopping_value = min(len(done_array),max_value)
    #     for ind in range(stopping_value - window_done + 1):
    #         x = np.mean(done_array[ind:ind + window_done])
    #         average.append(x)
    #
    #     all_done.append(average)
    # # for ind in range(len(all_episodes_easy_sum) - window + 1):
    # #     average_easy_y.append(np.mean(all_episodes_easy_sum[ind:ind + window]))
    #
    # colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    # for label, done, colour in zip(label_names, all_done, colours):
    #     plt.plot(done, label=label, color=colour)
    # # plt.plot(average_easy_y,label='average_easy_y ')
    #
    # # plt.plot(all_point_difficult_rewards,label='all_point_difficult_rewards')
    # # plt.plot(all_line_difficult_rewards,label='all_line_difficult_rewards')
    # plt.ylabel(f'Ratio of successful vs unsuccessful episodes \n averaged over a {window_done} episode window')
    # plt.xlabel('Number of episodes')
    # # plt.xticks(np.arange(0, 8000 + 1, 500))
    # # plt.legend(loc='upper center')
    # plt.grid()
    # plt.legend(loc='lower right')
    # plt.show()

# run(['770d010caa3','770d010caa3'], label_names= ['2/400 - 200 window','2/400 - 1500 window'], window_size=[200,1500])
# run(['2c5a756566f','2c5a756566f'], label_names= ['2/4 - 200 window','2/4 - 1500 window'], window_size=[200,1500])

# Run for differnt weight vectors
# run(['257465bf42a','169760e0cd9','1f96591647f','770d010caa3'], label_names= ['0.5/400','1.0/400','1.5/400','2.0/400'])

# Run for different state spaces
# run(['fe92e6d36e8','a1fc8624646','169760e0cd9'], label_names= ['Angles','Radii','Angles and Radii'])


# Constant vs Variable velocity action space
# run(['169760e0cd9','98bf3e6ed7a'], label_names= ['constant.velocity','variable.velocity'])
# run(['b1532c7504d','98bf3e6ed7a'], label_names= ['variable.velocity.no.smoothing','variable.velocity.smoothing'],window_size=[200,1500])
# PPO vs DQN vs SAC

run(['169760e0cd9','2f20aba770b','a3962295b7_1071ef25'], label_names= ['PPO','SAC','DQN'])


# run([], label_names= [])