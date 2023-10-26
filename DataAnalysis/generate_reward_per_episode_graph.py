import pickle

import numpy as np
from matplotlib import pyplot as plt

def open_pickle(filename):
    with open(filename + '.pickle', 'rb') as handle:
        return pickle.load(handle)
def run(filenames=[],label_names = []):
    all_reward = []
    plt.figure(figsize=(10, 4))

    for filename in filenames:
        done_array = open_pickle(f'done/{filename}_done')
        rewards = open_pickle(f'rewards/{filename}_total_reward')
        window = 200
        average = []
        average_easy_y = []
        stopping_value = min(len(done_array),9000)
        for ind in range(stopping_value - window + 1):
            x = np.mean(rewards[ind:ind + window])
            average.append(x)

        all_reward.append(average)
    # for ind in range(len(all_episodes_easy_sum) - window + 1):
    #     average_easy_y.append(np.mean(all_episodes_easy_sum[ind:ind + window]))
    colours = ['b','g','r','c','m','y','k','w']
    for label,reward, colour in zip(label_names,all_reward,colours):
        plt.plot(reward,label=label,color= colour)
    # plt.plot(average_easy_y,label='average_easy_y ')

    # plt.plot(all_point_difficult_rewards,label='all_point_difficult_rewards')
    # plt.plot(all_line_difficult_rewards,label='all_line_difficult_rewards')
    plt.ylabel(f'Episode reward averaged over \n a {window} episode window')
    plt.xlabel('Number of episodes')
    # plt.xticks(np.arange(0, 8000 + 1, 500))
    plt.grid()
    plt.legend(loc='lower right')
    plt.show()

    window = 250
    average_difficult_y = []
    average_easy_y = []
    for ind in range(len(done_array) - window + 1):
        x = np.mean(done_array[ind:ind + window])
        average_difficult_y.append(x)
    # for ind in range(len(all_episodes_easy_sum) - window + 1):
    #     average_easy_y.append(np.mean(all_episodes_easy_sum[ind:ind + window]))
    plt.plot(average_difficult_y,label='average_DIFFICULT _y')
    # plt.plot(average_easy_y,label='average_easy_y ')

    # plt.plot(all_point_difficult_rewards,label='all_point_difficult_rewards')
    # plt.plot(all_line_difficult_rewards,label='all_line_difficult_rewards')
    plt.ylabel('Ratio of successful vs unsuccessful episodes \n averaged over a 250 episode window')
    plt.xlabel('Number of episodes')
    # plt.xticks(np.arange(0, 8000 + 1, 500))
    # plt.legend(loc='upper center')
    plt.show()

# Run for differnt weight vectors
run(['257465bf42a','169760e0cd9','1f96591647f','770d010caa3'], label_names= ['0.5/400','1.0/400','1.5/400','2.0/400'])
# run([], label_names= [])