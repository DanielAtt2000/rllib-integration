from ray.tune import Tuner
path = "/home/daniel/ray_results/carla_rllib/dqn_7c96af36cf"

tuner = Tuner.restore(
    path=path,
    resume_unfinished= True
)


results = tuner.fit()

import pprint

best_result = results.get_best_result()
print("Best hyperparameters found were: ", results.get_best_result().config)

df = results.get_dataframe()
print(df)


print("\nBest performing trial's final reported metrics:\n")

metrics_to_print = [
    "episode_reward_mean",
    "episode_reward_max",
    "episode_reward_min",
    "episode_len_mean",
]
pprint.pprint({k: v for k, v in best_result.metrics.items() if k in metrics_to_print})

file = open("results_dataframes/" + path.split('/')[-1] + '.md','w')
file.write(df.to_markdown())
file.close()
print(df.to_markdown())