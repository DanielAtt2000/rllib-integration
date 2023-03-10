import pandas as pd

results = pd.read_csv('526b0deb59b__home_daniel_ray_results_carla_rllib_good_dqn_66b0e183bf_truck_lidar_240x320_CustomDQNTrainer_CarlaEnv_5b56a_00000_0_2023-03-05_18-01-24_checkpoint_000260.csv')
routes = results['route'].unique()


results['collision_truck'] = results['collision_truck'].astype('int')
results['collision_trailer'] = results['collision_trailer'].astype('int')
results['timeout'] = results['timeout'].astype('int')
results['completed'] = results['completed'].astype('int')

final_results = pd.DataFrame(columns=['route','timesteps',"collision_truck","collision_trailer","timeout", "completed"])
for route in routes:
    indicies = results.loc[results['route'] == route]
    no_of_runs = len(indicies)
    x = indicies.sum()
    x['route'] = route
    x['timesteps'] = x['timesteps']/no_of_runs
    print(x)