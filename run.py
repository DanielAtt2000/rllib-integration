#/usr/bin/python

from git import Repo

repo = Repo('.')
if repo.is_dirty():
    print('----------> Uncommitted files <-------------')
else:
    print("conda activate CarlaRlib")
    print(f"python ppo_train.py ppo_example/ppo_config.yaml --name ppo")
    print(f"python dqn_train.py dqn/dqn_config.yaml --name dqn")



