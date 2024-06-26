#/usr/bin/python
try:
    from git import Repo

    repo = Repo('.')
    if repo.is_dirty():
        print('----------> Uncommitted files <-------------')
    else:
        print("conda activate CarlaRlib")
        print(f"ulimit -c unlimited; python ppo_train.py ppo_example/ppo_config.yaml --name ppo")
        print(f"ulimit -c unlimited; python dqn_train.py dqn/dqn_config.yaml --name dqn")
        print(f"ulimit -c unlimited; python sac_train.py sac/sac_config.yaml --name sac")
        print("ulimit -c unlimited; python dqn_train.py dqn/dqn_config.yaml --name dqn --restore")

except:
    print("conda activate CarlaRlib")





