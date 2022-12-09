#/usr/bin/python

from git import Repo

repo = Repo('.')
if repo.is_dirty():
    print('----------> Uncommitted files <-------------')
else:
    remote = repo.remote('origin')
    remote.fetch()
    latest_remote_commit = remote.refs[repo.active_branch.name].commit
    latest_local_commit = repo.head.commit

    print(f"Latest commit is pushed: {latest_local_commit == latest_remote_commit}")
    commit_value = str(latest_local_commit)[:6]
    print()
    print()

    print("conda activate CarlaRlib")
    print(f"python ppo_train.py ppo_example/ppo_config.yaml --name ppo_{commit_value}")
    print(f"python dqn_train.py dqn/dqn_config.yaml --name dqn_{commit_value}")



