import random

from git import Repo

def check_with_user(check_commit):
    repo = Repo('.')
    verify_num = random.randint(0,100)
    print('=========================================')
    print(verify_num)
    print('Any observations in Rewards were checked?')
    print('Using the latest version of the experiment?')
    print('=========================================')

    if repo.is_dirty() and check_commit:
        print('----------> Uncommitted files <-------------')
        return False

    user_input = input(f'Enter the value: ')

    if int(user_input) == verify_num:
        return True

    return False

def commit_hash():
    repo = Repo('.')
    remote = repo.remote('origin')
    remote.fetch()
    latest_remote_commit = remote.refs[repo.active_branch.name].commit
    latest_local_commit = repo.head.commit

    assert latest_local_commit == latest_remote_commit
    commit_value = str(latest_local_commit)[:10]
    return commit_value