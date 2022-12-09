import random

from git import Repo

def check_with_user():
    repo = Repo('.')
    verify_num = random.randint(0,100)
    print('Any observations in Rewards were checked?')
    print('Everything committed?')
    print('Hash commit used in name?')
    print('Using the latest version of the experiment?')

    if repo.is_dirty():
        print('----------> Uncommitted files <-------------')
        return False

    user_input = input(f'Enter the value {verify_num}')

    if int(user_input) == verify_num:
        return True

    return False
