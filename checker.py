import random
def check_with_user():
    verify_num = random.randint(0,100)
    print('Any observations in Rewards were checked')
    print('Everything committed')
    print('Hash commit used in name')
    user_input = input(f'Enter the value {verify_num}')

    if int(user_input) == verify_num:
        return True

    return False
