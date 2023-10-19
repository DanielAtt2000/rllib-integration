import os
import signal
import time

from Helper import read_txt_file


previous_killed_time = '2023-10-19 10:04:11.603536'
while True:
    lines = read_txt_file('failed_pids')
    killed = False
    for l, line in enumerate(lines):
        if previous_killed_time == -1:
            break
        if previous_killed_time in line:
            pid_to_kill = lines[l+1].split('----')
            pid_to_kill = pid_to_kill[1].replace(' ','').replace('\n','')
            previous_killed_time = lines[l+1].split('----')[0]
            print(f'Killed PID {pid_to_kill}')
            os.kill(int(pid_to_kill), signal.SIGKILL)
            killed = True
            break
    if not killed:
        pid_to_kill = lines[0].split('----')
        pid_to_kill = pid_to_kill[1].replace(' ', '').replace('\n','')
        previous_killed_time = lines[0].split('----')[0]
        print(f'Killed PID {pid_to_kill}')
        os.kill(int(pid_to_kill), signal.SIGKILL)

    time.sleep(60)
