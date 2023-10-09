def generate_route(points_per_exit, no_of_lanes=2 ):
    if no_of_lanes == 2:
        output = []
        entry_no = 1
        for entry in (points_per_exit):
            exit_no = 1
            for exit in (points_per_exit):
                no_of_exists = len(points_per_exit)

                if entry_no > exit_no:
                    value = no_of_exists-entry_no+exit_no
                else:
                    value = abs(exit_no-entry_no)

                if value == 0:
                    output.append([entry[2],[exit[1]],'left'])
                elif value == 1:
                    output.append([entry[3],[exit[0]],'right'])
                elif value == 2:
                    output.append([entry[2],[exit[1]],'left'])
                    output.append([entry[3],[exit[0]],'right'])
                elif value == 3:
                    output.append([entry[2], [exit[1]], 'left'])
                exit_no +=1
            entry_no +=1

        output.sort()
        print('[')
        for out in output:
            print(out,end='')
            print(',',end='')
            print()
        print(']')
    elif no_of_lanes == 1:
        output = []
        entry_no = 1
        for entry in (points_per_exit):
            exit_no = 1
            for exit in (points_per_exit):
                no_of_exists = len(points_per_exit)

                if entry_no > exit_no:
                    value = no_of_exists - entry_no + exit_no
                else:
                    value = abs(exit_no - entry_no)

                if value == 0:
                    output.append([entry[1], [exit[0]], 'oneLane'])
                elif value == 1:
                    output.append([entry[1], [exit[0]], 'oneLane'])
                elif value == 2:
                    output.append([entry[1], [exit[0]], 'oneLane'])
                elif value == 3:
                    output.append([entry[1], [exit[0]], 'oneLane'])
                exit_no += 1
            entry_no += 1

        output.sort()
        print('[')
        for out in output:
            print(out, end='')
            print(',', end='')
            print()
        print(']')
#             [140,[91],'oneLane'],



generate_route(no_of_lanes=2, points_per_exit=[[20,8,10,12],[22,19,21,29],[5,11,1,18]])
generate_route(no_of_lanes=1, points_per_exit=[[10,11],[12,13],[14,15],[9,8]])
