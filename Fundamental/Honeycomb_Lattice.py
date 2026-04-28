import numpy as np

def honeycomb_points(nl):
    def num_sites_higherlevel(nl):
        return(((nl-2)*2*6)+(3*6))
    numbers_perlevel = np.array([])
    if nl >= 1:
        numbers_perlevel = np.append(numbers_perlevel,6)
    if nl >= 2:
        numbers_perlevel = np.append(numbers_perlevel, 18)
    if nl >= 3:
        for n in range(3,nl+1):
            numbers_perlevel = np.append(numbers_perlevel, num_sites_higherlevel(n))
    total_numpoints = np.sum(numbers_perlevel)
    return(numbers_perlevel,total_numpoints)

def honeycomb_lattice(nl):

    points_perlevel, totalnum_points = honeycomb_points(nl)
    point_cumulative = np.array([], dtype=int)
    for a in range(len(points_perlevel)):
        if a == 0:
            point_cumulative = np.append(point_cumulative, int(points_perlevel[0]))
        elif a == len(points_perlevel) - 1:
            point_cumulative = np.append(point_cumulative, int(totalnum_points))
        else:
            point_cumulative = np.append(point_cumulative, int(points_perlevel[a] + point_cumulative[-1]))
    print(point_cumulative)

    H0 = np.zeros((int(totalnum_points),int(totalnum_points)),dtype=int)
    t=1

    if nl >= 1:
        countern1 = 0
        for n in range(point_cumulative[0]):
            if n == 0:
                H0[n][n+1] = t
                H0[n+1][n] = t
                H0[n][point_cumulative[0]-1] = t
                H0[point_cumulative[0]-1][n] = t
                H0[n][point_cumulative[0]] = t
                H0[point_cumulative[0]][n] = t
                countern1 += 1
            elif n == point_cumulative[0]-1:
                H0[n][0] = t
                H0[0][n] = t
                H0[n][n - 1] = t
                H0[n - 1][n] = t
                H0[n][point_cumulative[0] + 3 * countern1] = t
                H0[point_cumulative[0] + 3 * countern1][n] = t
            else:
                H0[n][n+1] = t
                H0[n+1][n] = t
                H0[n][n-1] = t
                H0[n-1][n] = t
                H0[n][point_cumulative[0]+3*countern1] = t
                H0[point_cumulative[0]+3*countern1][n] = t
                countern1 += 1

    if nl >= 2:
        countern2 = 3
        twocountern2 = 0
        threecountern2 = 0
        for n in range(point_cumulative[0], point_cumulative[1]):
            if n == point_cumulative[0]:
                H0[n][n+1] = t
                H0[n+1][n] = t
                H0[n][point_cumulative[1]-1] = t
                H0[point_cumulative[1]-1][n] = t
            elif n == point_cumulative[0]+1:
                H0[n][n + 1] = t
                H0[n + 1][n] = t
                H0[n][n - 1] = t
                H0[n - 1][n] = t
                H0[n][point_cumulative[1] + 1] = t
                H0[point_cumulative[1] + 1][n] = t
            elif n == point_cumulative[1]-1:
                threecountern2 += 1
                H0[n][n-1] = t
                H0[n-1][n] = t
                H0[n][point_cumulative[1] + 1 + 3 * threecountern2 + 2 * twocountern2] = t
                H0[point_cumulative[1] + 1 + 3 * threecountern2 + 2 * twocountern2][n] = t
            elif countern2 == 3:
                threecountern2 += 1
                H0[n][n + 1] = t
                H0[n + 1][n] = t
                H0[n][n - 1] = t
                H0[n - 1][n] = t
                H0[n][point_cumulative[1] + 1 + 3*threecountern2 + 2*twocountern2] = t
                H0[point_cumulative[1] + 1 + 3*threecountern2 + 2*twocountern2][n] = t
                countern2 = 1
            elif countern2 == 1:
                twocountern2 += 1
                H0[n][n + 1] = t
                H0[n + 1][n] = t
                H0[n][n - 1] = t
                H0[n - 1][n] = t
                countern2 += 1
            elif countern2 == 2:
                H0[n][n + 1] = t
                H0[n + 1][n] = t
                H0[n][n - 1] = t
                H0[n - 1][n] = t
                H0[n][point_cumulative[1] + 1 + 3 * threecountern2 + 2 * twocountern2] = t
                H0[point_cumulative[1] + 1 + 3 * threecountern2 + 2 * twocountern2][n] = t
                countern2 += 1

    if nl >= 3:
        for nm in range(3,nl+1):
            if nm == nl:
                # pbc_add_factor = 0.5*(points_perlevel[-1])
                for n in range(point_cumulative[nm - 2], point_cumulative[nm - 1]):
                    if n == point_cumulative[nm-2]:
                        H0[n][n+1] = t
                        H0[n+1][n] = t
                        H0[n][point_cumulative[nm-1]-1] = t
                        H0[point_cumulative[nm-1]-1][n] = t
                        # if n < point_cumulative[nm-2] + 0.5*points_perlevel[-1]:
                        #     H0[n][int(n+pbc_add_factor)] = t
                        #     H0[int(n+pbc_add_factor)][n] = t
                    elif n == point_cumulative[nm-1]-1:
                        H0[n][n - 1] = t
                        H0[n - 1][n] = t
                        H0[n][point_cumulative[nm - 2]] = t
                        H0[point_cumulative[nm - 2]][n] = t
                    else:
                        H0[n][n + 1] = t
                        H0[n + 1][n] = t
                        H0[n][n - 1] = t
                        H0[n - 1][n] = t
                        # if n < point_cumulative[nm-2] + 0.5*points_perlevel[-1]:
                        #     H0[n][int(n+pbc_add_factor)] = t
                        #     H0[int(n+pbc_add_factor)][n] = t
            elif nm % 2 == 1:
                twocounter = 0
                threecounter = 0
                straight_region_polys = nm-2
                straight_region_count_lim = 2*straight_region_polys + 3
                straight_count = (straight_region_count_lim+1)/2
                on_straight_counter = 0
                straight_region = True
                turn_region = False
                counter = 0
                for n in range(point_cumulative[nm - 2], point_cumulative[nm - 1]):
                    if n == point_cumulative[nm - 2]:
                        H0[n][n+1] = t
                        H0[n + 1][n] = t
                        H0[n][point_cumulative[nm-1]-1] = t
                        H0[point_cumulative[nm-1]-1][n] = t
                        H0[n][point_cumulative[nm-1]] = t
                        H0[point_cumulative[nm-1]][n] = t
                        straight_count += 1
                    elif n == point_cumulative[nm - 1]-1:
                        H0[n][n-1] = t
                        H0[n-1][n] = t
                    elif straight_region == True:
                        if on_straight_counter == 0:
                            twocounter += 1
                            H0[n][n+1] = t
                            H0[n+1][n] = t
                            H0[n][n-1] = t
                            H0[n-1][n] = t
                            on_straight_counter = 1
                            straight_count += 1
                        elif on_straight_counter == 1:
                            H0[n][n + 1] = t
                            H0[n + 1][n] = t
                            H0[n][n - 1] = t
                            H0[n - 1][n] = t
                            H0[n][point_cumulative[nm-1] + 3*threecounter + 2*twocounter] = t
                            H0[point_cumulative[nm-1] + 3*threecounter + 2*twocounter][n] = t
                            on_straight_counter = 0
                            straight_count += 1
                        if straight_count == straight_region_count_lim+1:
                            straight_region = False
                            turn_region = True
                            straight_count = 2
                            on_straight_counter = 0
                    elif turn_region == True:
                        turn_region = False
                        straight_region = True
                        threecounter += 1
                        H0[n][n + 1] = t
                        H0[n + 1][n] = t
                        H0[n][n - 1] = t
                        H0[n - 1][n] = t
                        H0[n][point_cumulative[nm - 1] + 3 * threecounter + 2 * twocounter] = t
                        H0[point_cumulative[nm - 1] + 3 * threecounter + 2 * twocounter][n] = t
            elif nm % 2 == 0:
                twocounter = 0
                threecounter = 0
                straight_region_polys = nm - 2
                straight_region_count_lim = 2 * straight_region_polys + 3
                straight_count = (straight_region_count_lim + 1) / 2
                on_straight_counter = 1
                straight_region = True
                turn_region = False
                counter = 0
                for n in range(point_cumulative[nm - 2], point_cumulative[nm - 1]):
                    if n == point_cumulative[nm - 2]:
                        H0[n][n + 1] = t
                        H0[n + 1][n] = t
                        H0[n][point_cumulative[nm - 1] - 1] = t
                        H0[point_cumulative[nm - 1] - 1][n] = t
                        straight_count += 1
                    elif n == point_cumulative[nm - 1]-1:
                        H0[n][n-1] = t
                        H0[n-1][n] = t
                        H0[n][point_cumulative[nm - 1] + 1 + 3 * threecounter + 2 * twocounter] = t
                        H0[point_cumulative[nm - 1] + 1 + 3 * threecounter + 2 * twocounter][n] = t
                    elif straight_region == True:
                        if on_straight_counter == 0:
                            twocounter += 1
                            H0[n][n + 1] = t
                            H0[n + 1][n] = t
                            H0[n][n - 1] = t
                            H0[n - 1][n] = t
                            on_straight_counter = 1
                            straight_count += 1
                        elif on_straight_counter == 1:
                            H0[n][n + 1] = t
                            H0[n + 1][n] = t
                            H0[n][n - 1] = t
                            H0[n - 1][n] = t
                            H0[n][point_cumulative[nm - 1] + 1 + 3 * threecounter + 2 * twocounter] = t
                            H0[point_cumulative[nm - 1] + 1 + 3 * threecounter + 2 * twocounter][n] = t
                            on_straight_counter = 0
                            straight_count += 1
                        if straight_count == straight_region_count_lim+1:
                            straight_region = False
                            turn_region = True
                            straight_count = 2
                            on_straight_counter = 0
                    elif turn_region == True:
                        turn_region = False
                        straight_region = True
                        threecounter += 1
                        H0[n][n + 1] = t
                        H0[n + 1][n] = t
                        H0[n][n - 1] = t
                        H0[n - 1][n] = t
                        H0[n][point_cumulative[nm - 1] + 1 + 3 * threecounter + 2 * twocounter] = t
                        H0[point_cumulative[nm - 1] + 1 + 3 * threecounter + 2 * twocounter][n] = t
    return(H0)

def honeycomb_PeierlsSubstitution(nl,alpha):

    points_perlevel, totalnum_points = honeycomb_points(nl)
    point_cumulative = np.array([], dtype=int)
    for a in range(len(points_perlevel)):
        if a == 0:
            point_cumulative = np.append(point_cumulative, int(points_perlevel[0]))
        elif a == len(points_perlevel) - 1:
            point_cumulative = np.append(point_cumulative, int(totalnum_points))
        else:
            point_cumulative = np.append(point_cumulative, int(points_perlevel[a] + point_cumulative[-1]))
    print(point_cumulative)

    H0 = np.zeros((int(totalnum_points),int(totalnum_points)),dtype=np.complex_)
    t=1

    if nl >= 1:
        x1 = alpha/6
        t1 = t*np.exp(1j*x1)
        countern1 = 0
        for n in range(point_cumulative[0]):
            if n == 0:
                H0[n][n+1] = np.conj(t1)
                H0[n+1][n] = t1
                H0[n][point_cumulative[0]-1] = t1
                H0[point_cumulative[0]-1][n] = np.conj(t1)
                H0[n][point_cumulative[0]] = t
                H0[point_cumulative[0]][n] = t
                countern1 += 1
            elif n == point_cumulative[0]-1:
                H0[n][0] = np.conj(t1)
                H0[0][n] = t1
                H0[n][n - 1] = t1
                H0[n - 1][n] = np.conj(t1)
                H0[n][point_cumulative[0] + 3 * countern1] = t
                H0[point_cumulative[0] + 3 * countern1][n] = t
            else:
                H0[n][n+1] = np.conj(t1)
                H0[n+1][n] = t1
                H0[n][n-1] = t1
                H0[n-1][n] = np.conj(t1)
                H0[n][point_cumulative[0]+3*countern1] = t
                H0[point_cumulative[0]+3*countern1][n] = t
                countern1 += 1

    if nl >= 2:
        x2 = (alpha+x1)/3
        t2 = t*np.exp(1j*x2)
        countern2 = 3
        twocountern2 = 0
        threecountern2 = 0
        for n in range(point_cumulative[0], point_cumulative[1]):
            if n == point_cumulative[0]:
                H0[n][n+1] = np.conj(t2)
                H0[n+1][n] = t2
                H0[n][point_cumulative[1]-1] = t2
                H0[point_cumulative[1]-1][n] = np.conj(t2)
            elif n == point_cumulative[0]+1:
                H0[n][n + 1] = np.conj(t2)
                H0[n + 1][n] = t2
                H0[n][n - 1] = t2
                H0[n - 1][n] = np.conj(t2)
                H0[n][point_cumulative[1] + 1] = t
                H0[point_cumulative[1] + 1][n] = t
            elif n == point_cumulative[1]-1:
                threecountern2 += 1
                H0[n][n-1] = t2
                H0[n-1][n] = np.conj(t2)
                H0[n][point_cumulative[1] + 1 + 3 * threecountern2 + 2 * twocountern2] = t
                H0[point_cumulative[1] + 1 + 3 * threecountern2 + 2 * twocountern2][n] = t
            elif countern2 == 3:
                threecountern2 += 1
                H0[n][n + 1] = np.conj(t2)
                H0[n + 1][n] = t2
                H0[n][n - 1] = t2
                H0[n - 1][n] = np.conj(t2)
                H0[n][point_cumulative[1] + 1 + 3*threecountern2 + 2*twocountern2] = t
                H0[point_cumulative[1] + 1 + 3*threecountern2 + 2*twocountern2][n] = t
                countern2 = 1
            elif countern2 == 1:
                twocountern2 += 1
                H0[n][n + 1] = np.conj(t2)
                H0[n + 1][n] = t2
                H0[n][n - 1] = t2
                H0[n - 1][n] = np.conj(t2)
                countern2 += 1
            elif countern2 == 2:
                H0[n][n + 1] = np.conj(t2)
                H0[n + 1][n] = t2
                H0[n][n - 1] = t2
                H0[n - 1][n] = np.conj(t2)
                H0[n][point_cumulative[1] + 1 + 3 * threecountern2 + 2 * twocountern2] = t
                H0[point_cumulative[1] + 1 + 3 * threecountern2 + 2 * twocountern2][n] = t
                countern2 += 1

    if nl >= 3:
        prev_phase = np.array([x2])
        for nm in range(3,nl+1):
            if nm % 2 == 1:
                num_new_phases = len(np.unique(prev_phase))+1
                new_counter = 0
                new_x_list = np.array([])
                new_t_list = np.array([])
                for ip in range(num_new_phases):
                    if ip == 0:
                        new_x = (alpha+prev_phase[0])/3
                        new_x_list = np.append(new_x_list, new_x)
                        new_t_list = np.append(new_t_list, t*np.exp(1j*new_x))
                    elif ip == num_new_phases-1:
                        new_x = (alpha+2*prev_phase[-1])/2
                        new_x_list = np.append(new_x_list,new_x)
                        new_t_list = np.append(new_t_list,t*np.exp(1j*new_x))
                    else:
                        new_x = (alpha+prev_phase[new_counter]+prev_phase[new_counter+1])/2
                        new_counter += 1
                        new_x_list = np.append(new_x_list, new_x)
                        new_t_list = np.append(new_t_list, t * np.exp(1j * new_x))

                prev_phase = new_x_list
                new_x_list = np.append(new_x_list, np.flip(new_x_list)[1:])
                new_t_list = np.repeat(np.append(new_t_list, np.flip(new_t_list)[1:]),2)

                phasecounter = int((len(new_t_list)+2)/2)-1
                twocounter = 0
                threecounter = 0
                straight_region_polys = nm-2
                straight_region_count_lim = 2*straight_region_polys + 3
                straight_count = (straight_region_count_lim+1)/2
                on_straight_counter = 0
                straight_region = True
                turn_region = False
                counter = 0
                for n in range(point_cumulative[nm - 2], point_cumulative[nm - 1]):
                    if n == point_cumulative[nm - 2]:
                        H0[n][n+1] = np.conj(new_t_list[phasecounter])
                        H0[n + 1][n] = new_t_list[phasecounter]
                        H0[n][point_cumulative[nm-1]-1] = new_t_list[phasecounter-1]
                        H0[point_cumulative[nm-1]-1][n] = np.conj(new_t_list[phasecounter-1])
                        if nm != nl:
                            H0[n][point_cumulative[nm-1]] = t
                            H0[point_cumulative[nm-1]][n] = t
                        straight_count += 1
                        phasecounter += 1
                    elif n == point_cumulative[nm - 1]-1:
                        H0[n][n-1] = new_t_list[phasecounter-1]
                        H0[n-1][n] = np.conj(new_t_list[phasecounter-1])
                    elif straight_region == True:
                        if on_straight_counter == 0:
                            twocounter += 1
                            H0[n][n+1] = np.conj(new_t_list[phasecounter])
                            H0[n+1][n] = new_t_list[phasecounter]
                            H0[n][n-1] = new_t_list[phasecounter-1]
                            H0[n-1][n] = np.conj(new_t_list[phasecounter-1])
                            on_straight_counter = 1
                            straight_count += 1
                            phasecounter += 1
                        elif on_straight_counter == 1:
                            H0[n][n + 1] = np.conj(new_t_list[phasecounter])
                            H0[n + 1][n] = new_t_list[phasecounter]
                            H0[n][n - 1] = new_t_list[phasecounter - 1]
                            H0[n - 1][n] = np.conj(new_t_list[phasecounter - 1])
                            if nm != nl:
                                H0[n][point_cumulative[nm-1] + 3*threecounter + 2*twocounter] = t
                                H0[point_cumulative[nm-1] + 3*threecounter + 2*twocounter][n] = t
                            on_straight_counter = 0
                            straight_count += 1
                            phasecounter += 1
                        if straight_count == straight_region_count_lim+1:
                            straight_region = False
                            turn_region = True
                            straight_count = 2
                            on_straight_counter = 0
                            phasecounter = 1
                    elif turn_region == True:
                        turn_region = False
                        straight_region = True
                        threecounter += 1
                        H0[n][n+1] = np.conj(new_t_list[phasecounter])
                        H0[n+1][n] = new_t_list[phasecounter]
                        H0[n][n-1] = new_t_list[phasecounter-1]
                        H0[n-1][n] = np.conj(new_t_list[phasecounter-1])
                        if nm != nl:
                            H0[n][point_cumulative[nm - 1] + 3 * threecounter + 2 * twocounter] = t
                            H0[point_cumulative[nm - 1] + 3 * threecounter + 2 * twocounter][n] = t
                        phasecounter += 1
            elif nm % 2 == 0:

                num_new_phases = len(np.unique(prev_phase))
                new_counter = 0
                new_x_list = np.array([])
                new_t_list = np.array([])
                for ip in range(num_new_phases):
                    if ip == 0:
                        new_x = (alpha + prev_phase[0]) / 3
                        new_x_list = np.append(new_x_list, new_x)
                        new_t_list = np.append(new_t_list, t * np.exp(1j * new_x))
                    elif ip == num_new_phases - 1:
                        new_x = (alpha + prev_phase[-1] + prev_phase[-2]) / 2
                        new_x_list = np.append(new_x_list, new_x)
                        new_t_list = np.append(new_t_list, t * np.exp(1j * new_x))
                    else:
                        new_x = (alpha + prev_phase[new_counter] + prev_phase[new_counter + 1]) / 2
                        new_counter += 1
                        new_x_list = np.append(new_x_list, new_x)
                        new_t_list = np.append(new_t_list, t * np.exp(1j * new_x))

                prev_phase = new_x_list
                new_x_list = np.append(new_x_list, np.flip(new_x_list))
                new_t_list = np.repeat(np.append(new_t_list, np.flip(new_t_list)), 2)

                phasecounter = int((len(new_t_list) + 2) / 2) - 1
                twocounter = 0
                threecounter = 0
                straight_region_polys = nm - 2
                straight_region_count_lim = 2 * straight_region_polys + 3
                straight_count = (straight_region_count_lim + 1) / 2
                on_straight_counter = 1
                straight_region = True
                turn_region = False
                counter = 0
                for n in range(point_cumulative[nm - 2], point_cumulative[nm - 1]):
                    if n == point_cumulative[nm - 2]:
                        H0[n][n + 1] = np.conj(new_t_list[phasecounter])
                        H0[n + 1][n] = new_t_list[phasecounter]
                        H0[n][point_cumulative[nm - 1] - 1] = new_t_list[phasecounter-1]
                        H0[point_cumulative[nm - 1] - 1][n] = np.conj(new_t_list[phasecounter-1])
                        straight_count += 1
                        phasecounter += 1
                    elif n == point_cumulative[nm - 1]-1:
                        H0[n][n-1] = new_t_list[phasecounter-1]
                        H0[n-1][n] = np.conj(new_t_list[phasecounter-1])
                        if nm != nl:
                            H0[n][point_cumulative[nm - 1] + 1 + 3 * threecounter + 2 * twocounter] = t
                            H0[point_cumulative[nm - 1] + 1 + 3 * threecounter + 2 * twocounter][n] = t
                    elif straight_region == True:
                        if on_straight_counter == 0:
                            twocounter += 1
                            H0[n][n + 1] = np.conj(new_t_list[phasecounter])
                            H0[n + 1][n] = new_t_list[phasecounter]
                            H0[n][n - 1] = new_t_list[phasecounter-1]
                            H0[n - 1][n] = np.conj(new_t_list[phasecounter-1])
                            on_straight_counter = 1
                            straight_count += 1
                            phasecounter += 1
                        elif on_straight_counter == 1:
                            H0[n][n + 1] = np.conj(new_t_list[phasecounter])
                            H0[n + 1][n] = new_t_list[phasecounter]
                            H0[n][n - 1] = new_t_list[phasecounter - 1]
                            H0[n - 1][n] = np.conj(new_t_list[phasecounter - 1])
                            if nm != nl:
                                H0[n][point_cumulative[nm - 1] + 1 + 3 * threecounter + 2 * twocounter] = t
                                H0[point_cumulative[nm - 1] + 1 + 3 * threecounter + 2 * twocounter][n] = t
                            on_straight_counter = 0
                            straight_count += 1
                            phasecounter += 1
                        if straight_count == straight_region_count_lim+1:
                            straight_region = False
                            turn_region = True
                            straight_count = 2
                            on_straight_counter = 0
                            phasecounter = 1
                    elif turn_region == True:
                        turn_region = False
                        straight_region = True
                        threecounter += 1
                        H0[n][n + 1] = np.conj(new_t_list[phasecounter])
                        H0[n + 1][n] = new_t_list[phasecounter]
                        H0[n][n - 1] = new_t_list[phasecounter - 1]
                        H0[n - 1][n] = np.conj(new_t_list[phasecounter - 1])
                        if nm != nl:
                            H0[n][point_cumulative[nm - 1] + 1 + 3 * threecounter + 2 * twocounter] = t
                            H0[point_cumulative[nm - 1] + 1 + 3 * threecounter + 2 * twocounter][n] = t
                        phasecounter += 1
    H0 = np.conj(H0) #This is to make it consistent with ccw; originally coded to adhere to cw
    return(H0)

def add_periodic_boundary(nl, ham):
    t = 1
    points_perlevel, totalnum_points = honeycomb_points(nl)
    point_cumulative = np.array([], dtype=int)
    for a in range(len(points_perlevel)):
        if a == 0:
            point_cumulative = np.append(point_cumulative, int(points_perlevel[0]))
        elif a == len(points_perlevel) - 1:
            point_cumulative = np.append(point_cumulative, int(totalnum_points))
        else:
            point_cumulative = np.append(point_cumulative, int(points_perlevel[a] + point_cumulative[-1]))
    print(point_cumulative)

    straight_region_polys = nl - 2
    straight_region_count_lim = 2 * straight_region_polys + 3
    straight_count = (straight_region_count_lim + 1) / 2

    straight_region_centers = [n for n in range(point_cumulative[nl - 2], point_cumulative[nl - 1], int(straight_region_count_lim))]
    half_straight_count = 0.5*(straight_region_count_lim-1)

    pbc_add_factor = points_perlevel[-1]/2

    if nl % 2 == 1:
        for c in straight_region_centers[:3]:
            if c == straight_region_centers[0]:
                for n in range(int(c - half_straight_count), int(c + half_straight_count + 1)):
                    if n == c:
                        ham[n][int(n + pbc_add_factor)] = t
                        ham[int(n + pbc_add_factor)][n] = t
                    elif np.abs(n - c) % 2 == 0 and n-c < 0:
                        off_center = n - c
                        sites = np.linspace(0,np.size(ham,0)-1,np.size(ham,0))
                        ham[int(sites[int(off_center)])][int(c + pbc_add_factor - off_center)] = t
                        ham[int(c + pbc_add_factor - off_center)][int(sites[int(off_center)])] = t
                    elif np.abs(n - c) % 2 == 0 and n-c > 0:
                        off_center = n - c
                        ham[n][int(c + pbc_add_factor - off_center)] = t
                        ham[int(c + pbc_add_factor - off_center)][n] = t
            else:
                for n in range(int(c-half_straight_count), int(c+half_straight_count+1)):
                    if n == c:
                        ham[n][int(n+pbc_add_factor)] = t
                        ham[int(n+pbc_add_factor)][n] = t
                    elif np.abs(n-c) % 2 == 0:
                        off_center = n-c
                        print(n,off_center,int(n+pbc_add_factor-off_center))
                        ham[n][int(c+pbc_add_factor-off_center)] = t
                        ham[int(c+pbc_add_factor-off_center)][n] = t
    elif nl % 2 == 0:
        for c in straight_region_centers[:3]:
            if c == straight_region_centers[0]:
                for n in range(int(c - half_straight_count), int(c + half_straight_count + 1)):
                    if np.abs(n - c) % 2 == 1 and n - c < 0:
                        off_center = n - c
                        sites = np.linspace(0, np.size(ham, 0)-1, np.size(ham, 0))
                        ham[int(sites[int(off_center)])][int(c + pbc_add_factor - off_center)] = t
                        ham[int(c + pbc_add_factor - off_center)][int(sites[int(off_center)])] = t
                    elif np.abs(n - c) % 2 == 1 and n - c > 0:
                        off_center = n - c
                        ham[n][int(c + pbc_add_factor - off_center)] = t
                        ham[int(c + pbc_add_factor - off_center)][n] = t
            else:
                for n in range(int(c - half_straight_count), int(c + half_straight_count + 1)):
                    if np.abs(n - c) % 2 == 1:
                        off_center = n - c
                        print(n, off_center, int(n + pbc_add_factor - off_center))
                        ham[n][int(c + pbc_add_factor - off_center)] = t
                        ham[int(c + pbc_add_factor - off_center)][n] = t
    return(ham)

def honeycomb_lattice_periodic_boundary(nl):

    points_perlevel, totalnum_points = honeycomb_points(nl)
    point_cumulative = np.array([], dtype=int)
    for a in range(len(points_perlevel)):
        if a == 0:
            point_cumulative = np.append(point_cumulative, int(points_perlevel[0]))
        elif a == len(points_perlevel) - 1:
            point_cumulative = np.append(point_cumulative, int(totalnum_points))
        else:
            point_cumulative = np.append(point_cumulative, int(points_perlevel[a] + point_cumulative[-1]))
    print(point_cumulative)

    H0 = np.zeros((int(totalnum_points),int(totalnum_points)),dtype=int)
    t=1

    if nl >= 1:
        countern1 = 0
        for n in range(point_cumulative[0]):
            if n == 0:
                H0[n][n+1] = t
                H0[n+1][n] = t
                H0[n][point_cumulative[0]-1] = t
                H0[point_cumulative[0]-1][n] = t
                H0[n][point_cumulative[0]] = t
                H0[point_cumulative[0]][n] = t
                countern1 += 1
            elif n == point_cumulative[0]-1:
                H0[n][0] = t
                H0[0][n] = t
                H0[n][n - 1] = t
                H0[n - 1][n] = t
                H0[n][point_cumulative[0] + 3 * countern1] = t
                H0[point_cumulative[0] + 3 * countern1][n] = t
            else:
                H0[n][n+1] = t
                H0[n+1][n] = t
                H0[n][n-1] = t
                H0[n-1][n] = t
                H0[n][point_cumulative[0]+3*countern1] = t
                H0[point_cumulative[0]+3*countern1][n] = t
                countern1 += 1

    if nl >= 2:
        countern2 = 3
        twocountern2 = 0
        threecountern2 = 0
        for n in range(point_cumulative[0], point_cumulative[1]):
            if n == point_cumulative[0]:
                H0[n][n+1] = t
                H0[n+1][n] = t
                H0[n][point_cumulative[1]-1] = t
                H0[point_cumulative[1]-1][n] = t
            elif n == point_cumulative[0]+1:
                H0[n][n + 1] = t
                H0[n + 1][n] = t
                H0[n][n - 1] = t
                H0[n - 1][n] = t
                H0[n][point_cumulative[1] + 1] = t
                H0[point_cumulative[1] + 1][n] = t
            elif n == point_cumulative[1]-1:
                threecountern2 += 1
                H0[n][n-1] = t
                H0[n-1][n] = t
                H0[n][point_cumulative[1] + 1 + 3 * threecountern2 + 2 * twocountern2] = t
                H0[point_cumulative[1] + 1 + 3 * threecountern2 + 2 * twocountern2][n] = t
            elif countern2 == 3:
                threecountern2 += 1
                H0[n][n + 1] = t
                H0[n + 1][n] = t
                H0[n][n - 1] = t
                H0[n - 1][n] = t
                H0[n][point_cumulative[1] + 1 + 3*threecountern2 + 2*twocountern2] = t
                H0[point_cumulative[1] + 1 + 3*threecountern2 + 2*twocountern2][n] = t
                countern2 = 1
            elif countern2 == 1:
                twocountern2 += 1
                H0[n][n + 1] = t
                H0[n + 1][n] = t
                H0[n][n - 1] = t
                H0[n - 1][n] = t
                countern2 += 1
            elif countern2 == 2:
                H0[n][n + 1] = t
                H0[n + 1][n] = t
                H0[n][n - 1] = t
                H0[n - 1][n] = t
                H0[n][point_cumulative[1] + 1 + 3 * threecountern2 + 2 * twocountern2] = t
                H0[point_cumulative[1] + 1 + 3 * threecountern2 + 2 * twocountern2][n] = t
                countern2 += 1

    if nl >= 3:
        for nm in range(3,nl+1):
            if nm == nl:
                # pbc_add_factor = 0.5*(points_perlevel[-1])
                for n in range(point_cumulative[nm - 2], point_cumulative[nm - 1]):
                    if n == point_cumulative[nm-2]:
                        H0[n][n+1] = t
                        H0[n+1][n] = t
                        H0[n][point_cumulative[nm-1]-1] = t
                        H0[point_cumulative[nm-1]-1][n] = t
                        # if n < point_cumulative[nm-2] + 0.5*points_perlevel[-1]:
                        #     H0[n][int(n+pbc_add_factor)] = t
                        #     H0[int(n+pbc_add_factor)][n] = t
                    elif n == point_cumulative[nm-1]-1:
                        H0[n][n - 1] = t
                        H0[n - 1][n] = t
                        H0[n][point_cumulative[nm - 2]] = t
                        H0[point_cumulative[nm - 2]][n] = t
                    else:
                        H0[n][n + 1] = t
                        H0[n + 1][n] = t
                        H0[n][n - 1] = t
                        H0[n - 1][n] = t
                        # if n < point_cumulative[nm-2] + 0.5*points_perlevel[-1]:
                        #     H0[n][int(n+pbc_add_factor)] = t
                        #     H0[int(n+pbc_add_factor)][n] = t
            elif nm % 2 == 1:
                twocounter = 0
                threecounter = 0
                straight_region_polys = nm-2
                straight_region_count_lim = 2*straight_region_polys + 3
                straight_count = (straight_region_count_lim+1)/2
                on_straight_counter = 0
                straight_region = True
                turn_region = False
                counter = 0
                for n in range(point_cumulative[nm - 2], point_cumulative[nm - 1]):
                    if n == point_cumulative[nm - 2]:
                        H0[n][n+1] = t
                        H0[n + 1][n] = t
                        H0[n][point_cumulative[nm-1]-1] = t
                        H0[point_cumulative[nm-1]-1][n] = t
                        H0[n][point_cumulative[nm-1]] = t
                        H0[point_cumulative[nm-1]][n] = t
                        straight_count += 1
                    elif n == point_cumulative[nm - 1]-1:
                        H0[n][n-1] = t
                        H0[n-1][n] = t
                    elif straight_region == True:
                        if on_straight_counter == 0:
                            twocounter += 1
                            H0[n][n+1] = t
                            H0[n+1][n] = t
                            H0[n][n-1] = t
                            H0[n-1][n] = t
                            on_straight_counter = 1
                            straight_count += 1
                        elif on_straight_counter == 1:
                            H0[n][n + 1] = t
                            H0[n + 1][n] = t
                            H0[n][n - 1] = t
                            H0[n - 1][n] = t
                            H0[n][point_cumulative[nm-1] + 3*threecounter + 2*twocounter] = t
                            H0[point_cumulative[nm-1] + 3*threecounter + 2*twocounter][n] = t
                            on_straight_counter = 0
                            straight_count += 1
                        if straight_count == straight_region_count_lim+1:
                            straight_region = False
                            turn_region = True
                            straight_count = 2
                            on_straight_counter = 0
                    elif turn_region == True:
                        turn_region = False
                        straight_region = True
                        threecounter += 1
                        H0[n][n + 1] = t
                        H0[n + 1][n] = t
                        H0[n][n - 1] = t
                        H0[n - 1][n] = t
                        H0[n][point_cumulative[nm - 1] + 3 * threecounter + 2 * twocounter] = t
                        H0[point_cumulative[nm - 1] + 3 * threecounter + 2 * twocounter][n] = t
            elif nm % 2 == 0:
                twocounter = 0
                threecounter = 0
                straight_region_polys = nm - 2
                straight_region_count_lim = 2 * straight_region_polys + 3
                straight_count = (straight_region_count_lim + 1) / 2
                on_straight_counter = 1
                straight_region = True
                turn_region = False
                counter = 0
                for n in range(point_cumulative[nm - 2], point_cumulative[nm - 1]):
                    if n == point_cumulative[nm - 2]:
                        H0[n][n + 1] = t
                        H0[n + 1][n] = t
                        H0[n][point_cumulative[nm - 1] - 1] = t
                        H0[point_cumulative[nm - 1] - 1][n] = t
                        straight_count += 1
                    elif n == point_cumulative[nm - 1]-1:
                        H0[n][n-1] = t
                        H0[n-1][n] = t
                        H0[n][point_cumulative[nm - 1] + 1 + 3 * threecounter + 2 * twocounter] = t
                        H0[point_cumulative[nm - 1] + 1 + 3 * threecounter + 2 * twocounter][n] = t
                    elif straight_region == True:
                        if on_straight_counter == 0:
                            twocounter += 1
                            H0[n][n + 1] = t
                            H0[n + 1][n] = t
                            H0[n][n - 1] = t
                            H0[n - 1][n] = t
                            on_straight_counter = 1
                            straight_count += 1
                        elif on_straight_counter == 1:
                            H0[n][n + 1] = t
                            H0[n + 1][n] = t
                            H0[n][n - 1] = t
                            H0[n - 1][n] = t
                            H0[n][point_cumulative[nm - 1] + 1 + 3 * threecounter + 2 * twocounter] = t
                            H0[point_cumulative[nm - 1] + 1 + 3 * threecounter + 2 * twocounter][n] = t
                            on_straight_counter = 0
                            straight_count += 1
                        if straight_count == straight_region_count_lim+1:
                            straight_region = False
                            turn_region = True
                            straight_count = 2
                            on_straight_counter = 0
                    elif turn_region == True:
                        turn_region = False
                        straight_region = True
                        threecounter += 1
                        H0[n][n + 1] = t
                        H0[n + 1][n] = t
                        H0[n][n - 1] = t
                        H0[n - 1][n] = t
                        H0[n][point_cumulative[nm - 1] + 1 + 3 * threecounter + 2 * twocounter] = t
                        H0[point_cumulative[nm - 1] + 1 + 3 * threecounter + 2 * twocounter][n] = t
    H0 = add_periodic_boundary(nl,H0)
    return(H0)

def honeycomb_number_plaquets(nl):
    numberplaquets = 0
    for n in range(nl):
        if n+1 == 1:
            numberplaquets += 1
        else:
            numberplaquets += (n+1)*6 - 6
    return(numberplaquets)

'''
def honeycomb_points(n):
    number_perlevel = np.array([])
    if n >= 1:
        number_perlevel = np.append(number_perlevel, 6)
    if n >= 2:
        number_perlevel = np.append(number_perlevel, 18)
    if n >= 3:
        number_perlevel = np.append(number_perlevel, 30)
    if n>= 4:
        number_perlevel = np.append(number_perlevel, 42)
    total_numpoints = np.sum(number_perlevel)
    return(number_perlevel, total_numpoints)

def honeycomb_lattice(n):
    points_perlevel, totalnum_points = honeycomb_points(n)
    point_cumulative = np.array([], dtype=int)
    for a in range(len(points_perlevel)):
        if a == 0:
            point_cumulative = np.append(point_cumulative, int(points_perlevel[0]))
        elif a == len(points_perlevel) - 1:
            point_cumulative = np.append(point_cumulative, int(totalnum_points))
        else:
            point_cumulative = np.append(point_cumulative, int(points_perlevel[a] + point_cumulative[-1]))
    print(point_cumulative)
    H0 = np.zeros((int(totalnum_points), int(totalnum_points)), dtype=int)
    if n >= 1:
        threecounter = 0
        for i in range(0, point_cumulative[0]):
            if i == 0:
                H0[i][i + 1] = 1
                H0[i + 1][i] = 1
                H0[i][point_cumulative[0]] = 1
                H0[point_cumulative[0]][i] = 1
                H0[i][point_cumulative[0] + 3 * threecounter] = 1
                H0[point_cumulative[0] + 3 * threecounter][i] = 1
                threecounter += 1
            elif i == point_cumulative[0]-1:
                H0[i][0] = 1
                H0[0][i] = 1
                H0[i][point_cumulative[0] + 3 * threecounter] = 1
                H0[point_cumulative[0] + 3 * threecounter][i] = 1
            else:
                H0[i][i+1] = 1
                H0[i+1][i] = 1
                H0[i][point_cumulative[0]+3*threecounter] = 1
                H0[point_cumulative[0]+3*threecounter][i] = 1
                threecounter += 1
    if n >= 2:
        threecounter = -1
        twocounter = 0
        counter = 0
        for i in range(point_cumulative[0], point_cumulative[1]):
            if i == point_cumulative[1]-1:
                H0[i][point_cumulative[0]] = 1
                H0[point_cumulative[0]][i] = 1
                H0[i][point_cumulative[1] + 3 * threecounter + 2 * twocounter] = 1
                H0[point_cumulative[1] + 3 * threecounter + 2 * twocounter][i] = 1
            elif i == point_cumulative[0]:
                H0[i][i+1] = 1
                H0[i+1][i] = 1
                threecounter += 1
                counter += 1
            elif counter == 3:
                H0[i][i + 1] = 1
                H0[i + 1][i] = 1
                threecounter -= 1
                twocounter += 1
                counter = 1
            else:
                H0[i][i + 1] = 1
                H0[i + 1][i] = 1
                H0[i][point_cumulative[1] + 3 * threecounter + 2 * twocounter] = 1
                H0[point_cumulative[1] + 3 * threecounter + 2 * twocounter][i] = 1
                threecounter += 1
                counter += 1
    if n >= 3:
        threecounter = -1
        twocounter = 0
        counter = 0
        for i in range(point_cumulative[1],point_cumulative[2]):
            if i == point_cumulative[2]-1:
                H0[i][point_cumulative[1]] = 1
                H0[point_cumulative[1]][i] = 1
                print(i+1, point_cumulative[2] + 3 * threecounter + 2 * twocounter + 1)
                H0[i][point_cumulative[2] + 3 * threecounter + 2 * twocounter] = 1
                H0[point_cumulative[2] + 3 * threecounter + 2 * twocounter][i] = 1
            elif i == point_cumulative[1]:
                H0[i][i+1] = 1
                H0[i+1][i] = 1
                threecounter += 1
                counter += 1
            elif counter == 1:
                H0[i][i+1] = 1
                H0[i+1][i] = 1
                print(i+1, point_cumulative[2] + 3 * threecounter + 2 * twocounter + 1, counter)
                H0[i][point_cumulative[2] + 3 * threecounter + 2 * twocounter] = 1
                H0[point_cumulative[2] + 3 * threecounter + 2 * twocounter][i] = 1
                threecounter += 1
                counter += 1
            elif counter == 2:
                H0[i][i + 1] = 1
                H0[i + 1][i] = 1
                print(i+1, point_cumulative[2] + 3 * threecounter + 2 * twocounter + 1, counter)
                H0[i][point_cumulative[2] + 3 * threecounter + 2 * twocounter] = 1
                H0[point_cumulative[2] + 3 * threecounter + 2 * twocounter][i] = 1
                counter += 1
            elif counter == 3:
                H0[i][i + 1] = 1
                H0[i + 1][i] = 1
                twocounter += 1
                counter += 1
            elif counter == 4:
                H0[i][i + 1] = 1
                H0[i + 1][i] = 1
                print(i+1, point_cumulative[2] + 3 * threecounter + 2 * twocounter + 1, counter)
                H0[i][point_cumulative[2] + 3 * threecounter + 2 * twocounter] = 1
                H0[point_cumulative[2] + 3 * threecounter + 2 * twocounter][i] = 1
                counter += 1
            elif counter == 5:
                H0[i][i + 1] = 1
                H0[i + 1][i] = 1
                twocounter += 1
                counter = 1

    if n == 4:
        for i in range(point_cumulative[2], point_cumulative[3]):
            if i == point_cumulative[3]-1:
                H0[i][point_cumulative[2]] = 1
                H0[point_cumulative[2]][i] = 1
            else:
                H0[i][i+1] = 1
                H0[i+1][i] = 1
    return(H0)
'''

def pulling_axis_points(first_point_on_level, tot_number_points_on_level):
    lower_bound_point = first_point_on_level + (tot_number_points_on_level / 6) #point on pulling axis on side parallel to pulling axis
    upper_bound_point = first_point_on_level + (2 * tot_number_points_on_level / 3) #point on pulling axis on side anti-parallel to pulling axis
    return (lower_bound_point, upper_bound_point)

def nearest_neighbor(site_index, ham):
    all_nn = np.where(ham[int(site_index),:]!=0)[0] #Gets nearest-neighbor sites
    return(all_nn)

def bilayer_honeycomb_fermi_liquid_H0(nl, t_interlayer):

    points_per_level, tot_num_points = honeycomb_points(nl)
    tot_num_points = int(tot_num_points)

    point_cumulative = np.array([], dtype=int)
    for a in range(len(points_per_level)):
        if a == 0:
            point_cumulative = np.append(point_cumulative, int(points_per_level[0]))
        elif a == len(points_per_level) - 1:
            point_cumulative = np.append(point_cumulative, int(tot_num_points))
        else:
            point_cumulative = np.append(point_cumulative, int(points_per_level[a] + point_cumulative[-1]))
    print(point_cumulative)
    point_cumulative = np.concatenate((np.array([0]),point_cumulative)) #So now point_cumulative is site number of first site in nth generation (apart from last entry which is just tot_num_sites)

    single_layer_ham = honeycomb_lattice(nl)
    multilayer_ham = np.block([[single_layer_ham, np.zeros((tot_num_points, tot_num_points))],[np.zeros((tot_num_points, tot_num_points)), single_layer_ham]])

    #For level 1 specifically
    multilayer_ham[0, tot_num_points + 5] = t_interlayer
    multilayer_ham[tot_num_points + 5, 0] = np.conj(t_interlayer)
    multilayer_ham[2, tot_num_points + 3] = t_interlayer
    multilayer_ham[tot_num_points + 3, 2] = np.conj(t_interlayer)
    multilayer_ham[4, tot_num_points + 18] = t_interlayer
    multilayer_ham[tot_num_points + 18, 4] = np.conj(t_interlayer)

    #For odd numbered levels
    for level in np.arange(3,nl,2):

        first_paxis_pt, second_paxis_pt = pulling_axis_points(point_cumulative[level-1], points_per_level[level-1])
        first_paxis_pt = int(first_paxis_pt)
        second_paxis_pt = int(second_paxis_pt)
        #Only second_paxis_pt is connected to other layer in odd numbered levels
        multilayer_ham[second_paxis_pt, int(tot_num_points + np.max(nearest_neighbor(second_paxis_pt, single_layer_ham)))] = t_interlayer
        multilayer_ham[int(tot_num_points + np.max(nearest_neighbor(second_paxis_pt, single_layer_ham))), second_paxis_pt] = np.conj(t_interlayer)

        #Connection to other layer for first site in level; in odd numbered levels first site on level is connected to last site on level
        multilayer_ham[int(point_cumulative[level-1]), int(tot_num_points + point_cumulative[level]-1)] = t_interlayer
        multilayer_ham[int(tot_num_points + point_cumulative[level] - 1), int(point_cumulative[level - 1])] = np.conj(t_interlayer)

        #Iterate over part of level between first site and first_paxis_pt
        for pnt in np.arange(point_cumulative[level-1]+2, first_paxis_pt, 2):
            pnt = int(pnt)
            #Check if on side that is not parallel to pulling axis
            if pnt - point_cumulative[level-1] <= (((points_per_level[level-1]/6)-1)/2):
                multilayer_ham[pnt, int(tot_num_points + pnt - 1)] = t_interlayer
                multilayer_ham[int(tot_num_points + pnt - 1), pnt] = np.conj(t_interlayer)
            #If is on side parallel to pulling axis
            else:
                multilayer_ham[pnt, int(tot_num_points + np.min(nearest_neighbor(pnt, single_layer_ham)))] = t_interlayer
                multilayer_ham[int(tot_num_points + np.min(nearest_neighbor(pnt, single_layer_ham))), pnt] = np.conj(t_interlayer)

        #Iterate over part of level between first_paxis_pt and second_paxis_pt
        for pnt in np.arange(first_paxis_pt+1, second_paxis_pt, 2):
            pnt = int(pnt)
            #Check if on side parallel to pulling axis
            if pnt - first_paxis_pt <= (((points_per_level[level-1]/6)-1)/2):
                multilayer_ham[pnt, int(tot_num_points + np.min(nearest_neighbor(pnt, single_layer_ham)))] = t_interlayer
                multilayer_ham[int(tot_num_points + np.min(nearest_neighbor(pnt, single_layer_ham))), pnt] = np.conj(t_interlayer)
            #Check if on side anti-parallel to pulling axis
            elif second_paxis_pt - pnt <= (((points_per_level[level-1]/6)-1)/2):
                multilayer_ham[pnt, int(tot_num_points + np.max(nearest_neighbor(pnt, single_layer_ham)))] = t_interlayer
                multilayer_ham[int(tot_num_points + np.max(nearest_neighbor(pnt, single_layer_ham))), pnt] = np.conj(t_interlayer)
            #If on side not parallel to pulling axis
            else:
                multilayer_ham[pnt, int(tot_num_points + pnt + 1)] = t_interlayer
                multilayer_ham[int(tot_num_points + pnt + 1), pnt] = np.conj(t_interlayer)

        #Iterate over part of level between second_axis_pt and last point on level
        for pnt in np.arange(second_paxis_pt+2, point_cumulative[level], 2):
            pnt = int(pnt)
            #Check if on side anti-parallel to pulling axis
            if pnt - second_paxis_pt <= (((points_per_level[level-1]/6)-1)/2):
                multilayer_ham[pnt, int(tot_num_points + np.max(nearest_neighbor(pnt, single_layer_ham)))] = t_interlayer
                multilayer_ham[int(tot_num_points + np.max(nearest_neighbor(pnt, single_layer_ham))), pnt] = np.conj(t_interlayer)
            #If on side not parallel or anti-parallel to pulling axis
            else:
                multilayer_ham[pnt, int(tot_num_points + pnt - 1)] = t_interlayer
                multilayer_ham[int(tot_num_points + pnt - 1), pnt] = np.conj(t_interlayer)

    #For even numbered levels
    for level in np.arange(2, nl, 2):

        first_paxis_pt, second_paxis_pt = pulling_axis_points(point_cumulative[level-1], points_per_level[level-1])
        first_paxis_pt = int(first_paxis_pt)
        second_paxis_pt = int(second_paxis_pt)
        #Only first_paxis_pt connects to other layer for even numbered levels
        multilayer_ham[first_paxis_pt, int(tot_num_points + np.min(nearest_neighbor(first_paxis_pt, single_layer_ham)))] = t_interlayer
        multilayer_ham[int(tot_num_points + np.min(nearest_neighbor(first_paxis_pt, single_layer_ham))), first_paxis_pt] = np.conj(t_interlayer)

        #Iterate over part of level between first point and first_paxis_pt
        for pnt in np.arange(point_cumulative[level-1]+1, first_paxis_pt, 2):
            pnt = int(pnt)
            #Check if on side not parallel or anti-parallel to pulling axis
            if pnt - point_cumulative[level-1] <= (((points_per_level[level-1]/6)-1)/2):
                multilayer_ham[pnt, int(tot_num_points + pnt - 1)] = t_interlayer
                multilayer_ham[int(tot_num_points + pnt - 1), pnt] = np.conj(t_interlayer)
            #If on side parallel to pulling axis
            else:
                multilayer_ham[pnt, int(tot_num_points + np.min(nearest_neighbor(pnt, single_layer_ham)))] = t_interlayer
                multilayer_ham[int(tot_num_points + np.min(nearest_neighbor(pnt, single_layer_ham))), pnt] = np.conj(t_interlayer)

        #Iterate over points between first_paxis_pt and second_paxis_pt
        for pnt in np.arange(first_paxis_pt+2, second_paxis_pt, 2):
            pnt = int(pnt)
            #Check if on side parallel to pulling axis
            if pnt - first_paxis_pt <= (((points_per_level[level-1]/6)-1)/2):
                multilayer_ham[pnt, int(tot_num_points + np.min(nearest_neighbor(pnt, single_layer_ham)))] = t_interlayer
                multilayer_ham[int(tot_num_points + np.min(nearest_neighbor(pnt, single_layer_ham))), pnt] = np.conj(t_interlayer)
            #Check if on side anti-parallel to pulling axis
            elif second_paxis_pt - pnt <= (((points_per_level[level-1]/6)-1)/2):
                multilayer_ham[pnt, int(tot_num_points + np.max(nearest_neighbor(pnt, single_layer_ham)))] = t_interlayer
                multilayer_ham[int(tot_num_points + np.max(nearest_neighbor(pnt, single_layer_ham))), pnt] = np.conj(t_interlayer)
            #If on side not parallel or anti-parallel to pulling axis
            else:
                multilayer_ham[pnt, int(tot_num_points + pnt + 1)] = t_interlayer
                multilayer_ham[int(tot_num_points + pnt + 1), pnt] = np.conj(t_interlayer)

        #Iterate over points between second_paxis_pt and last point on level
        for pnt in np.arange(second_paxis_pt+1, point_cumulative[level], 2):
            pnt = int(pnt)
            #Check if on side anti-parallel to pulling axis
            if pnt - second_paxis_pt <= (((points_per_level[level-1]/6)-1)/2):
                multilayer_ham[pnt, int(tot_num_points + np.max(nearest_neighbor(pnt, single_layer_ham)))] = t_interlayer
                multilayer_ham[int(tot_num_points + np.max(nearest_neighbor(pnt, single_layer_ham))), pnt] = np.conj(t_interlayer)
            #If on side not parallel or anti-parallel to pulling axis
            else:
                multilayer_ham[pnt, int(tot_num_points + pnt - 1)] = t_interlayer
                multilayer_ham[int(tot_num_points + pnt - 1), pnt] = np.conj(t_interlayer)

    #For last layer which has an intricacy for interlayer connection for anti-parallel to pulling axis side
    #Code in if loops directly copied from above but anti-parallel side portions modified to remove interlayyer connection on that side for last level=nl
    level = nl
    if level%2 == 0:

        first_paxis_pt, second_paxis_pt = pulling_axis_points(point_cumulative[level - 1], points_per_level[level - 1])
        first_paxis_pt = int(first_paxis_pt)
        second_paxis_pt = int(second_paxis_pt)
        # Only first_paxis_pt connects to other layer for even numbered levels
        multilayer_ham[first_paxis_pt, int(
            tot_num_points + np.min(nearest_neighbor(first_paxis_pt, single_layer_ham)))] = t_interlayer
        multilayer_ham[
            int(tot_num_points + np.min(nearest_neighbor(first_paxis_pt, single_layer_ham))), first_paxis_pt] = np.conj(
            t_interlayer)

        # Iterate over part of level between first point and first_paxis_pt
        for pnt in np.arange(point_cumulative[level - 1] + 1, first_paxis_pt, 2):
            pnt = int(pnt)
            # Check if on side not parallel or anti-parallel to pulling axis
            if pnt - point_cumulative[level - 1] <= (((points_per_level[level - 1] / 6) - 1) / 2):
                multilayer_ham[pnt, int(tot_num_points + pnt - 1)] = t_interlayer
                multilayer_ham[int(tot_num_points + pnt - 1), pnt] = np.conj(t_interlayer)
            # If on side parallel to pulling axis
            else:
                multilayer_ham[
                    pnt, int(tot_num_points + np.min(nearest_neighbor(pnt, single_layer_ham)))] = t_interlayer
                multilayer_ham[int(tot_num_points + np.min(nearest_neighbor(pnt, single_layer_ham))), pnt] = np.conj(
                    t_interlayer)

        # Iterate over points between first_paxis_pt and second_paxis_pt
        for pnt in np.arange(first_paxis_pt + 2, second_paxis_pt, 2):
            pnt = int(pnt)
            # Check if on side parallel to pulling axis
            if pnt - first_paxis_pt <= (((points_per_level[level - 1] / 6) - 1) / 2):
                multilayer_ham[
                    pnt, int(tot_num_points + np.min(nearest_neighbor(pnt, single_layer_ham)))] = t_interlayer
                multilayer_ham[int(tot_num_points + np.min(nearest_neighbor(pnt, single_layer_ham))), pnt] = np.conj(
                    t_interlayer)
            # Check if on side anti-parallel to pulling axis
            elif second_paxis_pt - pnt <= (((points_per_level[level - 1] / 6) - 1) / 2):
                continue
            # If on side not parallel or anti-parallel to pulling axis
            else:
                multilayer_ham[pnt, int(tot_num_points + pnt + 1)] = t_interlayer
                multilayer_ham[int(tot_num_points + pnt + 1), pnt] = np.conj(t_interlayer)

        # Iterate over points between second_paxis_pt and last point on level
        for pnt in np.arange(second_paxis_pt + 1, point_cumulative[level], 2):
            pnt = int(pnt)
            # Check if on side anti-parallel to pulling axis
            if pnt - second_paxis_pt <= (((points_per_level[level - 1] / 6) - 1) / 2):
                continue
            # If on side not parallel or anti-parallel to pulling axis
            else:
                multilayer_ham[pnt, int(tot_num_points + pnt - 1)] = t_interlayer
                multilayer_ham[int(tot_num_points + pnt - 1), pnt] = np.conj(t_interlayer)

    elif level%2 == 1:

        first_paxis_pt, second_paxis_pt = pulling_axis_points(point_cumulative[level - 1], points_per_level[level - 1])
        first_paxis_pt = int(first_paxis_pt)
        second_paxis_pt = int(second_paxis_pt)
        # Only second_paxis_pt is connected to other layer in odd numbered levels (DELETED HERE for level = nl)

        # Connection to other layer for first site in level; in odd numbered levels first site on level is connected to last site on level
        multilayer_ham[
            int(point_cumulative[level - 1]), int(tot_num_points + point_cumulative[level] - 1)] = t_interlayer
        multilayer_ham[int(tot_num_points + point_cumulative[level] - 1), int(point_cumulative[level - 1])] = np.conj(
            t_interlayer)

        # Iterate over part of level between first site and first_paxis_pt
        for pnt in np.arange(point_cumulative[level - 1] + 2, first_paxis_pt, 2):
            pnt = int(pnt)
            # Check if on side that is not parallel to pulling axis
            if pnt - point_cumulative[level - 1] <= (((points_per_level[level - 1] / 6) - 1) / 2):
                multilayer_ham[pnt, int(tot_num_points + pnt - 1)] = t_interlayer
                multilayer_ham[int(tot_num_points + pnt - 1), pnt] = np.conj(t_interlayer)
            # If is on side parallel to pulling axis
            else:
                multilayer_ham[
                    pnt, int(tot_num_points + np.min(nearest_neighbor(pnt, single_layer_ham)))] = t_interlayer
                multilayer_ham[int(tot_num_points + np.min(nearest_neighbor(pnt, single_layer_ham))), pnt] = np.conj(
                    t_interlayer)

        # Iterate over part of level between first_paxis_pt and second_paxis_pt
        for pnt in np.arange(first_paxis_pt + 1, second_paxis_pt, 2):
            pnt = int(pnt)
            # Check if on side parallel to pulling axis
            if pnt - first_paxis_pt <= (((points_per_level[level - 1] / 6) - 1) / 2):
                multilayer_ham[
                    pnt, int(tot_num_points + np.min(nearest_neighbor(pnt, single_layer_ham)))] = t_interlayer
                multilayer_ham[int(tot_num_points + np.min(nearest_neighbor(pnt, single_layer_ham))), pnt] = np.conj(
                    t_interlayer)
            # Check if on side anti-parallel to pulling axis
            elif second_paxis_pt - pnt <= (((points_per_level[level - 1] / 6) - 1) / 2):
                continue
            # If on side not parallel to pulling axis
            else:
                multilayer_ham[pnt, int(tot_num_points + pnt + 1)] = t_interlayer
                multilayer_ham[int(tot_num_points + pnt + 1), pnt] = np.conj(t_interlayer)

        # Iterate over part of level between second_axis_pt and last point on level
        for pnt in np.arange(second_paxis_pt + 2, point_cumulative[level], 2):
            pnt = int(pnt)
            # Check if on side anti-parallel to pulling axis
            if pnt - second_paxis_pt <= (((points_per_level[level - 1] / 6) - 1) / 2):
                continue
            # If on side not parallel or anti-parallel to pulling axis
            else:
                multilayer_ham[pnt, int(tot_num_points + pnt - 1)] = t_interlayer
                multilayer_ham[int(tot_num_points + pnt - 1), pnt] = np.conj(t_interlayer)

    return(multilayer_ham)

#Copied and pasted from non-PBC code above. Only change is when defining block hamiltonian
#Specifically added line defining pbc ham then set that as top left and top right block in block ham
#Kept regular open BC ham so rest of code still refers to it so no changes happen
#Extra bit of code added to end to account for few extra interlayer connections added by PBC condition
def bilayer_honeycomb_fermi_liquid_H0_PBC(nl, t_interlayer):

    points_per_level, tot_num_points = honeycomb_points(nl)
    tot_num_points = int(tot_num_points)

    point_cumulative = np.array([], dtype=int)
    for a in range(len(points_per_level)):
        if a == 0:
            point_cumulative = np.append(point_cumulative, int(points_per_level[0]))
        elif a == len(points_per_level) - 1:
            point_cumulative = np.append(point_cumulative, int(tot_num_points))
        else:
            point_cumulative = np.append(point_cumulative, int(points_per_level[a] + point_cumulative[-1]))
    print(point_cumulative)
    point_cumulative = np.concatenate((np.array([0]),
                                       point_cumulative))  # So now point_cumulative is site number of first site in nth generation (apart from last entry which is just tot_num_sites)

    single_layer_ham = honeycomb_lattice(nl)
    single_layer_ham_pbc = honeycomb_lattice_periodic_boundary(nl)
    multilayer_ham = np.block([[single_layer_ham_pbc, np.zeros((tot_num_points, tot_num_points))],
                               [np.zeros((tot_num_points, tot_num_points)), single_layer_ham_pbc]])

    # For level 1 specifically
    multilayer_ham[0, tot_num_points + 5] = t_interlayer
    multilayer_ham[tot_num_points + 5, 0] = np.conj(t_interlayer)
    multilayer_ham[2, tot_num_points + 3] = t_interlayer
    multilayer_ham[tot_num_points + 3, 2] = np.conj(t_interlayer)
    multilayer_ham[4, tot_num_points + 18] = t_interlayer
    multilayer_ham[tot_num_points + 18, 4] = np.conj(t_interlayer)

    # For odd numbered levels
    for level in np.arange(3, nl, 2):

        first_paxis_pt, second_paxis_pt = pulling_axis_points(point_cumulative[level - 1], points_per_level[level - 1])
        first_paxis_pt = int(first_paxis_pt)
        second_paxis_pt = int(second_paxis_pt)
        # Only second_paxis_pt is connected to other layer in odd numbered levels
        multilayer_ham[second_paxis_pt, int(
            tot_num_points + np.max(nearest_neighbor(second_paxis_pt, single_layer_ham)))] = t_interlayer
        multilayer_ham[int(tot_num_points + np.max(
            nearest_neighbor(second_paxis_pt, single_layer_ham))), second_paxis_pt] = np.conj(t_interlayer)

        # Connection to other layer for first site in level; in odd numbered levels first site on level is connected to last site on level
        multilayer_ham[
            int(point_cumulative[level - 1]), int(tot_num_points + point_cumulative[level] - 1)] = t_interlayer
        multilayer_ham[int(tot_num_points + point_cumulative[level] - 1), int(point_cumulative[level - 1])] = np.conj(
            t_interlayer)

        # Iterate over part of level between first site and first_paxis_pt
        for pnt in np.arange(point_cumulative[level - 1] + 2, first_paxis_pt, 2):
            pnt = int(pnt)
            # Check if on side that is not parallel to pulling axis
            if pnt - point_cumulative[level - 1] <= (((points_per_level[level - 1] / 6) - 1) / 2):
                multilayer_ham[pnt, int(tot_num_points + pnt - 1)] = t_interlayer
                multilayer_ham[int(tot_num_points + pnt - 1), pnt] = np.conj(t_interlayer)
            # If is on side parallel to pulling axis
            else:
                multilayer_ham[
                    pnt, int(tot_num_points + np.min(nearest_neighbor(pnt, single_layer_ham)))] = t_interlayer
                multilayer_ham[int(tot_num_points + np.min(nearest_neighbor(pnt, single_layer_ham))), pnt] = np.conj(
                    t_interlayer)

        # Iterate over part of level between first_paxis_pt and second_paxis_pt
        for pnt in np.arange(first_paxis_pt + 1, second_paxis_pt, 2):
            pnt = int(pnt)
            # Check if on side parallel to pulling axis
            if pnt - first_paxis_pt <= (((points_per_level[level - 1] / 6) - 1) / 2):
                multilayer_ham[
                    pnt, int(tot_num_points + np.min(nearest_neighbor(pnt, single_layer_ham)))] = t_interlayer
                multilayer_ham[int(tot_num_points + np.min(nearest_neighbor(pnt, single_layer_ham))), pnt] = np.conj(
                    t_interlayer)
            # Check if on side anti-parallel to pulling axis
            elif second_paxis_pt - pnt <= (((points_per_level[level - 1] / 6) - 1) / 2):
                multilayer_ham[
                    pnt, int(tot_num_points + np.max(nearest_neighbor(pnt, single_layer_ham)))] = t_interlayer
                multilayer_ham[int(tot_num_points + np.max(nearest_neighbor(pnt, single_layer_ham))), pnt] = np.conj(
                    t_interlayer)
            # If on side not parallel to pulling axis
            else:
                multilayer_ham[pnt, int(tot_num_points + pnt + 1)] = t_interlayer
                multilayer_ham[int(tot_num_points + pnt + 1), pnt] = np.conj(t_interlayer)

        # Iterate over part of level between second_axis_pt and last point on level
        for pnt in np.arange(second_paxis_pt + 2, point_cumulative[level], 2):
            pnt = int(pnt)
            # Check if on side anti-parallel to pulling axis
            if pnt - second_paxis_pt <= (((points_per_level[level - 1] / 6) - 1) / 2):
                multilayer_ham[
                    pnt, int(tot_num_points + np.max(nearest_neighbor(pnt, single_layer_ham)))] = t_interlayer
                multilayer_ham[int(tot_num_points + np.max(nearest_neighbor(pnt, single_layer_ham))), pnt] = np.conj(
                    t_interlayer)
            # If on side not parallel or anti-parallel to pulling axis
            else:
                multilayer_ham[pnt, int(tot_num_points + pnt - 1)] = t_interlayer
                multilayer_ham[int(tot_num_points + pnt - 1), pnt] = np.conj(t_interlayer)

    # For even numbered levels
    for level in np.arange(2, nl, 2):

        first_paxis_pt, second_paxis_pt = pulling_axis_points(point_cumulative[level - 1], points_per_level[level - 1])
        first_paxis_pt = int(first_paxis_pt)
        second_paxis_pt = int(second_paxis_pt)
        # Only first_paxis_pt connects to other layer for even numbered levels
        multilayer_ham[first_paxis_pt, int(
            tot_num_points + np.min(nearest_neighbor(first_paxis_pt, single_layer_ham)))] = t_interlayer
        multilayer_ham[
            int(tot_num_points + np.min(nearest_neighbor(first_paxis_pt, single_layer_ham))), first_paxis_pt] = np.conj(
            t_interlayer)

        # Iterate over part of level between first point and first_paxis_pt
        for pnt in np.arange(point_cumulative[level - 1] + 1, first_paxis_pt, 2):
            pnt = int(pnt)
            # Check if on side not parallel or anti-parallel to pulling axis
            if pnt - point_cumulative[level - 1] <= (((points_per_level[level - 1] / 6) - 1) / 2):
                multilayer_ham[pnt, int(tot_num_points + pnt - 1)] = t_interlayer
                multilayer_ham[int(tot_num_points + pnt - 1), pnt] = np.conj(t_interlayer)
            # If on side parallel to pulling axis
            else:
                multilayer_ham[
                    pnt, int(tot_num_points + np.min(nearest_neighbor(pnt, single_layer_ham)))] = t_interlayer
                multilayer_ham[int(tot_num_points + np.min(nearest_neighbor(pnt, single_layer_ham))), pnt] = np.conj(
                    t_interlayer)

        # Iterate over points between first_paxis_pt and second_paxis_pt
        for pnt in np.arange(first_paxis_pt + 2, second_paxis_pt, 2):
            pnt = int(pnt)
            # Check if on side parallel to pulling axis
            if pnt - first_paxis_pt <= (((points_per_level[level - 1] / 6) - 1) / 2):
                multilayer_ham[
                    pnt, int(tot_num_points + np.min(nearest_neighbor(pnt, single_layer_ham)))] = t_interlayer
                multilayer_ham[int(tot_num_points + np.min(nearest_neighbor(pnt, single_layer_ham))), pnt] = np.conj(
                    t_interlayer)
            # Check if on side anti-parallel to pulling axis
            elif second_paxis_pt - pnt <= (((points_per_level[level - 1] / 6) - 1) / 2):
                multilayer_ham[
                    pnt, int(tot_num_points + np.max(nearest_neighbor(pnt, single_layer_ham)))] = t_interlayer
                multilayer_ham[int(tot_num_points + np.max(nearest_neighbor(pnt, single_layer_ham))), pnt] = np.conj(
                    t_interlayer)
            # If on side not parallel or anti-parallel to pulling axis
            else:
                multilayer_ham[pnt, int(tot_num_points + pnt + 1)] = t_interlayer
                multilayer_ham[int(tot_num_points + pnt + 1), pnt] = np.conj(t_interlayer)

        # Iterate over points between second_paxis_pt and last point on level
        for pnt in np.arange(second_paxis_pt + 1, point_cumulative[level], 2):
            pnt = int(pnt)
            # Check if on side anti-parallel to pulling axis
            if pnt - second_paxis_pt <= (((points_per_level[level - 1] / 6) - 1) / 2):
                multilayer_ham[
                    pnt, int(tot_num_points + np.max(nearest_neighbor(pnt, single_layer_ham)))] = t_interlayer
                multilayer_ham[int(tot_num_points + np.max(nearest_neighbor(pnt, single_layer_ham))), pnt] = np.conj(
                    t_interlayer)
            # If on side not parallel or anti-parallel to pulling axis
            else:
                multilayer_ham[pnt, int(tot_num_points + pnt - 1)] = t_interlayer
                multilayer_ham[int(tot_num_points + pnt - 1), pnt] = np.conj(t_interlayer)

    # For last layer which has an intricacy for interlayer connection for anti-parallel to pulling axis side
    # Code in if loops directly copied from above but anti-parallel side portions modified to remove interlayyer connection on that side for last level=nl
    level = nl
    if level % 2 == 0:

        first_paxis_pt, second_paxis_pt = pulling_axis_points(point_cumulative[level - 1], points_per_level[level - 1])
        first_paxis_pt = int(first_paxis_pt)
        second_paxis_pt = int(second_paxis_pt)
        # Only first_paxis_pt connects to other layer for even numbered levels
        multilayer_ham[first_paxis_pt, int(
            tot_num_points + np.min(nearest_neighbor(first_paxis_pt, single_layer_ham)))] = t_interlayer
        multilayer_ham[
            int(tot_num_points + np.min(nearest_neighbor(first_paxis_pt, single_layer_ham))), first_paxis_pt] = np.conj(
            t_interlayer)

        # Iterate over part of level between first point and first_paxis_pt
        for pnt in np.arange(point_cumulative[level - 1] + 1, first_paxis_pt, 2):
            pnt = int(pnt)
            # Check if on side not parallel or anti-parallel to pulling axis
            if pnt - point_cumulative[level - 1] <= (((points_per_level[level - 1] / 6) - 1) / 2):
                multilayer_ham[pnt, int(tot_num_points + pnt - 1)] = t_interlayer
                multilayer_ham[int(tot_num_points + pnt - 1), pnt] = np.conj(t_interlayer)
            # If on side parallel to pulling axis
            else:
                multilayer_ham[
                    pnt, int(tot_num_points + np.min(nearest_neighbor(pnt, single_layer_ham)))] = t_interlayer
                multilayer_ham[int(tot_num_points + np.min(nearest_neighbor(pnt, single_layer_ham))), pnt] = np.conj(
                    t_interlayer)

        # Iterate over points between first_paxis_pt and second_paxis_pt
        for pnt in np.arange(first_paxis_pt + 2, second_paxis_pt, 2):
            pnt = int(pnt)
            # Check if on side parallel to pulling axis
            if pnt - first_paxis_pt <= (((points_per_level[level - 1] / 6) - 1) / 2):
                multilayer_ham[
                    pnt, int(tot_num_points + np.min(nearest_neighbor(pnt, single_layer_ham)))] = t_interlayer
                multilayer_ham[int(tot_num_points + np.min(nearest_neighbor(pnt, single_layer_ham))), pnt] = np.conj(
                    t_interlayer)
            # Check if on side anti-parallel to pulling axis
            elif second_paxis_pt - pnt <= (((points_per_level[level - 1] / 6) - 1) / 2):
                continue
            # If on side not parallel or anti-parallel to pulling axis
            else:
                multilayer_ham[pnt, int(tot_num_points + pnt + 1)] = t_interlayer
                multilayer_ham[int(tot_num_points + pnt + 1), pnt] = np.conj(t_interlayer)

        # Iterate over points between second_paxis_pt and last point on level
        for pnt in np.arange(second_paxis_pt + 1, point_cumulative[level], 2):
            pnt = int(pnt)
            # Check if on side anti-parallel to pulling axis
            if pnt - second_paxis_pt <= (((points_per_level[level - 1] / 6) - 1) / 2):
                continue
            # If on side not parallel or anti-parallel to pulling axis
            else:
                multilayer_ham[pnt, int(tot_num_points + pnt - 1)] = t_interlayer
                multilayer_ham[int(tot_num_points + pnt - 1), pnt] = np.conj(t_interlayer)

    elif level % 2 == 1:

        first_paxis_pt, second_paxis_pt = pulling_axis_points(point_cumulative[level - 1], points_per_level[level - 1])
        first_paxis_pt = int(first_paxis_pt)
        second_paxis_pt = int(second_paxis_pt)
        # Only second_paxis_pt is connected to other layer in odd numbered levels (DELETED HERE for level = nl)

        # Connection to other layer for first site in level; in odd numbered levels first site on level is connected to last site on level
        multilayer_ham[
            int(point_cumulative[level - 1]), int(tot_num_points + point_cumulative[level] - 1)] = t_interlayer
        multilayer_ham[int(tot_num_points + point_cumulative[level] - 1), int(point_cumulative[level - 1])] = np.conj(
            t_interlayer)

        # Iterate over part of level between first site and first_paxis_pt
        for pnt in np.arange(point_cumulative[level - 1] + 2, first_paxis_pt, 2):
            pnt = int(pnt)
            # Check if on side that is not parallel to pulling axis
            if pnt - point_cumulative[level - 1] <= (((points_per_level[level - 1] / 6) - 1) / 2):
                multilayer_ham[pnt, int(tot_num_points + pnt - 1)] = t_interlayer
                multilayer_ham[int(tot_num_points + pnt - 1), pnt] = np.conj(t_interlayer)
            # If is on side parallel to pulling axis
            else:
                multilayer_ham[
                    pnt, int(tot_num_points + np.min(nearest_neighbor(pnt, single_layer_ham)))] = t_interlayer
                multilayer_ham[int(tot_num_points + np.min(nearest_neighbor(pnt, single_layer_ham))), pnt] = np.conj(
                    t_interlayer)

        # Iterate over part of level between first_paxis_pt and second_paxis_pt
        for pnt in np.arange(first_paxis_pt + 1, second_paxis_pt, 2):
            pnt = int(pnt)
            # Check if on side parallel to pulling axis
            if pnt - first_paxis_pt <= (((points_per_level[level - 1] / 6) - 1) / 2):
                multilayer_ham[
                    pnt, int(tot_num_points + np.min(nearest_neighbor(pnt, single_layer_ham)))] = t_interlayer
                multilayer_ham[int(tot_num_points + np.min(nearest_neighbor(pnt, single_layer_ham))), pnt] = np.conj(
                    t_interlayer)
            # Check if on side anti-parallel to pulling axis
            elif second_paxis_pt - pnt <= (((points_per_level[level - 1] / 6) - 1) / 2):
                continue
            # If on side not parallel to pulling axis
            else:
                multilayer_ham[pnt, int(tot_num_points + pnt + 1)] = t_interlayer
                multilayer_ham[int(tot_num_points + pnt + 1), pnt] = np.conj(t_interlayer)

        # Iterate over part of level between second_axis_pt and last point on level
        for pnt in np.arange(second_paxis_pt + 2, point_cumulative[level], 2):
            pnt = int(pnt)
            # Check if on side anti-parallel to pulling axis
            if pnt - second_paxis_pt <= (((points_per_level[level - 1] / 6) - 1) / 2):
                continue
            # If on side not parallel or anti-parallel to pulling axis
            else:
                multilayer_ham[pnt, int(tot_num_points + pnt - 1)] = t_interlayer
                multilayer_ham[int(tot_num_points + pnt - 1), pnt] = np.conj(t_interlayer)

    #Finally need part of code to add last extra interlayer connections due to introduced PBC
    level = nl
    points_per_side = points_per_level[level - 1] / 6
    starting_point_pullingaxis_side = point_cumulative[level - 1] + (((points_per_level[level - 1] / 6) - 1) / 2) + 1

    for pnt in np.arange(starting_point_pullingaxis_side, starting_point_pullingaxis_side + points_per_side, 2):
        pbc_connected_point = np.max(np.where(single_layer_ham_pbc[int(pnt), :]!=0)[0])
        multilayer_ham[int(pbc_connected_point), int(pnt + tot_num_points)] = t_interlayer
        multilayer_ham[int(pnt + tot_num_points), int(pbc_connected_point)] = np.conj(t_interlayer)

    return (multilayer_ham)
