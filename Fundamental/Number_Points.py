import numpy as np

# p = 14
# q = 4
# num_levels = 5

def points(p,q,num_levels):

    points_perlevel = np.array([])
    if q == 3:
        for n in range(num_levels):
            if n == 0:
                points_perlevel = np.append(points_perlevel, p)
            elif n == 1:
                points_perlevel = np.append(points_perlevel, p*(p-q))
            elif n == 2:
                num_secondlevel_polygons = points_perlevel[-2]
                n4borders_n2 = num_secondlevel_polygons
                n3borders_n2 = num_secondlevel_polygons*(p-q-2)
                p4borders = p-4
                p3borders = p-3
                points_perlevel = np.append(points_perlevel, n4borders_n2*p4borders + n3borders_n2*p3borders)
            elif n == 3:
                n4borders_n3 = n4borders_n2 + n3borders_n2
                n3borders_n3 = (p-q-3)*n4borders_n2 + (p-q-2)*n3borders_n2
                points_perlevel = np.append(points_perlevel, n4borders_n3*p4borders + n3borders_n3*p3borders)
            elif n == 4:
                n4borders_n4 = n4borders_n3 + n3borders_n3
                n3borders_n4 = (p-q-3)*n4borders_n3 + (p-q-2)*n3borders_n3
                points_perlevel = np.append(points_perlevel, n4borders_n4 * p4borders + n3borders_n4 * p3borders)
            elif n == 5:
                n4borders_n5 = n4borders_n4 + n3borders_n4
                n3borders_n5 = (p-q-3)*n4borders_n4 + (p-q-2)*n3borders_n4
                points_perlevel = np.append(points_perlevel, n4borders_n5 * p4borders + n3borders_n5 * p3borders)
            # if num_levels == 6:

            # if n == 5:
            #     n4borders_n5 = n4borders_n4 + n3borders_n4
            #     n3borders_n5 = (p - q - 3) * n4borders_n4 + (p - q - 2) * n3borders_n4
            #     points_perlevel = np.append(points_perlevel, n4borders_n5 * p4borders + n3borders_n5 * p3borders)

    elif q == 4:
        for n in range(num_levels):
            if n == 0:
                points_perlevel = np.append(points_perlevel, p)
            elif n == 1:
                n3borders_n1 = p
                n2borders_n1 = p
                points_perlevel = np.append(points_perlevel, n3borders_n1*(p-3+1))
            elif n == 2:
                n3borders_n2 = points_perlevel[-1]-p
                n2borders_n2 = points_perlevel[-1]
                points_perlevel = np.append(points_perlevel, n3borders_n2*(p-3+1) + n2borders_n1*(p-3))
            elif n == 3:
                n3borders_n3 = points_perlevel[-1] - p
                n2borders_n3 = points_perlevel[-1]
                points_perlevel = np.append(points_perlevel, n3borders_n3*(p-3+1))

    elif q == 5:
        for n in range(num_levels):
            if n == 0:
                points_perlevel = np.append(points_perlevel, p)
            elif n == 1:
                points_perlevel = np.append(points_perlevel, p*(p-3+1) + p)
                n1_2siders = 2*p
                n1_3siders = p
            elif n == 2:
                points_perlevel = np.append(points_perlevel, n1_2siders*(p-3+2) + n1_3siders*(p-3)*(p-3+2) - n1_3siders*1)

    total_numpoints = np.sum(points_perlevel)

    # print(total_numpoints)
    # print(points_perlevel)

    return(points_perlevel,total_numpoints)

# print(points(10,5,3))