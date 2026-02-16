import numpy as np
import math
def number_square_points(num_levels):
    return(num_levels**2)

def H0(num_levels):
    t=1
    totnumpoints = number_square_points(num_levels)
    ham = np.zeros((totnumpoints,totnumpoints))
    toplayer = False
    for i in range(totnumpoints-1): #Iterate over all points; dont need last point since, way coded last point needs no new connections added
        if toplayer == True: #top layer means no connections to an above layer
            ham[i,i+1] = t
            ham[i+1,i] = t
        elif (i+1)%num_levels == 0: #Check to see if point is on the right boundary of the square lattice
            ham[i,i+num_levels] = t
            ham[i+num_levels,i] = t
            if i+1 == totnumpoints-num_levels: #Check to see if the next point is first point on topmost layer
                toplayer = True
        else: #Connect point to point to right and point above on next level
            ham[i,i+num_levels] = t
            ham[i+num_levels,i] = t
            ham[i,i+1] = t
            ham[i+1,i] = t
    return(ham)

def square_site_assignment(num_levels):
    asites = np.array([])
    bsites = np.array([])
    site_num_mat = np.arange(num_levels**2).reshape(num_levels,num_levels)
    for row in range(num_levels):
        if row%2 == 0:
            asites = np.append(asites, [i for i in site_num_mat[row,:] if i % 2 == 0])
            bsites = np.append(bsites, [i for i in site_num_mat[row,:] if i % 2 == 1])
        elif row%2 == 1:
            asites = np.append(asites, [i for i in site_num_mat[row, :] if i % 2 == 1])
            bsites = np.append(bsites, [i for i in site_num_mat[row, :] if i % 2 == 0])
    return(asites, bsites)

def square_lattice_nonherm(num_levels,alpha):
    ham_oldbasis = H0(num_levels)
    asites, bsites = square_site_assignment(num_levels)
    newbasis = np.concatenate((asites, bsites))
    newbasis_int = [int(nbi) for nbi in newbasis]
    ham_newbasis = ham_oldbasis[newbasis_int,:][:,newbasis_int]
    hcdw = np.eye(num_levels**2)
    for i in range(int((num_levels**2)/2), int(num_levels**2)): hcdw[i,i] = -1
    return(ham_newbasis + alpha*np.matmul(hcdw,ham_newbasis))

def square_lattice_PeierlsSubstitution(num_levels, amag):
    t = 1
    ps_xvals = np.array([])
    ps_tvals = np.array([])
    for nlev in range(num_levels):
        if nlev == 0:
            ps_xvals = np.append(ps_xvals, 0) #Any choice for hopping on first level; just pick t1=t
            ps_tvals = np.append(ps_tvals, t)
        else:
            ps_xvals = np.append(ps_xvals, ps_xvals[-1]-amag) #Pattern is alpha = x_n - x_{n+1}
            ps_tvals = np.append(ps_tvals, t*np.exp(1j*ps_xvals[-1]))
    totnumpoints = number_square_points(num_levels)
    ham = np.zeros((totnumpoints, totnumpoints), dtype=np.complex_)
    toplayer = False
    for i in range(totnumpoints - 1):  # Iterate over all points; dont need last point since, way coded last point needs no new connections added
        whatlevel = math.floor(i/num_levels) #tells code what level it is on
        if toplayer == True:  # top layer means no connections to an above layer
            ham[i, i + 1] = ps_tvals[whatlevel]
            ham[i + 1, i] = np.conjugate(ps_tvals[whatlevel])
        elif (i + 1) % num_levels == 0:  # Check to see if point is on the right boundary of the square lattice
            ham[i, i + num_levels] = t
            ham[i + num_levels, i] = t
            if i + 1 == totnumpoints - num_levels:  # Check to see if the next point is first point on topmost layer
                toplayer = True
        else:  # Connect point to point to right and point above on next level
            ham[i, i + num_levels] = t
            ham[i + num_levels, i] = t
            ham[i, i + 1] = ps_tvals[whatlevel]
            ham[i + 1, i] = np.conjugate(ps_tvals[whatlevel])
    return (ham)

def square_lattice_nonherm_PeierlsSubstitution(num_levels, amag, alpha):
    ham_oldbasis = square_lattice_PeierlsSubstitution(num_levels, amag)
    asites, bsites = square_site_assignment(num_levels)
    newbasis = np.concatenate((asites, bsites))
    newbasis_int = [int(nbi) for nbi in newbasis]
    ham_newbasis = ham_oldbasis[newbasis_int, :][:, newbasis_int]
    hcdw = np.eye(num_levels ** 2)
    for i in range(int((num_levels ** 2) / 2), int(num_levels ** 2)): hcdw[i, i] = -1
    return (ham_newbasis + alpha * np.matmul(hcdw, ham_newbasis))

def add_periodic_boundary(nl,ham):
    bottom_edge_sites = np.arange(nl)
    top_edge_sites = np.arange((nl**2)-nl, nl**2)
    left_edge_sites = np.arange(0, (nl**2)-nl+1, nl)
    right_edge_sites = np.arange(nl-1, (nl**2), nl)
    for col in range(len(bottom_edge_sites)):
        ham[bottom_edge_sites[col], top_edge_sites[col]] = 1
        ham[top_edge_sites[col], bottom_edge_sites[col]] = 1
    for row in range(len(left_edge_sites)):
        ham[left_edge_sites[row], right_edge_sites[row]] = 1
        ham[right_edge_sites[row], left_edge_sites[row]] = 1
    return(ham)

def H0_PBC(nl):
    ham = H0(nl)
    ham_pbc = add_periodic_boundary(nl, ham)
    return(ham_pbc)

def square_lattice_nonherm_PBC(num_levels,alpha):
    ham_oldbasis = H0_PBC(num_levels)
    asites, bsites = square_site_assignment(num_levels)
    newbasis = np.concatenate((asites, bsites))
    newbasis_int = [int(nbi) for nbi in newbasis]
    ham_newbasis = ham_oldbasis[newbasis_int,:][:,newbasis_int]
    hcdw = np.eye(num_levels**2)
    for i in range(int((num_levels**2)/2), int(num_levels**2)): hcdw[i,i] = -1
    return(ham_newbasis + alpha*np.matmul(hcdw,ham_newbasis))