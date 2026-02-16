import numpy as np
from Fundamental.Number_Points import points
import scipy
# from Self_Consistent_Hartree_Fock.Self_Consistent_Hartree_Fock import site_assignment
# from Self_Consistent_Hartree_Fock.Self_Consistent_Hartree_Fock import selfconsist_hartreefock
from Fundamental.Hamiltonian import H0

def site_assignment(p,q,num_levels,ham):

    totalnum_points = points(p,q,num_levels)[1] #Get total num points from points func

    a_sites = np.array([]) #Initialize sublattice arrays
    b_sites = np.array([])
    #Know sites of first gen; let first site be a_site then they just alternate
    first_gen_sites = np.arange(points(p,q,num_levels)[0][0])
    a_sites = np.append(a_sites, np.where(first_gen_sites%2 == 0)[0])
    b_sites = np.append(b_sites, np.where(first_gen_sites%2 == 1)[0])

    site_assign_progress_count = 0 #keep track of how long to assign all sites to a or b
    while len(a_sites) != (totalnum_points/2) or len(b_sites) != (totalnum_points/2): #or qualifier makes sense because if either or is true then while loop continues
        for i in a_sites:
            b_sites = np.append(b_sites, np.where(ham[int(i),:] != 0)[0]) #append to b where a has connection (works by definition)
            b_sites = np.unique(b_sites)
        for i in b_sites:
            a_sites = np.append(a_sites, np.where(ham[int(i),:] != 0)[0]) #vice versa of above code
            a_sites = np.unique(a_sites)
        site_assign_progress_count += 1
        # print(site_assign_progress_count)
    # print('site assignment while loop done')
    return(a_sites, b_sites)

def inter_sublattice_block(p,q,num_levels,t):

    points_perlevel, totalnum_points = points(p,q,num_levels)
    ham = H0(p,q,num_levels) #generate usual hermitian tight binding hamiltonian
    a_sites, b_sites = site_assignment(p,q,num_levels,ham) #assign sites to a or b sublattices

    h0block1 = np.zeros((int(totalnum_points/2),int(totalnum_points/2))) #Initialize blocks; values to be changed
    h0block2 = np.zeros((int(totalnum_points/2),int(totalnum_points/2)))

    # b_site_connections = np.array([])
    for a in range(len(a_sites)):
        connections = np.where(ham[int(a_sites[a]),:] != 0)[0] #For given a_site what sites are connected to it
        for i in connections:
            # b_site_connections = np.append(b_site_connections, np.where(i == b_sites) + int(totalnum_points/2))
            # h0block[a][np.where(i == b_sites)[0] + int(totalnum_points/2)] = t
            h0block1[a][np.where(i == b_sites)[0]] = t #a for row as row is a_site index and np.where(...) for column since column is b_site index

    # a_site_connections = np.array([])
    for b in range(len(b_sites)): #Mirror of above code
        connections = np.where(ham[int(b_sites[b]), :] != 0)[0]
        for i in connections:
            # a_site_connections = np.append(a_site_connections, np.where(i == a_sites))
            # h0block[b+int(totalnum_points/2)][np.where(i == a_sites)[0]] = t
            h0block2[b][np.where(i == a_sites)[0]] = t

    return(h0block1, h0block2)

from Fundamental.Honeycomb_Lattice import honeycomb_points
from Fundamental.Honeycomb_Lattice import honeycomb_lattice

def site_assignment_honeycomb(nl,ham):

    totalnum_points = honeycomb_points(nl)[1]

    a_sites = np.array([]) #initialize sublattice lists
    b_sites = np.array([])
    #Know sites of first gen; let first site be a_site then they just alternate
    first_gen_sites = np.arange(honeycomb_points(nl)[0][0])
    a_sites = np.append(a_sites, np.where(first_gen_sites%2 == 0)[0])
    b_sites = np.append(b_sites, np.where(first_gen_sites%2 == 1)[0])

    site_assign_progress_count = 0
    while len(a_sites) != (totalnum_points/2) or len(b_sites) != (totalnum_points/2):
        for i in a_sites:
            b_sites = np.append(b_sites, np.where(ham[int(i),:] != 0)[0])
            b_sites = np.unique(b_sites)
        for i in b_sites:
            a_sites = np.append(a_sites, np.where(ham[int(i),:] != 0)[0])
            a_sites = np.unique(a_sites)
        site_assign_progress_count += 1
        print(site_assign_progress_count)
    print('site assignment while loop done')
    return(a_sites, b_sites)

def inter_sublattice_block_honeycomb(nl,t):

    points_perlevel, totalnum_points = honeycomb_points(nl)
    ham = honeycomb_lattice(nl)
    a_sites, b_sites = site_assignment_honeycomb(nl,ham)

    h0block1 = np.zeros((int(totalnum_points/2),int(totalnum_points/2)))
    h0block2 = np.zeros((int(totalnum_points/2),int(totalnum_points/2)))

    # b_site_connections = np.array([])
    for a in range(len(a_sites)):
        connections = np.where(ham[int(a_sites[a]),:] != 0)[0]
        for i in connections:
            # b_site_connections = np.append(b_site_connections, np.where(i == b_sites) + int(totalnum_points/2))
            # h0block[a][np.where(i == b_sites)[0] + int(totalnum_points/2)] = t
            h0block1[a][np.where(i == b_sites)[0]] = t

    # a_site_connections = np.array([])
    for b in range(len(b_sites)):
        connections = np.where(ham[int(b_sites[b]), :] != 0)[0]
        for i in connections:
            # a_site_connections = np.append(a_site_connections, np.where(i == a_sites))
            # h0block[b+int(totalnum_points/2)][np.where(i == a_sites)[0]] = t
            h0block2[b][np.where(i == a_sites)[0]] = t

    return(h0block1, h0block2)

def NonHermitian_Hamiltonian(p,q,num_levels,alpha,t):

    points_perlevel, totalnum_points = points(p,q,num_levels)
    # t = 1

    h0block_part1 = np.zeros((int(totalnum_points/2),int(totalnum_points/2)))
    # h0block_part2 = t*np.eye(int(totalnum_points/2))
    h0block_part2, foo = inter_sublattice_block(p,q,num_levels,t)
    h0block_part3 = np.transpose(h0block_part2)
    h0 = np.block([[h0block_part1, h0block_part2],[h0block_part3, h0block_part1]])

    hcdwblock_part1 = np.eye(int(totalnum_points/2))
    '''
    # initial_ham = H0(p,q,num_levels)
    # deltas = selfconsist_hartreefock(p,q,num_levels,initial_ham,0.1,10**(-14),[1])[0]
    # delta_diagonal = np.repeat(deltas, np.size(initial_ham,0)/2)
    # hcdwblock_part1_asites = np.diag(delta_diagonal)
    # hcdwblock_part1_bsites = -1*np.diag(delta_diagonal)
    '''
    hcdwblock_part2 = np.zeros((int(totalnum_points/2),int(totalnum_points/2)))
    hcdw = np.block([[hcdwblock_part1, hcdwblock_part2],[hcdwblock_part2, -1*hcdwblock_part1]])
    '''
    hcdw = np.block([[hcdwblock_part1_asites,hcdwblock_part2],[hcdwblock_part2, hcdwblock_part1_bsites]])
    '''
    # foo = alpha*hcdw*h0
    nh_ham = h0 + alpha*np.matmul(hcdw,h0)

    return(nh_ham)

# def NonHermitian_Hamiltonian_Dagger(p,q,num_levels,alpha,t):
#
#     points_perlevel, totalnum_points = points(p,q,num_levels)
#     # t = 1
#
#     h0block_part1 = np.zeros((int(totalnum_points/2),int(totalnum_points/2)))
#     # h0block_part2 = t*np.eye(int(totalnum_points/2))
#     h0block_part2, foo = inter_sublattice_block(p,q,num_levels,t)
#     h0block_part3 = np.transpose(h0block_part2)
#     h0 = np.block([[h0block_part1, h0block_part2],[h0block_part3, h0block_part1]])
#
#     hcdwblock_part1 = np.eye(int(totalnum_points/2))
#     '''
#     # initial_ham = H0(p,q,num_levels)
#     # deltas = selfconsist_hartreefock(p,q,num_levels,initial_ham,0.1,10**(-14),[1])[0]
#     # delta_diagonal = np.repeat(deltas, np.size(initial_ham,0)/2)
#     # hcdwblock_part1_asites = np.diag(delta_diagonal)
#     # hcdwblock_part1_bsites = -1*np.diag(delta_diagonal)
#     '''
#     hcdwblock_part2 = np.zeros((int(totalnum_points/2),int(totalnum_points/2)))
#     hcdw = np.block([[hcdwblock_part1, hcdwblock_part2],[hcdwblock_part2, -1*hcdwblock_part1]])
#     '''
#     hcdw = np.block([[hcdwblock_part1_asites,hcdwblock_part2],[hcdwblock_part2, hcdwblock_part1_bsites]])
#     '''
#     # foo = alpha*hcdw*h0
#     nh_ham = h0 - alpha*np.matmul(hcdw,h0)
#
#     return(nh_ham)


def NonHermitian_Honeycomb(nl,t,alpha):
    points_perlevel, totalnum_points = honeycomb_points(nl)

    h0block_part1 = np.zeros((int(totalnum_points / 2), int(totalnum_points / 2)))
    # h0block_part2 = t*np.eye(int(totalnum_points/2))
    h0block_part2, foo = inter_sublattice_block_honeycomb(nl,t)
    h0block_part3 = np.transpose(h0block_part2)
    h0 = np.block([[h0block_part1, h0block_part2], [h0block_part3, h0block_part1]])

    hcdwblock_part1 = np.eye(int(totalnum_points/2))
    hcdwblock_part2 = np.zeros((int(totalnum_points / 2), int(totalnum_points / 2)))
    hcdw = np.block([[hcdwblock_part1, hcdwblock_part2], [hcdwblock_part2, -1 * hcdwblock_part1]])

    nh_ham = h0 + alpha * np.matmul(hcdw, h0)

    return (nh_ham)

def inter_sublattice_block_honeycomb_periodic_boundary(nl,t):
    from Fundamental.Honeycomb_Lattice import honeycomb_lattice_periodic_boundary
    points_perlevel, totalnum_points = honeycomb_points(nl)
    ham = honeycomb_lattice_periodic_boundary(nl)
    a_sites, b_sites = site_assignment_honeycomb(nl,ham)

    h0block1 = np.zeros((int(totalnum_points/2),int(totalnum_points/2)))
    h0block2 = np.zeros((int(totalnum_points/2),int(totalnum_points/2)))

    # b_site_connections = np.array([])
    for a in range(len(a_sites)):
        connections = np.where(ham[int(a_sites[a]),:] != 0)[0]
        for i in connections:
            # b_site_connections = np.append(b_site_connections, np.where(i == b_sites) + int(totalnum_points/2))
            # h0block[a][np.where(i == b_sites)[0] + int(totalnum_points/2)] = t
            h0block1[a][np.where(i == b_sites)[0]] = t

    # a_site_connections = np.array([])
    for b in range(len(b_sites)):
        connections = np.where(ham[int(b_sites[b]), :] != 0)[0]
        for i in connections:
            # a_site_connections = np.append(a_site_connections, np.where(i == a_sites))
            # h0block[b+int(totalnum_points/2)][np.where(i == a_sites)[0]] = t
            h0block2[b][np.where(i == a_sites)[0]] = t

    return(h0block1, h0block2)

def NonHermitian_Honeycomb_periodic_boundary(nl,t,alpha):
    points_perlevel, totalnum_points = honeycomb_points(nl)

    h0block_part1 = np.zeros((int(totalnum_points / 2), int(totalnum_points / 2)))
    # h0block_part2 = t*np.eye(int(totalnum_points/2))
    h0block_part2, foo = inter_sublattice_block_honeycomb_periodic_boundary(nl,t)
    h0block_part3 = np.transpose(h0block_part2)
    h0 = np.block([[h0block_part1, h0block_part2], [h0block_part3, h0block_part1]])

    hcdwblock_part1 = np.eye(int(totalnum_points/2))
    hcdwblock_part2 = np.zeros((int(totalnum_points / 2), int(totalnum_points / 2)))
    hcdw = np.block([[hcdwblock_part1, hcdwblock_part2], [hcdwblock_part2, -1 * hcdwblock_part1]])

    nh_ham = h0 + alpha * np.matmul(hcdw, h0)

    return (nh_ham)

def inter_sublattice_block_PeierlsSub(p,q,num_levels,t,alphamag):

    from Fundamental.Hamiltonian_PeierlsSubstitution import H0 as H0_mag
    points_perlevel, totalnum_points = points(p,q,num_levels)
    ham = H0_mag(p,q,num_levels,alphamag)
    a_sites, b_sites = site_assignment(p,q,num_levels,ham)

    h0block1 = np.zeros((int(totalnum_points/2),int(totalnum_points/2)), dtype=np.complex_)
    h0block2 = np.zeros((int(totalnum_points/2),int(totalnum_points/2)), dtype=np.complex_)

    # b_site_connections = np.array([])
    for a in range(len(a_sites)):
        connections = np.where(ham[int(a_sites[a]),:] != 0)[0]
        for i in connections:
            # b_site_connections = np.append(b_site_connections, np.where(i == b_sites) + int(totalnum_points/2))
            # h0block[a][np.where(i == b_sites)[0] + int(totalnum_points/2)] = t
            h0block1[a][np.where(i == b_sites)[0]] = ham[int(a_sites[a]), int(i)]

    # a_site_connections = np.array([])
    for b in range(len(b_sites)):
        connections = np.where(ham[int(b_sites[b]), :] != 0)[0]
        for i in connections:
            # a_site_connections = np.append(a_site_connections, np.where(i == a_sites))
            # h0block[b+int(totalnum_points/2)][np.where(i == a_sites)[0]] = t
            h0block2[b][np.where(i == a_sites)[0]] = ham[int(b_sites[b]), int(i)]

    return(h0block1, h0block2)

def NonHermitian_PeierlsSub_Hamiltonian(p,q,num_levels,alpha,t,alphamag):

    points_perlevel, totalnum_points = points(p, q, num_levels)
    # t = 1

    h0block_part1 = np.zeros((int(totalnum_points / 2), int(totalnum_points / 2)))
    # h0block_part2 = t*np.eye(int(totalnum_points/2))
    h0block_part2, foo = inter_sublattice_block_PeierlsSub(p, q, num_levels, t, alphamag)
    h0block_part3 = np.conj(np.transpose(h0block_part2))
    h0 = np.block([[h0block_part1, h0block_part2], [h0block_part3, h0block_part1]])

    hcdwblock_part1 = np.eye(int(totalnum_points / 2))
    hcdwblock_part2 = np.zeros((int(totalnum_points / 2), int(totalnum_points / 2)))
    hcdw = np.block([[hcdwblock_part1, hcdwblock_part2], [hcdwblock_part2, -1 * hcdwblock_part1]])
    # foo = alpha*hcdw*h0
    nh_ham = h0 + alpha * np.matmul(hcdw, h0)

    return (nh_ham)

from Fundamental.Honeycomb_Lattice import honeycomb_PeierlsSubstitution
def NonHermitian_PeierlsSub_Honeycomb(nl,alpha,alphamag):

    points_perlevel, totalnum_points = honeycomb_points(nl)

    original_h0 = honeycomb_PeierlsSubstitution(nl,alphamag)
    asites, bsites = site_assignment_honeycomb(nl, original_h0)
    asites = [ int(i) for i in asites ]
    bsites = [ int(i) for i in bsites ]
    newbasis = np.concatenate((asites, bsites))
    new_h0 = original_h0[newbasis,:][:,newbasis]

    hcdw = np.eye(int(totalnum_points), dtype=np.complex_)
    for row in range(np.size(hcdw,0)):
        if row >= totalnum_points/2:
            hcdw[row,row] = -1

    nh_ham = new_h0 + alpha * np.matmul(hcdw, new_h0)

    return (nh_ham)

from Fundamental.Honeycomb_Lattice import bilayer_honeycomb_fermi_liquid_H0
def site_assignment_bilayer_honeycomb(nl):
    onelayer_ham = honeycomb_lattice(nl)
    onelayer_totnumpoints = honeycomb_points(nl)[1]
    asites, bsites = site_assignment_honeycomb(nl, onelayer_ham)
    asites_otherlayer = bsites + onelayer_totnumpoints
    bsites_otherlayer = asites + onelayer_totnumpoints
    return(asites, bsites, asites_otherlayer, bsites_otherlayer)

def NonHermitian_Bilayer_Honeycomb(nl, t_interlayer, alpha):
    ham_ogbasis = bilayer_honeycomb_fermi_liquid_H0(nl, t_interlayer)
    asites, bsites, asites_otherlayer, bsites_otherlayer = site_assignment_bilayer_honeycomb(nl)
    newbasis = np.concatenate((asites, bsites, asites_otherlayer, bsites_otherlayer))
    newbasis = np.array([int(i) for i in newbasis])
    ham_newbasis = ham_ogbasis[:,newbasis][newbasis,:]
    hcdw = np.eye(np.size(ham_ogbasis,0), np.size(ham_ogbasis,1))
    for i in np.arange(np.size(hcdw,0)/4, 3*np.size(hcdw,0)/4):
        hcdw[int(i),int(i)] = -1
    return(ham_newbasis + alpha*np.matmul(hcdw, ham_newbasis))

from Fundamental.Honeycomb_Lattice import bilayer_honeycomb_fermi_liquid_H0_PBC

def NonHermitian_Bilayer_Honeycomb_PBC(nl, t_interlayer, alpha):
    ham_ogbasis = bilayer_honeycomb_fermi_liquid_H0_PBC(nl, t_interlayer)
    asites, bsites, asites_otherlayer, bsites_otherlayer = site_assignment_bilayer_honeycomb(nl)
    newbasis = np.concatenate((asites, bsites, asites_otherlayer, bsites_otherlayer))
    newbasis = np.array([int(i) for i in newbasis])
    ham_newbasis = ham_ogbasis[:,newbasis][newbasis,:]
    hcdw = np.eye(np.size(ham_ogbasis,0), np.size(ham_ogbasis,1))
    for i in np.arange(np.size(hcdw,0)/4, 3*np.size(hcdw,0)/4):
        hcdw[int(i),int(i)] = -1
    return(ham_newbasis + alpha*np.matmul(hcdw, ham_newbasis))