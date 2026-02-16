import numpy as np
from Fundamental.General_Hamiltonian import get_if_point_is_connected_with_upper_layer_q3_general
from Fundamental.General_Hamiltonian import get_points_that_connect_with_prev_layer_q3_general
from Fundamental.Honeycomb_Lattice import honeycomb_lattice
from Fundamental.Number_Points import points
from Fundamental.NonHermitian_Hamiltonian import site_assignment, site_assignment_honeycomb
from Fundamental.Hamiltonian import H0

def get_plaquet_boundary_sites_hyperbolicq3(p, num_levels):
    plaquet_boundary_sites = np.zeros((1, p))

    # Need to hard code first gen since "get_if_point_is_connected_with_upper_layer_q3_general" fails for first gen
    plaquet_boundary_sites = np.vstack((plaquet_boundary_sites, np.linspace(0, p-1, p)))

    points_on_level, tot_num_sites = points(p, 3, num_levels)
    points_connected_with_next_level = np.linspace(0, p - 1, p)  # Hard code for second generation to start loop
    for nl in range(1, num_levels):
        sites_on_gen = np.arange(np.sum(points_on_level[:nl]), np.sum(points_on_level[:nl + 1]))
        points_connect_with_prev_level = get_points_that_connect_with_prev_layer_q3_general(p, nl, sites_on_gen[0])
        for pcnl in range(len(points_connected_with_next_level) - 1):
            plaquet_boundary_sites = np.vstack((plaquet_boundary_sites, np.concatenate((np.flip(np.arange(points_connected_with_next_level[pcnl],
                                                                                                  points_connected_with_next_level[pcnl + 1] + 1)),
                                                                                        np.arange(points_connect_with_prev_level[pcnl],
                                                                                                  points_connect_with_prev_level[pcnl + 1] + 1)
                                                                                        ))
                                                                                        ))

        # Extra bit of code to account for last plaquet in each generation which is missed by above loop
        if nl != 1:  # Special since nl=1 only gen with last plaquet being a three not four sider
            plaquet_boundary_sites = np.vstack((plaquet_boundary_sites,
                                                np.concatenate((np.arange(points_connect_with_prev_level[-1],
                                                                          sites_on_gen[-1] + 1),
                                                                np.array([sites_on_gen[0]]),
                                                                np.array([points_connected_with_next_level[0],
                                                                          points_connected_with_next_level[0] - 1,
                                                                          points_connected_with_next_level[-1]]
                                                                          )
                                                                ))
                                                ))
        else:
            plaquet_boundary_sites = np.vstack((plaquet_boundary_sites,
                                                np.concatenate((np.array([points_connected_with_next_level[0],
                                                                          points_connected_with_next_level[-1]]),
                                                                np.arange(points_connect_with_prev_level[-1],
                                                                          sites_on_gen[-1] + 1),
                                                                np.array([sites_on_gen[0]])
                                                                ))
                                                ))

        points_connected_with_next_level = sites_on_gen[get_if_point_is_connected_with_upper_layer_q3_general(p, nl)]

    return (plaquet_boundary_sites[1:, :])

def get_nnn_hoppings_around_plaquet(plaquet_boundary_sites):
    nnn_hoppings_by_plaquet_isites = np.zeros(np.shape(plaquet_boundary_sites), dtype=int)
    nnn_hoppings_by_plaquet_jsites = np.zeros(np.shape(plaquet_boundary_sites), dtype=int)
    for row in range(np.size(plaquet_boundary_sites,0)):
        sites = plaquet_boundary_sites[row, :]
        plaquet_nnn_hops_isites = np.array([])
        plaquet_nnn_hops_jsites = np.array([])
        for s in range(len(sites)):
            plaquet_nnn_hops_isites = np.append(plaquet_nnn_hops_isites, np.take(sites, s, mode='wrap'))
            plaquet_nnn_hops_jsites = np.append(plaquet_nnn_hops_jsites, np.take(sites, s+2, mode='wrap'))
        nnn_hoppings_by_plaquet_isites[row, :] = plaquet_nnn_hops_isites
        nnn_hoppings_by_plaquet_jsites[row, :] = plaquet_nnn_hops_jsites
    return(nnn_hoppings_by_plaquet_isites, nnn_hoppings_by_plaquet_jsites)

def get_nnn_hoppings_around_plaquet_sublatticebasis_hyperbolicq3(p, num_levels, plaquet_boundary_sites):
    asites, bsites = site_assignment(p, 3, num_levels, H0(p, 3, num_levels))
    asites = [int(a) for a in asites]
    bsites = [int(b) for b in bsites]
    newbasis = np.concatenate((asites, bsites))
    nnn_hoppings_by_plaquet_isites, nnn_hoppings_by_plaquet_jsites = get_nnn_hoppings_around_plaquet(plaquet_boundary_sites)
    for row in range(np.size(nnn_hoppings_by_plaquet_isites, 0)):
        for col in range(np.size(nnn_hoppings_by_plaquet_isites, 1)):
            nnn_hoppings_by_plaquet_isites[row, col] = np.where(nnn_hoppings_by_plaquet_isites[row, col] == newbasis)[0]
            nnn_hoppings_by_plaquet_jsites[row, col] = np.where(nnn_hoppings_by_plaquet_jsites[row, col] == newbasis)[0]
    return(nnn_hoppings_by_plaquet_isites, nnn_hoppings_by_plaquet_jsites)

def calculate_haldane_current_around_plaquet_sublatticebasis(t2mat, nnn_hoppings_by_plaquet_isites, nnn_hoppings_by_plaquet_jsites):
    return(np.sum(t2mat[nnn_hoppings_by_plaquet_isites, nnn_hoppings_by_plaquet_jsites], axis=1))

from Axial_Magnetic_Field.honeycomb import connected_to_next_gen_points, connected_to_prev_gen_points, points_on_level

def get_plaquet_boundary_sites_honeycomb(num_levels):
    plaq_bound_sites = np.arange(0, 6, dtype=int) #hard code first generation
    for nl in range(1, num_levels):
        points_conn_next_gen = connected_to_next_gen_points(nl)
        points_conn_prev_gen = connected_to_prev_gen_points(nl + 1)
        for i in range(0, len(points_conn_next_gen)-1):
            bottom = np.flip(np.arange(points_conn_next_gen[i], points_conn_next_gen[i+1]+1, dtype=int))
            top = np.arange(points_conn_prev_gen[i], points_conn_prev_gen[i+1]+1, dtype=int)
            plaq_bound_sites = np.vstack((plaq_bound_sites, np.concatenate((bottom, top))))
        #Handle last plaquet on each level
        sites_on_level = points_on_level(nl).astype(int)
        sites_on_next_level = points_on_level(nl + 1).astype(int)
        bottom = np.flip(np.concatenate((sites_on_level[np.where(points_conn_next_gen[-1] == sites_on_level)[0][0]:],
                                 sites_on_level[:np.where(points_conn_next_gen[0] == sites_on_level)[0][0]+1])))
        top = np.concatenate((sites_on_next_level[np.where(points_conn_prev_gen[-1] == sites_on_next_level)[0][0]:],
                                 sites_on_next_level[:np.where(points_conn_prev_gen[0] == sites_on_next_level)[0][0]+1]))
        bottom = np.array([int(b) for b in bottom])
        top = np.array([int(t) for t in top])
        plaq_bound_sites = np.vstack((plaq_bound_sites, np.concatenate((bottom, top))))
    return(plaq_bound_sites)

def get_nnn_hoppings_around_plaquet_sublatticebasis_honeycomb(num_levels, plaquet_boundary_sites):
    asites, bsites = site_assignment_honeycomb(num_levels, honeycomb_lattice(num_levels))
    asites = [int(a) for a in asites]
    bsites = [int(b) for b in bsites]
    newbasis = np.concatenate((asites, bsites))
    nnn_hoppings_by_plaquet_isites, nnn_hoppings_by_plaquet_jsites = get_nnn_hoppings_around_plaquet(plaquet_boundary_sites)
    for row in range(np.size(nnn_hoppings_by_plaquet_isites, 0)):
        for col in range(np.size(nnn_hoppings_by_plaquet_isites, 1)):
            nnn_hoppings_by_plaquet_isites[row, col] = np.where(nnn_hoppings_by_plaquet_isites[row, col] == newbasis)[0]
            nnn_hoppings_by_plaquet_jsites[row, col] = np.where(nnn_hoppings_by_plaquet_jsites[row, col] == newbasis)[0]
    return(nnn_hoppings_by_plaquet_isites, nnn_hoppings_by_plaquet_jsites)

def calculate_haldane_current_around_plaquet_sublatticebasis_sublatticespecific(t2mat, nnn_hoppings_by_plaquet_isites, nnn_hoppings_by_plaquet_jsites):
    numsites = np.size(t2mat, 0)
    t2mat_asublattice_fluxs = []
    t2mat_bsublattice_fluxs = []
    for row in range(np.size(nnn_hoppings_by_plaquet_isites, 0)):
        rel_asites = np.where(nnn_hoppings_by_plaquet_isites[row, :] < int(numsites/2))[0]
        rel_bsites = np.where(nnn_hoppings_by_plaquet_isites[row, :] >= int(numsites/2))[0]
        t2mat_asublattice_fluxs.append(np.sum(t2mat[nnn_hoppings_by_plaquet_isites[row, :][rel_asites], nnn_hoppings_by_plaquet_jsites[row, :][rel_asites]]))
        t2mat_bsublattice_fluxs.append(np.sum(t2mat[nnn_hoppings_by_plaquet_isites[row, :][rel_bsites], nnn_hoppings_by_plaquet_jsites[row, :][rel_bsites]]))
    return(np.array(t2mat_asublattice_fluxs), np.array(t2mat_bsublattice_fluxs))

####################################

from Fundamental.Honeycomb_Lattice import honeycomb_points
from Fundamental.Honeycomb_Lattice import honeycomb_lattice_periodic_boundary
from Fundamental.NonHermitian_Hamiltonian import site_assignment_honeycomb

def get_plaquet_boundary_sites_honeycomb_PBC(num_levels):
    plaq_bound_sites = get_plaquet_boundary_sites_honeycomb(num_levels)
    lastgensites = np.arange(np.sum(honeycomb_points(num_levels)[0][:-1]), np.sum(honeycomb_points(num_levels)[1]), dtype=int)

    honeycomb_OBC = honeycomb_lattice(num_levels)
    honeycomb_PBC = honeycomb_lattice_periodic_boundary(num_levels)
    PBC_links = honeycomb_PBC - honeycomb_OBC

    for lgs in lastgensites:
        if len(np.where(PBC_links[lgs, :] != 0)[0]) == 0: #Meaning its a site that does not connect over the PBC
            currentindex = np.where(lgs == lastgensites)[0][0]
            oneupsite = np.take(lastgensites, currentindex + 1, mode='wrap')
            onedownsite = np.take(lastgensites, currentindex - 1, mode='wrap')
            oneuppbcconnsite = np.where(PBC_links[oneupsite, :] != 0)[0]
            onedownpbcconnsite = np.where(PBC_links[onedownsite, :] != 0)[0]
            oneuppbcconnindex = np.where(oneuppbcconnsite == lastgensites)[0][0]
            onedownpbcconnindex = np.where(onedownpbcconnsite == lastgensites)[0][0]
            if np.abs(oneuppbcconnindex - onedownpbcconnindex) == 2: #ie the simple case where dont have first-site last site problem
                pbcconninbetweenindex = (oneuppbcconnindex + onedownpbcconnindex) / 2
                pbcconninbetweensite = lastgensites[int(pbcconninbetweenindex)]
            else: #Exploit fact that depending on eveness/oddness of nl problem from first-last site wrap has trend
                if num_levels % 2 == 0:
                    pbcconninbetweensite = lastgensites[0]
                else:
                    pbcconninbetweensite = lastgensites[-1]

            newplaqbound = np.flip(np.array([lgs, oneupsite, oneuppbcconnsite[0], pbcconninbetweensite, onedownpbcconnsite[0], onedownsite]))
            plaq_bound_sites = np.vstack((plaq_bound_sites, newplaqbound))

    #Now last problem case are the corner points
    for lgsi in range(len(lastgensites)-1): #-1 to avoid error from first-last site which I can safely ignore since this is never the corner site
        #if statement to check if corner
        if len(np.where(PBC_links[lastgensites[lgsi], :] != 0)[0]) != 0 and len(np.where(PBC_links[lastgensites[lgsi+1], :] != 0)[0]) != 0:
            pbcconnsite1 = np.where(PBC_links[lastgensites[lgsi], :] != 0)[0][0]
            pbcconnsite2 = np.where(PBC_links[lastgensites[lgsi+1], :] != 0)[0][0]
            pbcconnsites = np.concatenate((np.array([pbcconnsite1]), np.array([pbcconnsite2])))
            if pbcconnsites[0] < pbcconnsites[1]:
                newplaqbound = np.flip(np.array([lastgensites[lgsi], lastgensites[lgsi+1], pbcconnsites[1], pbcconnsites[1] + 1, pbcconnsites[0] - 1, pbcconnsites[0]]))
            else:
                newplaqbound = np.flip(np.array([lastgensites[lgsi], lastgensites[lgsi + 1], pbcconnsites[1], pbcconnsites[1] + 1, pbcconnsites[0] - 1, pbcconnsites[0]]))

            plaq_bound_sites = np.vstack((plaq_bound_sites, newplaqbound))

    return (plaq_bound_sites)

def get_nnn_hoppings_around_plaquet_honeycomb_PBC(num_levels, plaq_bound_sites):
    isites = np.zeros(np.shape(plaq_bound_sites))
    jsites = np.zeros(np.shape(plaq_bound_sites))
    for row in range(np.size(plaq_bound_sites, 0)):
        relsites = plaq_bound_sites[row, :]
        for rsi in range(len(relsites)):
            isites[row, rsi] = relsites[rsi]
            jsites[row, rsi] = np.take(relsites, rsi+2, mode='wrap')
    return(isites, jsites)

def get_nnn_hoppings_around_plaquet_sublatticebasis_honeycomb_PBC(num_levels, plaq_bound_sites):
    isites_ogbasis, jsites_ogbasis = get_nnn_hoppings_around_plaquet_honeycomb_PBC(num_levels, plaq_bound_sites)
    asites, bsites = site_assignment_honeycomb(num_levels, honeycomb_lattice(num_levels))
    asites = np.array([int(a) for a in asites])
    bsites = np.array([int(b) for b in bsites])
    sublatticebasis = np.concatenate((asites, bsites))
    isites_sblbasis = np.zeros(np.shape(isites_ogbasis), dtype=int)
    jsites_sblbasis = np.zeros(np.shape(jsites_ogbasis), dtype=int)
    for row in range(np.size(isites_ogbasis, 0)):
        for col in range(np.size(isites_ogbasis, 1)):
            newentry = np.where(isites_ogbasis[row, col] == sublatticebasis)[0]
            isites_sblbasis[row, col] = newentry
            newentry = np.where(jsites_ogbasis[row, col] == sublatticebasis)[0]
            jsites_sblbasis[row, col] = newentry

    return(isites_sblbasis, jsites_sblbasis)




