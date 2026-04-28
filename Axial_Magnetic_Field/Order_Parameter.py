import numpy as np
from Fundamental.NonHermitian_Hamiltonian import site_assignment
from Fundamental.Hamiltonian import H0
from Fundamental.Number_Points import points

from Fundamental.Honeycomb_Lattice import honeycomb_points
from Fundamental.Honeycomb_Lattice import honeycomb_lattice
from Fundamental.NonHermitian_Hamiltonian import site_assignment_honeycomb

def get_gen_sites_sublattice_basis(p, num_levels):
    sites_on_gen = {}
    points_per_level, tot_num_points = points(p, 3, num_levels)
    asites, bsites = site_assignment(p, 3, num_levels, H0(p, 3, num_levels))
    newbasis = np.concatenate((asites, bsites))
    newbasis = np.array([int(i) for i in newbasis])
    for current_gen in range(1, num_levels+1):
        sites_on_level_indices = np.arange(np.sum(points_per_level[:current_gen-1]),
                                           np.sum(points_per_level[:current_gen]), dtype=int)
        sites_on_level_indices_newbasis = np.intersect1d(sites_on_level_indices, newbasis, return_indices=True)[2]
        new_entry = sites_on_level_indices_newbasis
        sites_on_gen['n={}'.format(current_gen)] = np.array([int(i) for i in new_entry])
    return(sites_on_gen)

def get_delta_sublattice_spinless_genlocalized(deltavals, p, numlevels):
    cdw_orders = np.array([])
    sites_on_gen = get_gen_sites_sublattice_basis(p, numlevels)
    tot_num_points = points(p, 3, numlevels)[1]
    for nl in range(1, numlevels + 1):
        relsites = sites_on_gen['n={}'.format(nl)]
        asites = deltavals[relsites[np.where(relsites < tot_num_points / 2)[0]]]
        bsites = deltavals[relsites[np.where(relsites >= tot_num_points / 2)[0]]]
        cdw_orders = np.append(cdw_orders, 0.5 * (np.abs(np.average(asites)) + np.abs(np.average(bsites))))
    return (cdw_orders)

def cdw_order_parameters_localgen(rawdatadir, p, num_gens):
    cdw_orders = np.zeros((1, num_gens+1))
    rawdata = np.load(rawdatadir)
    for row in range(np.size(rawdata, 0)):
        vval = rawdata[row, 0]
        deltavals = rawdata[row, 1:]
        cdw_vals = get_delta_sublattice_spinless_genlocalized(deltavals, p, num_gens)
        cdw_orders = np.vstack((cdw_orders, np.concatenate((np.array([vval]), cdw_vals))))
    return(cdw_orders[1:,:])

def get_gen_sites_sublattice_basis_spin_doubled(p, num_levels):
    sites_on_gen = {}
    points_per_level, tot_num_points = points(p, 3, num_levels)
    asites, bsites = site_assignment(p, 3, num_levels, H0(p, 3, num_levels))
    newbasis = np.concatenate((asites, bsites))
    newbasis = np.array([int(i) for i in newbasis])
    for current_gen in range(1, num_levels+1):
        sites_on_level_indices = np.arange(np.sum(points_per_level[:current_gen-1]),
                                           np.sum(points_per_level[:current_gen]), dtype=int)
        sites_on_level_indices_newbasis = np.intersect1d(sites_on_level_indices, newbasis, return_indices=True)[2]
        new_entry = np.concatenate((sites_on_level_indices_newbasis, sites_on_level_indices_newbasis + tot_num_points))
        sites_on_gen['n={}'.format(current_gen)] = np.array([int(i) for i in new_entry])
    return(sites_on_gen)

def get_delta_sublattice_spin_vals_genlocalized(deltavals, p, numlevels):
    afm_orders = np.array([])
    mag_orders = np.array([])
    sites_on_gen = get_gen_sites_sublattice_basis_spin_doubled(p, numlevels)
    tot_num_points =  points(p, 3, numlevels)[1]
    for nl in range(1, numlevels+1):
        relsites = sites_on_gen['n={}'.format(nl)]
        aupspin = deltavals[relsites[np.where((relsites < tot_num_points/2))[0]]]
        bupspin = deltavals[relsites[np.where((relsites >= tot_num_points/2) & (relsites < tot_num_points))[0]]]
        adownspin = deltavals[relsites[np.where((relsites >= tot_num_points) & (relsites < 3*tot_num_points/2))[0]]]
        bdownspin = deltavals[relsites[np.where((relsites >= 3*tot_num_points/2))[0]]]
        afm_orders = np.append(afm_orders, 0.5*(np.abs(np.average(aupspin)) + np.abs(np.average(bupspin)) +
                                                np.abs(np.average(adownspin)) + np.abs(np.average(bdownspin))))
        mag_orders = np.append(mag_orders, 0.5*(np.abs(np.average(aupspin)) - np.abs(np.average(bupspin)) +
                                                np.abs(np.average(adownspin)) - np.abs(np.average(bdownspin))))
    return(afm_orders, mag_orders)

def magnetic_order_parameters_localgen(rawdatadir, p, num_gens):
    afm_orders = np.zeros((1, num_gens+1))
    magnetization_orders = np.zeros((1, num_gens+1))
    rawdata = np.load(rawdatadir)
    for row in range(np.size(rawdata, 0)):
        uval = rawdata[row, 0]
        deltavals = rawdata[row, 1:]
        afm_vals, mag_vals = get_delta_sublattice_spin_vals_genlocalized(deltavals, p, num_gens)
        afm_orders = np.vstack((afm_orders, np.concatenate((np.array([uval]), afm_vals))))
        magnetization_orders = np.vstack((magnetization_orders, np.concatenate((np.array([uval]), mag_vals))))
    return(afm_orders[1:,:], magnetization_orders[1:,:])

def total_magnetization_calculate(rawdata):
    uvals = rawdata[:,0]
    # print(np.min(rawdata[:, 1:int(1+(np.size(rawdata,1)-1)/4)]), np.max(rawdata[:, 1:int(1+(np.size(rawdata,1)-1)/4)]))
    # print(np.min(rawdata[:, int(1+(np.size(rawdata,1)-1)/4):int(1+(np.size(rawdata,1)-1)/2)]), np.max(rawdata[:, int(1+(np.size(rawdata,1)-1)/4):int(1+(np.size(rawdata,1)-1)/2)]))
    # print(np.min(rawdata[:, int(1+(np.size(rawdata,1)-1)/2):int(1+(3*np.size(rawdata,1)-1)/4)]), np.max(rawdata[:, int(1+(np.size(rawdata,1)-1)/2):int(1+(3*np.size(rawdata,1)-1)/4)]))
    # print(np.min(rawdata[:, int(1+(3*np.size(rawdata,1)-1)/4):]), np.max(rawdata[:, int(1+(3*np.size(rawdata,1)-1)/4):]))
    adataupspin = np.average((rawdata[:, 1:int(1+(np.size(rawdata,1)-1)/4)]), axis=1)
    bdataupspin = np.average((rawdata[:, int(1+(np.size(rawdata,1)-1)/4):int(1+(np.size(rawdata,1)-1)/2)]), axis=1)
    adatadownspin = np.average((rawdata[:, int(1+(np.size(rawdata,1)-1)/2):int(1+3*(np.size(rawdata,1)-1)/4)]), axis=1)
    bdatadownspin = np.average((rawdata[:, int(1+3*(np.size(rawdata,1)-1)/4):]), axis=1)
    return(uvals, 0.5*(np.abs(adataupspin) + np.abs(adatadownspin) - np.abs(bdataupspin) - np.abs(bdatadownspin)))

def total_afm_calculate(rawdata):
    uvals = rawdata[:, 0]
    adataupspin = np.average((rawdata[:, 1:int(1 + (np.size(rawdata, 1) - 1) / 4)]), axis=1)
    bdataupspin = np.average((rawdata[:, int(1 + (np.size(rawdata, 1) - 1) / 4):int(1 + (np.size(rawdata, 1) - 1) / 2)]), axis=1)
    adatadownspin = np.average((rawdata[:, int(1 + (np.size(rawdata, 1) - 1) / 2):int(1 + 3*(np.size(rawdata, 1) - 1) / 4)]), axis=1)
    bdatadownspin = np.average((rawdata[:, int(1 + 3*(np.size(rawdata, 1) - 1) / 4):]), axis=1)
    return (uvals, 0.5 * (np.abs(adataupspin) + np.abs(adatadownspin) + np.abs(bdataupspin) + np.abs(bdatadownspin)))

def total_cdw_calculate(rawdata):
    vvals = rawdata[:, 0]
    adata = np.average((rawdata[:, 1:int(1 + (np.size(rawdata, 1) - 1) / 2)]), axis=1)
    bdata = np.average((rawdata[:, int(1 + (np.size(rawdata, 1) - 1) / 2):]), axis=1)
    return(vvals, 0.5 * (np.abs(adata) + np.abs(bdata)))

###########################################################################################################

def haldane_order_parameters_localgen(haldane_result, p ,num_levels):
    sites_on_gen = get_gen_sites_sublattice_basis(p, num_levels)
    t2vals_localgen = np.array([])
    for n in range(1, num_levels):
        currentgen_sites = sites_on_gen['n={}'.format(n)]
        nextgen_sites = sites_on_gen['n={}'.format(n+1)]

        currentgen_mask = np.isin(haldane_result[:2, :], currentgen_sites)
        nextgen_mask = np.isin(haldane_result[:2, :], nextgen_sites)
        combo_mask = currentgen_mask + nextgen_mask

        relhops = np.where(np.min(currentgen_mask, axis=0) == True)[0]
        # relhops = np.append(relhops, np.where(np.min(nextgen_mask, axis=0) == True)[0])
        relhops = np.append(relhops, np.where(np.min(combo_mask, axis=0) == True)[0])

        relhops = np.unique(relhops)
        t2vals_localgen = np.append(t2vals_localgen, np.average(np.abs(np.imag(haldane_result[2, relhops]))))

    currentgen_sites = sites_on_gen['n={}'.format(num_levels)]
    currentgen_mask = np.isin(haldane_result[:2, :], currentgen_sites)
    relhops = np.where(np.min(currentgen_mask, axis=0) == True)[0]
    t2vals_localgen = np.append(t2vals_localgen, np.average(np.abs(np.imag(haldane_result[2, relhops]))))

    return(t2vals_localgen)

# def haldane_order_parameters_localgen(haldane_result, p ,num_levels):
#     sites_on_gen = get_gen_sites_sublattice_basis(p, num_levels)
#     t2vals_localgen = np.array([])
#     for n in range(1, num_levels+1):
#         currentgen_sites = sites_on_gen['n={}'.format(n)]
#
#         currentgen_mask = np.isin(haldane_result[:2, :], currentgen_sites)
#
#         relhops = np.where(np.max(currentgen_mask, axis=0) == True)[0]
#
#         t2vals_localgen = np.append(t2vals_localgen, np.average(np.abs(np.imag(haldane_result[2, relhops]))))
#
#     return(t2vals_localgen)

#######################################################################################################################

def get_gen_sites_sublattice_basis_honeycomb(num_levels):
    sites_on_gen = {}
    points_per_level, tot_num_points = honeycomb_points(num_levels)
    asites, bsites = site_assignment_honeycomb(num_levels, honeycomb_lattice(num_levels))
    newbasis = np.concatenate((asites, bsites))
    newbasis = np.array([int(i) for i in newbasis])
    for current_gen in range(1, num_levels+1):
        sites_on_level_indices = np.arange(np.sum(points_per_level[:current_gen-1]),
                                           np.sum(points_per_level[:current_gen]), dtype=int)
        sites_on_level_indices_newbasis = np.intersect1d(sites_on_level_indices, newbasis, return_indices=True)[2]
        new_entry = sites_on_level_indices_newbasis
        sites_on_gen['n={}'.format(current_gen)] = np.array([int(i) for i in new_entry])
    return(sites_on_gen)

def get_delta_sublattice_spinless_genlocalized_honeycomb(deltavals, numlevels):
    cdw_orders = np.array([])
    sites_on_gen = get_gen_sites_sublattice_basis_honeycomb(numlevels)
    tot_num_points = honeycomb_points(numlevels)[1]
    for nl in range(1, numlevels + 1):
        relsites = sites_on_gen['n={}'.format(nl)]
        asites = deltavals[relsites[np.where(relsites < tot_num_points / 2)[0]]]
        bsites = deltavals[relsites[np.where(relsites >= tot_num_points / 2)[0]]]
        cdw_orders = np.append(cdw_orders, 0.5 * (np.abs(np.average(asites)) + np.abs(np.average(bsites))))
    return (cdw_orders)

def cdw_order_parameters_localgen_honeycomb(rawdatadir, num_gens):
    cdw_orders = np.zeros((1, num_gens+1))
    rawdata = np.load(rawdatadir)
    for row in range(np.size(rawdata, 0)):
        vval = rawdata[row, 0]
        deltavals = rawdata[row, 1:]
        cdw_vals = get_delta_sublattice_spinless_genlocalized_honeycomb(deltavals, num_gens)
        cdw_orders = np.vstack((cdw_orders, np.concatenate((np.array([vval]), cdw_vals))))
    return(cdw_orders[1:,:])

##############################################################################################################

def get_gen_sites_sublattice_basis_spin_doubled_honeycomb(num_levels):
    sites_on_gen = {}
    points_per_level, tot_num_points = honeycomb_points(num_levels)
    asites, bsites = site_assignment_honeycomb(num_levels, honeycomb_lattice(num_levels))
    newbasis = np.concatenate((asites, bsites))
    newbasis = np.array([int(i) for i in newbasis])
    for current_gen in range(1, num_levels+1):
        sites_on_level_indices = np.arange(np.sum(points_per_level[:current_gen-1]),
                                           np.sum(points_per_level[:current_gen]), dtype=int)
        sites_on_level_indices_newbasis = np.intersect1d(sites_on_level_indices, newbasis, return_indices=True)[2]
        new_entry = np.concatenate((sites_on_level_indices_newbasis, sites_on_level_indices_newbasis + tot_num_points))
        sites_on_gen['n={}'.format(current_gen)] = np.array([int(i) for i in new_entry])
    return(sites_on_gen)

def get_delta_sublattice_spin_vals_genlocalized_honeycomb(deltavals, numlevels):
    afm_orders = np.array([])
    mag_orders = np.array([])
    sites_on_gen = get_gen_sites_sublattice_basis_spin_doubled_honeycomb(numlevels)
    tot_num_points =  honeycomb_points(numlevels)[1]
    for nl in range(1, numlevels+1):
        relsites = sites_on_gen['n={}'.format(nl)]
        aupspin = deltavals[relsites[np.where((relsites < tot_num_points/2))[0]]]
        bupspin = deltavals[relsites[np.where((relsites >= tot_num_points/2) & (relsites < tot_num_points))[0]]]
        adownspin = deltavals[relsites[np.where((relsites >= tot_num_points) & (relsites < 3*tot_num_points/2))[0]]]
        bdownspin = deltavals[relsites[np.where((relsites >= 3*tot_num_points/2))[0]]]
        afm_orders = np.append(afm_orders, 0.5*(np.abs(np.average(aupspin)) + np.abs(np.average(bupspin)) +
                                                np.abs(np.average(adownspin)) + np.abs(np.average(bdownspin))))
        mag_orders = np.append(mag_orders, 0.5*(np.abs(np.average(aupspin)) - np.abs(np.average(bupspin)) +
                                                np.abs(np.average(adownspin)) - np.abs(np.average(bdownspin))))
    return(afm_orders, mag_orders)

def magnetic_order_parameters_localgen_honeycomb(rawdatadir, num_gens):
    afm_orders = np.zeros((1, num_gens+1))
    magnetization_orders = np.zeros((1, num_gens+1))
    rawdata = np.load(rawdatadir)
    for row in range(np.size(rawdata, 0)):
        uval = rawdata[row, 0]
        deltavals = rawdata[row, 1:]
        afm_vals, mag_vals = get_delta_sublattice_spin_vals_genlocalized_honeycomb(deltavals, num_gens)
        afm_orders = np.vstack((afm_orders, np.concatenate((np.array([uval]), afm_vals))))
        magnetization_orders = np.vstack((magnetization_orders, np.concatenate((np.array([uval]), mag_vals))))
    return(afm_orders[1:,:], magnetization_orders[1:,:])

######################################################################################################################

def haldane_order_parameters_localgen_honeycomb(haldane_result, num_levels):
    sites_on_gen = get_gen_sites_sublattice_basis_honeycomb(num_levels)
    t2vals_localgen = np.array([])
    for n in range(1, num_levels):
        currentgen_sites = sites_on_gen['n={}'.format(n)]
        nextgen_sites = sites_on_gen['n={}'.format(n+1)]

        currentgen_mask = np.isin(haldane_result[:2, :], currentgen_sites)
        nextgen_mask = np.isin(haldane_result[:2, :], nextgen_sites)
        combo_mask = currentgen_mask + nextgen_mask

        relhops = np.where(np.min(currentgen_mask, axis=0) == True)[0]
        # relhops = np.append(relhops, np.where(np.min(nextgen_mask, axis=0) == True)[0])
        relhops = np.append(relhops, np.where(np.min(combo_mask, axis=0) == True)[0])

        relhops = np.unique(relhops)
        t2vals_localgen = np.append(t2vals_localgen, np.average(np.abs(np.imag(haldane_result[2, relhops]))))

    currentgen_sites = sites_on_gen['n={}'.format(num_levels)]
    currentgen_mask = np.isin(haldane_result[:2, :], currentgen_sites)
    relhops = np.where(np.min(currentgen_mask, axis=0) == True)[0]
    t2vals_localgen = np.append(t2vals_localgen, np.average(np.abs(np.imag(haldane_result[2, relhops]))))

    return(t2vals_localgen)
