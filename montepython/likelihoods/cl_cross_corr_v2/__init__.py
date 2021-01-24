from scipy.interpolate import interp1d
import os
import yaml
import itertools
import numpy as np
from montepython.likelihood_class import Likelihood
import montepython.io_mp as io_mp
import warnings
import pyccl as ccl
import shutil  # To copy the yml file to the outdir
import sacc



class cl_cross_corr_v2(Likelihood):

    # initialization routine

    def __init__(self, path, data, command_line):

        ##########
        # First read the YAML file and populate self.use_nuisance
        self.outdir = command_line.folder

        # Read arguments & copy the .yml file to output folder
        with open(os.path.abspath(data.cosmo_arguments['params_dir'])) as f:
            self.params = yaml.safe_load(f)
        shutil.copy2(os.path.abspath(data.cosmo_arguments['params_dir']),
                     self.outdir)

        # TODO: Make this clever
        self.use_nuisance = self.params['use_nuisance']

        # Initialize the Likelihood class
        Likelihood.__init__(self, path, data, command_line)
        ##########

        # Load sacc files
        self.scovG = self.load_sacc_file(self.params['sacc_covG'])

        # Save the data vector
        self.data = self.scovG.mean

        # Compute the full covmat
        # TODO: Remove after debugging
        self.cov = self.scovG.covariance.covmat
        # for trs1 in self.scovG.get_tracer_combinations():
        #     ix1 = self.scovG.indices(tracers=trs1)
        #     for trs2 in self.scovG.get_tracer_combinations():
        #         ix2 = self.scovG.indices(tracers=trs2)
        #         cov[ix1[0], ix2[0]] *= 10
        if ('sacc_covNG' in self.params) and os.path.isfile(self.params['sacc_covNG']):
            scovNG = self.load_sacc_file(self.params['sacc_covNG'])
            self.cov += scovNG.covariance.covmat

        # Invert covariance matrix
        self.icov = np.linalg.inv(self.cov)

        # Print vector size and dof
        npars = len(data.get_mcmc_parameters(['varying']))
        vecsize = self.data.size
        self.dof = vecsize - npars
        print('    -> Varied parameters = {}'.format(npars))
        print('    -> cl_cross_corr data vector size = {}'.format(vecsize))
        print('    -> cl_cross_corr dof = {}'.format(self.dof))

        # Save the ells considered for each Cl
        ells = np.array([])
        for tr0, tr1 in self.scovG.get_tracer_combinations():
            dtype = self.get_dtype_for_trs(tr0, tr1)
            ells = np.concatenate((ells, self.scovG.get_ell_cl(dtype, tr0, tr1)[0]))

        # Save a copy of the covmat, ells and cls for the tracer_combinations used for debugging
        np.savez_compressed(os.path.join(self.outdir, 'cl_cross_corr_data_info.npz'), cov=self.cov,
                            ells=ells, cls=self.data, tracers=self.scovG.get_tracer_combinations(), dof=self.dof)
        # end of initialization

    def load_sacc_file(self, sacc_file):
        print(f'Loading {sacc_file}')
        s = sacc.Sacc.load_fits(sacc_file)
        # Check used tracers are in the sacc file
        tracers_sacc = [trd for trd in s.tracers]
        for tr in self.params['tracers'].keys():
            if tr not in tracers_sacc:
                raise ValueError('The tracer {} is not present in {}'.format(tr, sacc_file))
        # TODO: Remove unused tracers
        # used_tracers = self.params['tracers'].keys()
        # Remove unused Cls
        s.remove_selection(data_type='cl_0b')
        s.remove_selection(data_type='cl_eb')
        s.remove_selection(data_type='cl_be')
        s.remove_selection(data_type='cl_bb')

        # Remove unused tracer combinations
        used_tracer_combinations = []
        for tracers_d in self.params['tracer_combinations']:
            tr1, tr2 = tracers_d['tracers']
            used_tracer_combinations.append((tr1, tr2))
            # Take into account that both are the same when comparing!
            used_tracer_combinations.append((tr2, tr1))

        used_tracer_combinations_sacc = []
        for tracers in s.get_tracer_combinations():
            if tracers not in used_tracer_combinations:
                s.remove_selection(tracers=tracers)
            used_tracer_combinations_sacc.append(tracers)

        # Cut scales below and above scale cuts for each tracer combination
        for tracers_d in self.params['tracer_combinations']:
                lmin, lmax = tracers_d['ell_cuts']
                tracers = tuple(tracers_d['tracers'])
                print(tracers, lmin, lmax)
                if tracers not in used_tracer_combinations_sacc:
                    # if not present is because they have the opposite ordering
                    tracers = tracers[::-1]
                s.remove_selection(ell__lt=lmin, tracers=tracers)
                s.remove_selection(ell__gt=lmax, tracers=tracers)
        print()

        return s

    def get_loggaussprior(self, value, center, var):
        lp = -0.5 * ((value - center) / var)**2.
        return lp

    def get_interpolated_cl(self, cosmo, ls, ccl_tracers, tr1, tr2, p_of_k_a=None):
        ls_nodes = np.unique(np.geomspace(2, ls[-1], 30).astype(int)).astype(float)
        # B.H. added the customized P(k,a)
        cls_nodes = ccl.angular_cl(cosmo.cosmo_ccl,
                                   ccl_tracers[tr1],
                                   ccl_tracers[tr2],
                                   ls_nodes,
                                   p_of_k_a=p_of_k_a)
        cli = interp1d(np.log(ls_nodes), cls_nodes,
                       fill_value=0, bounds_error=False)
        msk = ls >= 2
        cls = np.zeros(len(ls))
        cls[msk] = cli(np.log(ls[msk]))
        return cls

    def get_dtype_suffix_for_tr(self, tr):
        if ('gc' in tr) or ('cv' in tr):
            return '0'
        elif ('wl' in tr) or ('bin' in tr):
            return 'e'
        else:
            raise ValueError('dtype not found for tracer {}'.format(tr))


    def get_dtype_for_trs(self, tr0, tr1):
        dtype = 'cl_'
        if ('wl' in tr0) and ('cv' in tr1):
            dtype += self.get_dtype_suffix_for_tr(tr1)
            dtype += self.get_dtype_suffix_for_tr(tr0)
        else:
            dtype += self.get_dtype_suffix_for_tr(tr0)
            dtype += self.get_dtype_suffix_for_tr(tr1)

        return dtype
    
    # compute likelihood
    def loglkl(self, cosmo, data):

        # Initialize logprior
        lp = 0.

        # Initialize dictionary with ccl_tracers
        ccl_tracers = {}

        # B.H. Initialize biases for tracers
        bias_eft_tracers = {}
        
        # Get Tracers
        for trname, trvals in self.params['tracers'].items():
            stracer = self.scovG.get_tracer(trname)
            if 'cv' not in trname:
                z = stracer.z
                pz = stracer.nz

            if 'dz' in trvals:
                # Calculate z bias
                pname = '{}_dz_{}'.format(trvals['type'], trvals['bin'])
                dz = data.mcmc_parameters[pname]['current']*data.mcmc_parameters[pname]['scale']
                # Get log prior for dz
                lp = lp + self.get_loggaussprior(dz, *trvals['dz'])
                # Compute z - dz nd remove points with z_dz < 0:
                z_dz = z[z >= dz] - dz
                pz = pz[z >= dz]

            if trvals['type'] == 'gc':
                # og
                # Calculate bias
                #pname = 'gc_b_{}'.format(trvals['bin'])
                #bias = data.mcmc_parameters[pname]['current']*data.mcmc_parameters[pname]['scale']
                #bz = bias*np.ones(z.shape)

                # B.H. linear bias should be set to 1 
                bz = np.ones(z.shape)
                
                # create a dictionary with all bias parameters: 1, b1, b2, bn, bs
                bias_eft = {}
                for b in cosmo.bias_eft_names:
                    pname = 'gc_%s_%d'%(b, trvals['bin'])
                    bias_eft[b] = data.mcmc_parameters[pname]['current']*data.mcmc_parameters[pname]['scale']
                bias_eft['1'] = 1.
                
                # dictionary of dictionaries with EFT biases for each tracer
                bias_eft_tracers[trname] = bias_eft
                
                # Get tracer
                ccl_tracers[trname] = ccl.NumberCountsTracer(cosmo.cosmo_ccl, has_rsd=False, dndz=(z_dz, pz), bias=(z, bz))
                
            elif trvals['type'] == 'wl':
                # Get log prior for m
                pname = 'wl_m_{}'.format(trvals['bin'])
                value = data.mcmc_parameters[pname]['current']*data.mcmc_parameters[pname]['scale']
                lp = lp + self.get_loggaussprior(value, *trvals['m'])
                # Calculate bias IA
                A = data.mcmc_parameters['wl_ia_A']['current']*data.mcmc_parameters['wl_ia_A']['scale']
                eta = data.mcmc_parameters['wl_ia_eta']['current']*data.mcmc_parameters['wl_ia_eta']['scale']
                z0 = data.mcmc_parameters['wl_ia_z0']['current']*data.mcmc_parameters['wl_ia_z0']['scale']
                # intrinsic alignments bias
                bz = A*((1.+z)/(1.+z0))**eta*0.0139/0.013872474  # pyccl2 -> has already the factor inside. Only needed bz
                # Get tracer
                ccl_tracers[trname] = ccl.WeakLensingTracer(cosmo.cosmo_ccl, dndz=(z_dz,pz), ia_bias=(z,bz))
                
            elif trvals['type'] == 'cv':
                ccl_tracers[trname] = ccl.CMBLensingTracer(cosmo.cosmo_ccl, z_source=1100)#TODO: correct z_source
                
            else:
                raise ValueError('Type of tracer not recognized. It can be gc, wl or cv!')

        # Get theory Cls
        theory = np.zeros_like(self.data)
        for tr1, tr2 in self.scovG.get_tracer_combinations():
            # Get the indices for this part of the data vector
            dtype = self.get_dtype_for_trs(tr1, tr2)
            ind = self.scovG.indices(data_type=dtype, tracers=(tr1, tr2))  # not necessary because we have removed all b-correlations
            # Get the bandpower window function.
            # w.values contains the values of ell at which it is sampled
            w = self.scovG.get_bandpower_windows(ind)
            # Unbinned power spectrum.
            # pk2d object

            # B.H. information about the two tracers
            trvals1 = self.params['tracers'][tr1]
            trvals2 = self.params['tracers'][tr2]

            # fill the P(k,a) array with the power spectrum predictions
            Pk_a = np.zeros_like(cosmo.Pk_a_ij[cosmo.template_comb_names[0]])
            # if both tracers are galaxies, Pk^{tr1,tr2} = f_i^bin1 * f_j^bin2 * Pk_ij
            if trvals1['type'] == 'gc' and trvals2['type'] == 'gc':
                bias_eft1 = bias_eft_tracers[tr1]
                bias_eft2 = bias_eft_tracers[tr2]
                
                for key1 in bias_eft1.keys():
                    bias1 = bias_eft1[key1]
                    
                    for key2 in bias_eft2.keys():
                        bias2 = bias_eft2[key2]
                        
                        if key1+'_'+key2 in cosmo.Pk_a_ij.keys():
                            comb = key1+'_'+key2
                        else:
                            comb = key2+'_'+key1
                        Pk_a += bias1*bias2*cosmo.Pk_a_ij[comb]

            # if first tracer is galaxies, Pk^{tr1,tr2} = f_i^bin1 * 1. * Pk_0i
            elif trvals1['type'] == 'gc' and trvals2['type'] == 'wl':
                bias_eft1 = bias_eft_tracers[tr1]

                for key1 in bias_eft1.keys():
                    bias1 = bias_eft1[key1]
                    comb = '1'+'_'+key1
                    Pk_a += bias1*cosmo.Pk_a_ij[comb]

            # if second tracer is galaxies, Pk^{tr1,tr2} = f_j^bin2 * 1. * Pk_0j
            elif trvals1['type'] == 'wl' and trvals2['type'] == 'gc':
                bias_eft2 = bias_eft_tracers[tr2]

                for key2 in bias_eft2.keys():
                    bias2 = bias_eft2[key2]
                    comb = '1'+'_'+key2
                    Pk_a += bias2*cosmo.Pk_a_ij[comb]

            # if both tracers are weak lensing, Pk^{tr1,tr2} = 1. * 1. * Pk_00
            else:
                Pk_a = cosmo.Pk_a_ij['1_1']

            # create a 2D power spectrum object
            pk_tmp = ccl.Pk2D(a_arr=cosmo.a_arr, lk_arr=cosmo.Pk_a_ij['lk_arr'], pk_arr=Pk_a, is_logp=False)

            if self.params['interpolate_cls'] is True:
                # apparently interpolation makes the code run faster
                # B.H. adding our customized P(k,a) array
                cl_unbinned = self.get_interpolated_cl(cosmo, w.values, ccl_tracers, tr1, tr2, p_of_k_a=pk_tmp)
            else:
                # B.H. adding our customized P(k,a) array
                cl_unbinned = ccl.angular_cl(cosmo.cosmo_ccl, ccl_tracers[tr1], ccl_tracers[tr2], w.values, p_of_k_a=pk_tmp)

            # Convolved with window functions.
            cl_binned = np.dot(w.weight.T, cl_unbinned)
            for tr in [tr1, tr2]:
                trvals = self.params['tracers'][tr]
                if  trvals['type'] == 'wl':
                    pname = 'wl_m_{}'.format(trvals['bin'])
                    #  multiplicative bias
                    m = data.mcmc_parameters[pname]['current']*data.mcmc_parameters[pname]['scale']
                    cl_binned = (1.+m)*cl_binned

            # Assign to theory vector.
            theory[ind] = cl_binned

        # Get chi2
        chi2 = (self.data-theory).dot(self.icov).dot(self.data-theory)

        lkl = lp - 0.5 * chi2

        # print('chi2 =', chi2)
        # print('lp =', 2* lp)
        # np.savez_compressed(os.path.join(self.outdir, 'cl_cross_corr_bestfit_info.npz'), chi2_nolp=chi2, lp_chi2=2*lp, chi2=2*lkl, chi2dof=2*lkl/self.dof,
        #                     cls=theory)#, ells=self.ells_tosave, tracers=self.tracers_tosave)


        return lkl
