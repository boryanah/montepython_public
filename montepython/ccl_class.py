import os

import numpy as np
import pyccl as ccl
import asdf
from classy import Class

class CCL():
    """
    General class for the CCL object.

    """

    def __init__(self):

        self.state = 1

        # initialized CCL parameters
        self.pars = {}

        # B.H. names of the bias EFT parameters
        bias_eft_names = ['b1', 'b2', 'bn', 'bs']
        self.bias_eft_names = bias_eft_names
        
        # load the fiducial template and the derivatives
        R_smooth = 0.
        # todo: move to param
        data_dir = os.path.expanduser("~/repos/hybrid_eft_nbody/data/AbacusSummit_base_c000_ph006/z0.100/")
        fid_file = os.path.join(data_dir, "fid_Pk_dPk_templates_%d.asdf"%(int(R_smooth)))
        with asdf.open(fid_file, lazy_load=False, copy_arrays=True) as f:
            self.fid_dPk_Pk_templates = f['data']
            self.ks = f['data']['ks']
            header = f['header']

        # get just the keys for the templates
        template_comb_names = []
        for i, bi in enumerate(bias_eft_names):
            for j, bj in enumerate(bias_eft_names):
                if j < i: continue
                template_comb_names.append(bi+'_'+bj)
        for bi in bias_eft_names:
            template_comb_names.append('1'+'_'+bi)
        template_comb_names.append('1'+'_'+'1')
        self.template_comb_names = template_comb_names

        
        # parameters used for the EFT approach: varied are used with the derivatives; fixed just to initialize CLASS; extra is magic to make fast and accurate
        varied_param_names = ['omega_b', 'omega_cdm', 'n_s', 'sigma8_cb']
        fixed_param_names = ['h', 'A_s', 'alpha_s', 'N_ur', 'N_ncdm', 'omega_ncdm'] # and w0 = -1, wa = 0 by def
        class_extra_param_names = ['T_cmb', 'Omega_dcdmdr', 'Omega_k', 'Omega_Lambda', 'Omega_scf', 'use_ppf', 'c_gamma_over_c_fld', 'cs2_fld', 'Omega_idm_dr', 'stat_f_idr', 'YHe', 'recombination', 'reio_parametrization', 'tau_reio', 'reionization_exponent', 'reionization_width', 'helium_fullreio_redshift', 'helium_fullreio_width', 'annihilation', 'decay', 'gauge', 'P_k_ini type', 'k_pivot', 'headers', 'format', 'write background', 'write thermodynamics', 'write primordial', 'input_verbose', 'background_verbose', 'thermodynamics_verbose', 'perturbations_verbose', 'transfer_verbose', 'primordial_verbose', 'spectra_verbose', 'nonlinear_verbose', 'lensing_verbose', 'output_verbose']
        
        # separate header items into cosmology parameters and redshift of the tempaltes
        fid_cosmo = {}
        fid_fixed_params = {}
        fid_varied_params = {}
        z_templates = {}
        for key in header.keys():
            if key in varied_param_names:
                fid_cosmo[key] = header[key]
                fid_varied_params[key] = header[key]
            elif key in fixed_param_names:
                fid_cosmo[key] = header[key]
                fid_fixed_params[key] = header[key]
            elif key in class_extra_param_names:
                fid_cosmo[key] = header[key]
            elif 'ztmp' in key:
                z_templates[key] = header[key]

        # initilize  the fiducial cosmology to get the 100_theta_s parameter
        class_cosmo = Class()
        # remove the sigma8_cb parameter as CLASS uses A_s
        fid_cosmo.pop('sigma8_cb')
        class_cosmo.set(fid_cosmo)
        class_cosmo.compute()
        fid_theta = class_cosmo.theta_s_100()
        print(fid_theta)
        assert np.abs(header['theta_s_100'] - fid_theta) < 1.e-6, "CLASS not initialized properly" 
        
        # fiducial cosmology with all CLASS parameters
        self.fid_cosmo = fid_cosmo
        self.fid_fixed_params = fid_fixed_params
        self.fid_varied_params = fid_varied_params
        self.fid_theta = header['theta_s_100']
        self.z_templates = z_templates
        
    def H0_search(self, new_cosmo, theta_def, h_ini=0.55, h_fin=0.85, prec=1.e5, tol_t=1.e-5):
        """
        Bisection search for the hubble parameter, h, given target and current cosmology
        """
        
        # array of hubble parameter values to search through
        hs = np.arange(h_ini*prec,h_fin*prec)/prec
        N_h = len(hs)

        # allowed tolerance b/n the new theta and the def
        this_theta = new_cosmo.theta_s_100()

        iterations = 0
        left = 0 # Determines the starting index of the list we have to search in
        right = N_h-1 # Determines the last index of the list we have to search in
        mid = (right + left)//2

        while(np.abs(this_theta-theta_def) > tol_t): # If this is not our search element
            # If the current middle element is less than x then move the left next to mid
            # Else we move right next to mid
            new_cosmo.set({'h':hs[mid]})
            new_cosmo.compute()
            this_theta = new_cosmo.theta_s_100()
            if  this_theta < theta_def:
                left = mid + 1
            else:
                right = mid - 1
            mid = (right + left)//2
            iterations += 1

            if right-left == 1: break

        # Final output
        print('iterations = ',str(iterations))
        print('h = ',new_cosmo.h())

        return new_cosmo.h()
    

    def get_cosmo_ccl(self):
        param_dict = dict({'transfer_function': 'boltzmann_class'},
                          **self.pars)
        try:
            param_dict.pop('output')
        except KeyError:
            pass
        cosmo_ccl = ccl.Cosmology(**param_dict)
        return cosmo_ccl

    def get_sigma8(self):
        return ccl.sigma8(self.cosmo_ccl)

    def get_Omegam(self):
        Omm = self.pars['Omega_c'] + self.pars['Omega_b']
        return Omm

    def get_S8(self):
        S8 = self.get_sigma8()*(self.get_Omegam()/0.3)**(0.5)
        return S8

    def get_growth_factor(self, a):
        return ccl.background.growth_factor(self.cosmo_ccl, a)

    def struct_cleanup(self):
        return

    def empty(self):
        return

    
    # Set up the dictionary
    def set(self, *pars_in, **kars):
        if ('A_s' in pars_in[0].keys()) and ('sigma8' in self.pars.keys()):
            self.pars.pop('sigma8')
        if len(pars_in) == 1:
            self.pars.update(dict(pars_in[0]))
        elif len(pars_in) != 0:
            raise RuntimeError("bad call")
        ### Check for parmeters of cl_cross_corr lkl
        if 'params_dir' in self.pars.keys():
            del[self.pars['params_dir']]
        if 'fiducial_cov' in self.pars.keys():
            del[self.pars['fiducial_cov']]
        #
        if 'tau_reio' in self.pars.keys():
            raise ValueError('CCL does not read tau_reio. Remove it.')
        # Translate w_0, w_a CLASS vars to CCL w0, wa
        if 'w_0' in self.pars.keys():
            self.pars['w0'] = self.pars.pop('w_0')
        if 'w_a' in self.pars.keys():
            self.pars['wa'] = self.pars.pop('w_a')
        if 'omega_cdm' in self.pars.keys():
            self.pars['Omega_c'] = self.pars.pop('omega_cdm')
        if 'omega_b' in self.pars.keys():
            self.pars['Omega_b'] = self.pars.pop('omega_b')
            
        self.pars.update(kars)
        return True

    
    def compute(self, level=[]):
        '''
        # B.H. not sure why the growth param thing
        # og
        self.cosmo_ccl = self.get_cosmo_ccl()
        # Modified growth part
        if 'growth_param' in self.pars:
            pk = ccl.boltzmann.get_class_pk_lin(self.cosmo_ccl)
            pknew = ccl.Pk2D(pkfunc=self.pk2D_new(pk), cosmo=self.cosmo_ccl,
                             is_logp=False)
            ccl.ccllib.cosmology_compute_linear_power(self.cosmo_ccl.cosmo,
                                                      pknew.psp, 0)

        # self.cosmo_ccl.compute_nonlin_power()
        ccl.sigma8(self.cosmo_ccl)  # David's suggestion
        '''

        # B.H. parameters for the ccl object
        param_dict = dict({'transfer_function': 'boltzmann_class'},
                          **self.pars)
        try:
            param_dict.pop('output')
        except KeyError:
            pass

        
        # We set h = 1 in param and then treat Omega_b and Omega_c as omega_b and omega_cdm
        sigma8_cb = param_dict['sigma8']
        omega_cdm = param_dict['Omega_c']#param_dict['omega_cdm']
        omega_b = param_dict['Omega_b']#param_dict['omega_b']
        n_s = param_dict['n_s']
        A_s = (param_dict['sigma8']/self.fid_varied_params['sigma8_cb'])**2*self.fid_fixed_params['A_s']
        # updated dictionary without sigma8 because class takes only A_s and not sigma8
        updated_dict = {'A_s': A_s, 'omega_b': omega_b, 'omega_cdm': omega_cdm, 'n_s': n_s}

        # update the CLASS object with the current parameters
        class_cosmo = Class()
        # remove the sigma8_cb parameter as CLASS uses A_s
        try:
            self.fid_cosmo.pop('sigma8_cb')
        except:
            pass
        # remove the Hubble parameter as CLASS can use theta_s_100
        try:
            self.fid_cosmo.pop('h')
        except:
            pass
        # TESTING
        # 100 theta can be passed earlier and output verbose seems useless
        #self.fid_cosmo['100*theta_s'] = self.fid_theta
        #self.fid_cosmo['output_verbose'] = 1
        print(self.fid_cosmo.items())
        print(updated_dict.items())
        
        # update the cosmology
        print(class_cosmo.h())
        # TESTING
        # create a big dic
        #self.fid_cosmo = {**self.fid_cosmo, **updated_dict}
        print(self.fid_cosmo.items())
        class_cosmo.set(self.fid_cosmo)
        #class_cosmo.set(updated_dict)
        class_cosmo.compute()
        
        # search for the corresponding value of H0 that keeps theta_s constant and update Omega_b and c
        # TESTING
        h = class_cosmo.h()
        print(h, class_cosmo.theta_s_100())
        # og
        import time
        t1 = time.time()
        h = self.H0_search(class_cosmo, self.fid_theta, prec=1.e4, tol_t=1.e-4)
        print(time.time()-t1, h, class_cosmo.theta_s_100())
        quit()
        param_dict['h'] = h
        param_dict['Omega_c'] = omega_cdm/h**2
        param_dict['Omega_b'] = omega_b/h**2
        try:
            param_dict.pop('100*theta_s')
        except:
            pass
            
        # cosmology of the current step
        cosmo_ccl = ccl.Cosmology(**param_dict)
        self.cosmo_ccl = cosmo_ccl
 
        # interpolate for the cosmological parameters that are being varied
        Pk_a_ij = {}
        a_arr = np.zeros(len(self.z_templates))
        # for a given redshift
        for combo in self.template_comb_names:
            Pk_a = np.zeros((len(self.z_templates), len(self.ks)))
            for i in range(len(self.z_templates)):
                z_str = 'ztmp%d'%i
                a_arr[i] = 1./(1+self.z_templates[z_str])
                key = z_str+'_'+combo
                Pk = self.fid_dPk_Pk_templates[key] + \
                     self.fid_dPk_Pk_templates[key+'_'+'omega_b'] * (omega_b - self.fid_varied_params['omega_b']) + \
                     self.fid_dPk_Pk_templates[key+'_'+'omega_cdm'] * (omega_cdm - self.fid_varied_params['omega_cdm']) + \
                     self.fid_dPk_Pk_templates[key+'_'+'n_s'] * (n_s - self.fid_varied_params['n_s']) + \
                     self.fid_dPk_Pk_templates[key+'_'+'sigma8_cb'] * (sigma8_cb - self.fid_varied_params['sigma8_cb'])
                # convert to Mpc^3 rather than [Mpc/h]^3
                Pk_a[i, :] = Pk/h**3.
            Pk_a_ij[combo] = Pk_a
        # convert to Mpc^-1 rather than h/Mpc
        Pk_a_ij['lk_arr'] = np.log(self.ks*h)
        self.Pk_a_ij = Pk_a_ij
        self.a_arr = a_arr
        return

    def get_current_derived_parameters(self, names):

        derived = {}
        for name in names:
            if name == 'sigma_8':
                value = self.get_sigma8()
            elif name == 'Omega_m':
                value = self.get_Omegam()
            elif name == 'S_8':
                value = self.get_S8()
            else:
                msg = "%s was not recognized as a derived parameter" % name
                raise RuntimeError(msg)
            derived[name] = value

        return derived

    def dpk(self, a):
        result = 0
        if self.pars['growth_param'] == 'linear':
            i = 0
            while True:
                pname = 'dpk' + str(i)
                if pname not in self.pars:
                    break
                dpki = self.pars[pname]
                result += dpki / np.math.factorial(i) * (1-a)**i
                i += 1
        return result

    # B.H. I think we don't care about this but not sure what it is
    def pk2D_new(self, pk):
        def pknew(k, a):
            return (1 + self.dpk(a)) ** 2 * pk.eval(k, a, self.cosmo_ccl)
        return pknew
