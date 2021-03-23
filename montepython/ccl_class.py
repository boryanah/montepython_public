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
        self.want_ratio = True
        # todo: move to param
        data_dir = os.path.expanduser("~/repos/hybrid_eft_nbody/data/AbacusSummit_base_c000_ph000/z0.100/")
        if self.want_ratio:
            fid_file = os.path.join(data_dir, "fid_rat_Pk_dPk_templates_%d.asdf"%(int(R_smooth)))
        else:
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

        # parameters used for the EFT approach: fid_deriv are used with the derivatives; other params are just to initialize CLASS
        deriv_param_names = ['omega_b', 'omega_cdm', 'n_s', 'sigma8_cb']
        
        # separate header items into cosmology parameters and redshift of the tempaltes
        fid_cosmo = {}
        fid_deriv_params = {}
        z_templates = {}
        for key in header.keys():
            if key in deriv_param_names:
                fid_cosmo[key] = header[key]
                fid_deriv_params[key] = header[key]
            elif 'ztmp' in key:
                z_templates[key] = header[key]
            elif 'A_s' == key:
                self.fid_A_s = header[key]
            elif 'theta_s_100' == key:
                theta_s_100 = header[key]
            else:
                fid_cosmo[key] = header[key]
        self.fid_deriv_params = fid_deriv_params
        self.z_templates = z_templates
        
        # remove the sigma8_cb parameter as CLASS uses A_s
        fid_cosmo.pop('sigma8_cb')
                
        # initilize  the fiducial cosmology to check that you recover the theta_s from the header
        class_cosmo = Class()
        class_cosmo.set(fid_cosmo)
        class_cosmo.compute()
        fid_theta = class_cosmo.theta_s_100()
        assert np.abs(header['theta_s_100'] - fid_theta) < 1.e-6, "CLASS not initialized properly" 
        
        # fiducial cosmology with all CLASS parameters
        fid_cosmo['100*theta_s'] = fid_theta
        self.fid_theta = fid_theta

        # removing h since the parameter that is held fixed is 100*theta_s
        fid_cosmo.pop('h')
        self.fid_cosmo = fid_cosmo
        
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
        # og
        #Omm = self.pars['Omega_c'] + self.pars['Omega_b']
        # B.H. we are working with the little omega's here
        Omm = (self.pars['omega_cdm'] + self.pars['omega_b'])/self.pars['h']**2
        return Omm

    def get_Omegac(self):
        # B.H.
        Omc = self.pars['omega_cdm']/self.pars['h']**2
        return Omc

    def get_Omegab(self):
        # B.H.
        Omb = self.pars['omega_b']/self.pars['h']**2
        return Omb

    def get_A_s(self):
        # B.H.
        A_s = (self.pars['sigma8_cb']/self.fid_deriv_params['sigma8_cb'])**2*self.fid_A_s
        return A_s
    
    def get_H0(self):
        # B.H.
        H0 = self.pars['h']*100.
        return H0

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
        if ('A_s' in pars_in[0].keys()) and ('sigma8_cb' in self.pars.keys()):
            self.pars.pop('sigma8_cb')
        if len(pars_in) == 1:
            self.pars.update(dict(pars_in[0]))
        elif len(pars_in) != 0:
            raise RuntimeError("bad call")
        ### Check for parmeters of cl_cross_corr lkl
        if 'params_dir' in self.pars.keys():
            del[self.pars['params_dir']]
        if 'fiducial_cov' in self.pars.keys():
            del[self.pars['fiducial_cov']]
        if 'tau_reio' in self.pars.keys():
            raise ValueError('CCL does not read tau_reio. Remove it.')
        # Translate w_0, w_a CLASS vars to CCL w0, wa
        if 'w_0' in self.pars.keys():
            self.pars['w0'] = self.pars.pop('w_0')
        if 'w_a' in self.pars.keys():
            self.pars['wa'] = self.pars.pop('w_a')
            
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

        '''

        # B.H. parameters for the ccl object
        param_dict = dict({'transfer_function': 'boltzmann_class'},
                          **self.pars)
        
        # We set h = 1 in param and then treat Omega_b and Omega_c as omega_b and omega_cdm
        sigma8_cb = param_dict['sigma8_cb']
        omega_cdm = param_dict['omega_cdm']
        omega_b = param_dict['omega_b']
        n_s = param_dict['n_s']
        # TESTING varying H0 (insert next line)
        #h = param_dict['h']
        A_s = self.get_A_s()

        # TESTING start
        '''
        h = 0.6736
        omega_b = self.fid_deriv_params['omega_b']#*0.99
        #omega_b = 0.0493*h**2
        #omega_cdm = 0.2640*h**2
        omega_cdm = self.fid_deriv_params['omega_cdm']#*1.01
        n_s = self.fid_deriv_params['n_s']#*0.95
        #sigma8_cb = 0.8111
        sigma8_cb = self.fid_deriv_params['sigma8_cb']#*1.01
        
        #omega_b = 1.823163e-02
        #omega_cdm = 1.038904e-01
        #sigma8_cb = 8.700650e-01
        #n_s = 1.019583e+00
        param_dict['sigma8_cb'] = sigma8_cb
        param_dict['n_s'] = n_s
        self.pars['sigma8_cb'] = sigma8_cb
        A_s = self.get_A_s()
        print(A_s)
        '''
        # TESTING end

        # updated dictionary without sigma8 because class takes only A_s and not sigma8
        updated_dict = {'A_s': A_s, 'omega_b': omega_b, 'omega_cdm': omega_cdm, 'n_s': n_s}
        
        # update the CLASS object with the current parameters
        # TESTING varying H0 (comment out)
        class_cosmo = Class()
        
        # update the cosmology
        new_cosmo = {**self.fid_cosmo, **updated_dict}
        # TESTING start
        '''
        new_cosmo['output'] = 'mPk'
        new_cosmo['z_max_pk'] = 1.1
        '''
        # TESTING end

        # TESTING varying H0  (comment out)
        class_cosmo.set(new_cosmo)
        class_cosmo.compute()
        
        # search for the corresponding value of H0 that keeps theta_s constant and update Omega_b and c
        # TESTING start
        #h = 0.6736
        # TESTING end
        # TESTING varying H0 (comment out)
        h = class_cosmo.h()
        #h = self.H0_search(class_cosmo, self.fid_theta, prec=1.e6, tol_t=1.e-6)
        
        # update the H0 value and the energy densities
        param_dict['h'] = h
        param_dict['Omega_c'] = omega_cdm/h**2
        param_dict['Omega_b'] = omega_b/h**2
        # I think that the normal setting is to use A_s here
        param_dict['A_s'] = A_s
        # I think that we don't want sigma8 for class and ccl but want sigma8 for the templates
        #param_dict['sigma8'] = param_dict.pop('sigma8_cb')

        # remove parameters not recognized by ccl
        param_not_ccl = ['100*theta_s', 'omega_cdm', 'omega_b', 'output', 'sigma8_cb']
        for p in param_not_ccl:
            if p in param_dict.keys(): param_dict.pop(p)
            
        # cosmology of the current step
        cosmo_ccl = ccl.Cosmology(**param_dict)
        self.cosmo_ccl = cosmo_ccl
        self.pars['h'] = h

        # interpolate for the cosmological parameters that are being deriv
        Pk_a_ij = {}
        a_arr = np.zeros(len(self.z_templates))
        # for a given redshift
        for combo in self.template_comb_names:
            Pk_a = np.zeros((len(self.z_templates), len(self.ks)))
            for i in range(len(self.z_templates)):
                z_str = 'ztmp%d'%i
                a_arr[i] = 1./(1+self.z_templates[z_str])
                key = z_str+'_'+combo
                # og fixed (no evolution)
                Pk = self.fid_dPk_Pk_templates[key] + \
                     self.fid_dPk_Pk_templates[key+'_'+'omega_b'] * (omega_b - self.fid_deriv_params['omega_b']) + \
                     self.fid_dPk_Pk_templates[key+'_'+'omega_cdm'] * (omega_cdm - self.fid_deriv_params['omega_cdm']) + \
                     self.fid_dPk_Pk_templates[key+'_'+'n_s'] * (n_s - self.fid_deriv_params['n_s']) + \
                     self.fid_dPk_Pk_templates[key+'_'+'sigma8_cb'] * (sigma8_cb - self.fid_deriv_params['sigma8_cb'])
                '''
                # TESTING fixed (no evolution)
                Pk = self.fid_dPk_Pk_templates[key]+0.
                '''
                if self.want_ratio:
                    Pk *= ccl.nonlin_matter_power(self.cosmo_ccl, self.ks*h, a=a_arr[i])
                else:
                    # convert to Mpc^3 rather than [Mpc/h]^3
                    Pk /= h**3.
                # TESTING using halofit instead
                #Pk = ccl.nonlin_matter_power(self.cosmo_ccl, self.ks*h, a=a_arr[i])
                Pk_a[i, :] = Pk
            Pk_a_ij[combo] = Pk_a
        # convert to Mpc^-1 rather than h/Mpc
        Pk_a_ij['lk_arr'] = np.log(self.ks*h)
        self.Pk_a_ij = Pk_a_ij
        self.a_arr = a_arr

        #self.cosmo_ccl.compute_nonlin_power()
        #ccl.sigma8(self.cosmo_ccl)  # David's suggestion

        # TESTING start
        '''
        for i in range(1):#(6):
            z_test = self.z_templates['ztmp%d'%i]
            NL = ccl.nonlin_matter_power(self.cosmo_ccl, self.ks*h, a=1./(1+z_test))
            print(z_test, h)
            
            #print(Pk_a_ij['1_1'][0,:], NL, z_test)
            np.save('lk_arr_%d.npy'%i, Pk_a_ij['lk_arr'])
            np.save('Pk_11_%d.npy'%i, Pk_a_ij['1_1'][i,:])
            class_ks = np.logspace(-5, np.log10(1), 1000)
            np.save("class_%d.npy"%i, np.array([class_cosmo.pk(ki, z_test) for ki in class_ks]))
            np.save("class_ks_%d.npy"%i, class_ks)
            np.save('NL_%d.npy'%i, NL)
        '''
        # TESTING end
        
        return

    def get_current_derived_parameters(self, names):

        derived = {}
        for name in names:
            if name == 'sigma_8':
                value = self.get_sigma8()
            elif name == 'Omega_m':
                value = self.get_Omegam()
            elif name == 'Omega_b':
                value = self.get_Omegab()
            elif name == 'Omega_c':
                value = self.get_Omegac()
            elif name == 'A_s':
                value = self.get_A_s()
            elif name == 'S_8':
                value = self.get_S8()
            elif name == 'H0':
                value = self.get_H0()
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
