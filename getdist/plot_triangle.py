import os
import numpy as np
import getdist
from getdist import plots, MCSamples
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

chain_name = 'vary_b1_true'
path2MP = '/users/boryanah/repos/montepython_public/'
path2sh = os.path.join(path2MP, 'chains/'+chain_name)
chain_sh_all = getdist.loadMCSamples(os.path.join(path2sh, '2021-01-26_1000000_'),
                           settings={'ignore_rows':0.7})

chain_sh_all.removeBurn(0.7)
#chain_sh_all.addDerived(p.S_8 / np.sqrt(p.Omega_m/0.3),
#                     'sigma_8', label='\sigma_8')
chain_sh_data_all = np.load(os.path.join(path2sh, 'cl_cross_corr_data_info.npz' ))

'''
g = plots.get_subplot_plotter(width_inch=5)
g.plot_2d(chain_sh_all, 'omega_cdm', 'omega_b', filled=True)
plt.savefig("try.png")
plt.close()
#g.finish_plot()
'''
der_dic = {'Omega_m': 0.3133, 'Omega_c': 0.2640, 'Omega_b': 0.0493, 'H0': 67.36}
cosmo_dic = {'omega_b': der_dic['Omega_b']*(der_dic['H0']/100.)**2, 'omega_cdm': der_dic['Omega_c']*(der_dic['H0']/100.)**2, 'n_s': 0.9649, 'sigma8_cb': 0.8111}


params = ['omega_b', 'omega_cdm', 'sigma8_cb', 'n_s', 'gc_b1_0', 'gc_b1_1', 'gc_b1_2', 'gc_b1_3', 'gc_b1_4']
params_derived = ['A_s', 'Omega_m', 'S_8', 'Omega_c', 'Omega_b', 'H0']



g = plots.get_subplot_plotter(width_inch=12)
g.triangle_plot([chain_sh_all], params=params, filled=True)

lw = 1.5
for i,key_i in enumerate(cosmo_dic.keys()):
    for j,key_j in enumerate(cosmo_dic.keys()):
        if j == i:
            print("i,j = ",i,j)
            ax = g.get_axes_for_params(key_i)
            print(ax)
            g.add_x_marker(marker=cosmo_dic[key_i], color='blue', ax=ax, lw=lw, ls='--')
            ax = None
        else:#if j > i:
            print("i,j = ",i,j)
            ax = g.get_axes((key_j,key_i))
            if ax is not None:
                ax.scatter(np.array(cosmo_dic[key_j]), np.array(cosmo_dic[key_i]), color='blue', s=18., marker='x')
            print(ax)
            #g.add_y_marker(marker=cosmo_dic[key_i],color='blue',ax=ax,lw=lw,ls='--')
            #g.add_x_marker(marker=cosmo_dic[key_j],color='blue',ax=ax,lw=lw,ls='--')
            ax = None
            
plt.savefig("params.png")
plt.close()
#g.finish_plot()

g = plots.get_subplot_plotter(width_inch=12)
g.triangle_plot([chain_sh_all], params=params_derived, filled=True)
for i,key_i in enumerate(der_dic.keys()):
    for j,key_j in enumerate(der_dic.keys()):
        if j == i:
            print("i,j = ",i,j)
            ax = g.get_axes_for_params(key_i)
            print(ax)
            g.add_x_marker(marker=der_dic[key_i], color='blue', ax=ax, lw=lw, ls='--')
            ax = None
        else:#if j > i:
            print("i,j = ",i,j)
            ax = g.get_axes((key_j,key_i))
            if ax is not None:
                ax.scatter(np.array(der_dic[key_j]), np.array(der_dic[key_i]), color='blue', s=18., marker='x')
            print(ax)
            #g.add_y_marker(marker=der_dic[key_i],color='blue',ax=ax,lw=lw,ls='--')
            #g.add_x_marker(marker=der_dic[key_j],color='blue',ax=ax,lw=lw,ls='--')
            ax = None
plt.savefig("params_derived.png")
plt.close()
#g.finish_plot()

'''
omega_b          \omega_{b }
omega_cdm        \omega_{cdm }
sigma8   \sigma8
n_s      n_{s }
gc_b1_0          gc_{b1 0 }
gc_b1_1          gc_{b1 1 }
gc_b1_2          gc_{b1 2 }
gc_b1_3          gc_{b1 3 }
gc_b1_4          gc_{b1 4 }
gc_b2_0          gc_{b2 0 }
gc_b2_1          gc_{b2 1 }
gc_b2_2          gc_{b2 2 }
gc_b2_3          gc_{b2 3 }
gc_b2_4          gc_{b2 4 }
gc_bn_0          gc_{bn 0 }
gc_bn_1          gc_{bn 1 }
gc_bn_2          gc_{bn 2 }
gc_bn_3          gc_{bn 3 }
gc_bn_4          gc_{bn 4 }
gc_bs_0          gc_{bs 0 }
gc_bs_1          gc_{bs 1 }
gc_bs_2          gc_{bs 2 }
gc_bs_3          gc_{bs 3 }
gc_bs_4          gc_{bs 4 }
gc_dz_0          gc_{dz 0 }
gc_dz_1          gc_{dz 1 }
gc_dz_2          gc_{dz 2 }
gc_dz_3          gc_{dz 3 }
gc_dz_4          gc_{dz 4 }
wl_ia_A          wl_{ia A }
wl_ia_eta        wl_{ia eta }
wl_m_0   wl_{m 0 }
wl_m_1   wl_{m 1 }
wl_m_2   wl_{m 2 }
wl_m_3   wl_{m 3 }
wl_dz_0          wl_{dz 0 }
wl_dz_1          wl_{dz 1 }
wl_dz_2          wl_{dz 2 }
wl_dz_3          wl_{dz 3 }
Omega_m          \Omega_{m }
S_8      S_{8 }
'''
