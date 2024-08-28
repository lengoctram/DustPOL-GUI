##import Built-in functions
import numpy as np
import os,re
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import warnings
from astropy import log, constants
from matplotlib.colors import LogNorm
from joblib import Parallel, delayed#, Memory
from scipy.interpolate import interp1d
from astropy.io import fits
from scipy import interpolate

##import customized functions
from decorators import auto_refresh, printProgressBar, bcolors
import rad_func
import disrupt
import qq
import radiation
import align 
import size_distribution
import pol_degree
##Ignore the warnings
warnings.filterwarnings('ignore')

class DustPOL:
    """This is the main routine of the DustPOL-py
        -- update_radiation     : check and update for radiation
        -- update_grain_size    : check and update for grain size
        -- size_distribution    : calling size distribution
        -- get_Planck_function  : calling Planck function
        Inputs:
        -------
            These args are passed from the input file
                + U,mean_lam,gamma, ##radiation
                + Tgas,Tdust,ngas,  ##physical conditions
                + ratd,Smax,        ##rotational disruption
                + amin,amax,power_index,dust_type,dust_to_gas_ratio,GSD_law, ##grain-size
                + RATalign,f_min,f_max,alpha,   ##alignment physics
                + B_angle                       ##B-field inclination
        Outputs:
        --------
            Some sub-routines to call-out
                + cal_pol_abs   : to compute the degree of absorption polarisation
                + cal_pol_emi   : to compute the degree of emisison polarisation
                + starless_r0   : to compute the degree of absorption and emission polarisations
                                :            of starless core at a given line of sight (r0 -- x-coorinate)
                + starless      : to compute the degree of absorption and emission polarisations in starless core 

        Examples:
        ---------
        - Starless core with the line of sight Av fixed for different amax
                starless(Av_fixed_amax=True,fixed_amax_value=1.e-4)
        - Starless core with the line of sight Av varried for different amax
                starless(Av_fixed_amax=False)
    """

    @auto_refresh
    def __init__(self):
        from DustPOL_io import (
            U, u_ISRF, mean_lam, gamma, Tgas, Tdust, ngas,
            ratd, Smax, amin, amax, power_index, rho, dust_type, 
            dust_to_gas_ratio, GSD_law, RATalign, f_max,alpha,
            B_angle, Bfield, Ncl, phi_sp, fp, rflat, rout,nsample,
            path, pc
        )

        self.U = U             #No-unit
        self.u_ISRF=u_ISRF     #erg cm-3
        self.gamma=gamma       #No-unit
        self.mean_lam=mean_lam #cm
        self.Tgas=Tgas         #K
        self.Tdust=Tdust       #K
        self.ngas=ngas         #cm-3
        self.ratd=ratd         #[option]
        self.Smax=Smax         #erg cm-3
        self.amin=amin         #cm
        self.amax=amax         #cm
        self.power_index      =power_index       #No-unit
        self.dust_type        =dust_type         #[option]
        self.dust_to_gas_ratio=dust_to_gas_ratio #No-unit
        self.GSD_law          =GSD_law           #[option]
        self.RATalign         =RATalign          #[option]
        self.Bfield = Bfield                     #[MRAT] -- otherwise, nan
        self.Ncl    = Ncl                        #[MRAT] -- otherwise, nan
        self.phi_sp = phi_sp                     #[MRAT] -- otherwise, nan
        self.fp     = fp                         #[MRAT] -- otherwise, nan
        self.f_min  = 0.0      #%
        self.f_max  = f_max    #%
        self.alpha  = alpha    #No-unit
        self.B_angle=B_angle   #radiant
        self.rho    =rho       #g cm-3
        self.Urange_tempdist=[]

        ##test starless core
        # self.n0_gas=1e5
        self.rflat = rflat #cm #17000.*constants.au.cgs.value
        self.rout  = rout  #cm #624.e6*constants.au.cgs.value
        self.nsample= nsample #int 5#50

        # # ------- get constants -------
        self.pc = pc

        # # ------- get path to directory -------
        self.path=path

        # ------- Initialization wavelength, grain size, and cross-sections from the file -------
        if self.dust_type=='astro' or self.dust_type=='Astro':
            hdr_lines=4
            skip_lines=4
            len_a=169
            len_w=1129
            num_cols=8
            self.Data_aAstro=rad_func.readDC(self.path+'data/astrodust/Q_aAstro_%.3f'%(self.alpha)+'_P0.2_Fe0.00.DAT',hdr_lines,skip_lines,len_a,len_w,num_cols)

        else:
            #BELOW DOESN'T WORK FOR OBLATE SHAPE WITH S=2
            if float(self.alpha)==0.3333:
                hdr_lines = 4
                skip_lines=4
                len_a_sil=70
                len_a_car=100
                len_w=800
                num_cols=8
            elif float(self.alpha)==2.0:
                hdr_lines=4
                skip_lines=4
                len_a_sil=160
                len_a_car=160
                len_w=104
                num_cols=8
            else:
                log.error('Values of alpha is not regconized! [\033[1;5;7;91m failed \033[0m]')
            self.Data_sil = rad_func.readDC(self.path+'data/sil_car/Q_aSil2001_'+str(self.alpha)+'_p20B.DAT',hdr_lines,skip_lines,len_a_sil,len_w,num_cols)
            self.Data_mCBE = rad_func.readDC(self.path+'data/sil_car/Q_amCBE_'+str(self.alpha)+'.DAT',hdr_lines,skip_lines,len_a_car,len_w,num_cols)
        self.get_coefficients_files()
        
        # ------- Initialization grain-size distribution -------
        self.grain_size_distribution()  

    @auto_refresh
    def update_radiation(self):
        self.U = radiation.radiation_retrieve(self).retrieve()
        # self.U = rad.retrieve()

    @auto_refresh
    def update_grain_size(self,a):
        #update radiation
        #self.update_radiation()

        if self.amax>max(a):
            raise ValueError('SORRY - amax should be %.5f [um] at most [\033[1;5;7;91m failed \033[0m]'%(max(a)*1e4))

        self.lmin = np.searchsorted(a, self.amin)
        self.lmax = np.searchsorted(a, min(self.amax, disrupt.radiative_disruption(self).a_disrupt(a))) if self.ratd == 'on' else np.searchsorted(a, self.amax + 0.1 * self.amax)

        self.a = a[self.lmin:self.lmax]
        self.na = len(self.a)
        return

    @auto_refresh
    def dP_dT(self):
        ##This function reads the dust temperature distribution, pre-calculated
        qT = rad_func.T_dust(self.path, self.na, self.U)#T_dust(na,UINDEX)
        T_gra = qT[0]
        T_sil = qT[1]
        dP_dlnT_gra = qT[2]
        dP_dlnT_sil = qT[3]
        return T_sil,T_gra,dP_dlnT_sil,dP_dlnT_gra

    @auto_refresh
    def get_coefficients_files(self):
        ##This function reads the cross-sections, pre-calculated
        ##The outputs are 2d-array: a function of wavelength and grain-size
        if self.dust_type.lower()=='astro':
            self.w = self.Data_aAstro[1,:,0]*1e-4 ## wavelength in cm
            a = self.Data_aAstro[0,0,:]*1e-4      ## grain size in cm
            self.update_grain_size(a)             ## update grain size --> self.a
            [self.Qext_astro, self.Qabs_astro, self.Qpol_astro, self.Qpol_abs_astro] = qq.Qext_grain_astrodust(self.Data_aAstro,self.w,self.a,self.alpha)
            return
        else:
            self.w = rad_func.wave(self.path)     ##good for prolate shape
            a = rad_func.a_dust(self.path,10.0)[1] ##good for prolate shape
            self.update_grain_size(a)    ##update grain size
            
            if float(self.alpha)==2.0: ##efficiences data from POLARIS has a problem <-- fixed
                [self.Qext_sil, self.Qabs_sil, self.Qpol_sil, self.Qpol_abs_sil] = qq.Qext_grain(self.Data_sil,self.w,self.a,self.alpha,fixed=False,wmin=172e-4,wmax=628e-4,dtype='sil')        
                [self.Qext_amCBE, self.Qabs_amCBE, self.Qpol_amCBE, self.Qpol_abs_amCBE] = qq.Qext_grain(self.Data_mCBE,self.w,self.a,self.alpha,fixed=True,wmin=172e-4,wmax=300e-4,dtype='car')
            else:
                [self.Qext_sil, self.Qabs_sil, self.Qpol_sil, self.Qpol_abs_sil] = qq.Qext_grain(self.Data_sil,self.w,self.a,self.alpha)        
                [self.Qext_amCBE, self.Qabs_amCBE, self.Qpol_amCBE, self.Qpol_abs_amCBE] = qq.Qext_grain(self.Data_mCBE,self.w,self.a,self.alpha)
            return

    @auto_refresh
    def grain_size_distribution(self,fix_amax=False,fix_amax_value=None):
        ##This function compute the grain-size distribution
        ##The output is a 1d-array: a function of grain-size
        if self.dust_type.lower()=='astro':
            if not fix_amax:
                GSD_params = [self.a.min(),self.a.max(),2.74,self.dust_to_gas_ratio,self.power_index]
            else:
                GSD_params = [self.a.min(),fix_amax_value,2.74,self.dust_to_gas_ratio,self.power_index]
            self.dn_da_astro = size_distribution.dnda_astro(self.a,sizedist=self.GSD_law,MRN_params=GSD_params)
            return
        else:
            self.dn_da_gra = size_distribution.dnda(6,'carbon',self.a,self.GSD_law,self.power_index,self.dust_to_gas_ratio)
            self.dn_da_sil = size_distribution.dnda(6,'silicate',self.a,self.GSD_law,self.power_index,self.dust_to_gas_ratio)
            return

    @auto_refresh
    def get_Planck_function(self,Tdust,dP_dlnT=None):
        ##This function calculates the Planck-function
        ##The output is a 2d-array: a function of wavelength and grain-size
        if dP_dlnT is None:
            B_  = rad_func.planck_equi(self.w,self.na,Tdust) ##Tdust must be an array with 'na' element
        else:
            B_ = rad_func.planck(self.w,self.na,Tdust,dP_dlnT) ##function of U and na
        return B_

    @auto_refresh
    def extinction(self):
        ##This function return the extinction curve, normalized by Ngas
        if self.dust_type.lower()=='astro':
            dtau = self.Qext_astro * np.pi*self.a**2 * self.dn_da_astro
        elif self.dust_type.lower()=='sil':# in ['sil','silicate']:
            dtau = self.Qext_sil * np.pi*self.a**2 * self.dn_da_sil
        else:
            dtau_sil = self.Qext_sil * np.pi*self.a**2 * self.dn_da_sil
            dtau_car = self.Qext_amCBE * np.pi*self.a**2 * self.dn_da_gra
            dtau = dtau_sil + dtau_car
        
        tau_per_Ngas = integrate.simps(dtau,self.a)
        return 1.086*tau_per_Ngas

    @auto_refresh
    def cal_pol_abs(self,progress=False):
        ##This function calculates the degree of starlight polarization
        ##The output is a 1d-array: a function of wavelength
        self.progress=progress
        if not (self.progress):
            print('\t \033[1;7;34m U=%.3f \033[0m   \t\t '%(self.U))
        w,dP_abs_sil,dP_abs_mix = pol_degree.pol_degree(self)._pol_degree_absorption_(self)
        return w,dP_abs_sil,dP_abs_mix

    @auto_refresh
    def cal_pol_emi(self,tau=0.0,Tdust=None,progress=False):#,get_planck_option=True):
        ##This function calculates the degree of thermal dust polarization
        ##The output is a 1d-array: a function of wavelength
        ## If dust_type is Astrodust (astro):
        ##      return [total intensity, polarized intensity, zeros_array], [pol. degree, zeros_array]
        ## If dust_type is silicate (sil):
        ##      return [total intensity, polarized intensity of sil, polarized intensity of sil+car], [pol. degree of sil, pol. degree of sil+car]

        self.progress=progress

        if self.dust_type.lower()=='astro':
            if Tdust is None:
                Tdust = 16.4* self.U**(1./6) * (self.a/1.e-5)**(-1./15)#* np.ones(self.na)
                log.info('\033[1;7;34m U=%.3f : radiation -->> Tdust \033[0m   \t\t '%(self.U))
            elif isinstance(Tdust,(float,int)):
                log.info('\033[1;7;34m U=%.3f and Tdust=%.3f (K) \033[0m   \t\t '%(self.U,Tdust))
                Tdust = float(Tdust) * (self.a/1.e-5)**(-1./15)
            self.B_astro=self.get_Planck_function(Tdust)
        else:
            T_sil,T_gra,dP_dlnT_sil,dP_dlnT_gra = self.dP_dT() ##function of U and na
            self.B_gra=self.get_Planck_function(T_gra,dP_dlnT_gra)
            self.B_sil=self.get_Planck_function(T_sil,dP_dlnT_sil)

        w,I_list,P_list = pol_degree.pol_degree(self)._pol_degree_emission_(self,tau=tau)
        return w,I_list,P_list
