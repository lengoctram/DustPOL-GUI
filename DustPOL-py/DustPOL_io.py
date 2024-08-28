import numpy as np
import os
from astropy import log, constants
# -------------------------------------------------------------------------
# Physical constants
# -------------------------------------------------------------------------
au  = constants.au.cgs.value
H   = 6.625e-27
C   = 3e10
K   = 1.38e-16
amu = 1.6605402e-24 #atomic mass unit
yr  = 365.*24.*60.*60.
pc  = constants.pc.cgs.value

#path= '/Users/lengoctram/Documents/postdoc/MPIfR/research/dustpol/src_v1.5/DustPOL-RAT+MRAT+2-layer_original/'#os.getcwd()
path = os.getcwd()+'/DustPOL-py/'

## -------------------------------------------------------------------------
## Experiment setup
##-------------------------------------------------------------------------
# from optparse import OptionParser
# parser = OptionParser()
# parser.add_option("-f", "--file",
#                   dest="file",
#                   help="file with data")
# (options, args) = parser.parse_args()

# inputs = options.file                 

inputs = path+'input.dustpol'
q = np.genfromtxt(inputs,skip_header=1,dtype=None,names=['names','params'],\
comments='!',usecols=(0,1),encoding='utf=8')
params     = q['params']
cloud      = params[0]
ratd       = params[1]
working_lam= eval(params[2])
rin        = eval(params[3]) * au #cm
rout       = eval(params[4]) * au #cm
rflat      = eval(params[5]) * au #cm
nsample    = eval(params[6])
U          = eval(params[7])
gamma      = eval(params[8])
mean_lam   = eval(params[9])*1e-4 #cm
dpc        = params[10]+'kpc'
ngas       = eval(params[11])
Tgas       = eval(params[12])
Avrange    = eval(params[13])
mgas       = 1.3*amu #90%H + 10%He
dust_type  = params[14]
amin       = eval(params[15])*1e-4 #cm
amax       = eval(params[16])*1e-4 #cm
Tdust      = eval(params[17])
rho        = eval(params[18])
alpha      = eval(params[19])
Smax       = eval(params[20])
dust_to_gas_ratio=eval(params[21])
GSD_law    = params[22]
power_index= eval(params[23])
RATalign = params[24].lower() # RAT or MRAT
f_max    = eval(params[25])
B_angle  = eval(params[27])*np.pi/180 # rad.
if RATalign=='mrat':
    Bfield = eval(params[26])
    Ncl    = eval(params[28])
    phi_sp = eval(params[29])
    fp     = eval(params[30])
else:
    Bfield = np.nan
    Ncl    = np.nan
    phi_sp = np.nan
    fp     = np.nan
# model_layer = eval(params[31])

# parallel   = eval(params[35])
# n_jobs     = eval(params[36])
# overwrite  = params[37]
# if overwrite=='No' or overwrite=='no':
#     checking=True
# else:
#     checking=False
u_ISRF = 8.64e-13 # typical interstellar radiation field

class output():
    def __init__(self,parent,filename,data):
        self.U=parent.U
        self.alpha=parent.alpha
        # subpath = path+'output/starless/astrodust/'#U=%.2f'%(self.U)+'/'
        # if not os.path.exists(subpath):
        #     os.mkdir(subpath)
        # subsubpath = subpath+'U=%.2f_alpha=%.4f'%(self.U,self.alpha)+'/Av_fixed_amax/'
        # if not os.path.exists(subsubpath):
        #     os.mkdir(subsubpath)

        subpath = path+'output/starless/astrodust/'+'U=%.2f_alpha=%.4f'%(self.U,self.alpha)+'/Av_fixed_amax/'
        if not os.path.exists(subpath):
            os.makedirs(subpath)
        self.filename=subpath+filename
        self.ngas=parent.ngas
        self.mean_lam=parent.mean_lam
        self.gamma=parent.gamma
        self.Av_array=parent.Av_array
        self.data=data

        # output_abs = path+'amax=%.2f'%(self.amax*1e4)+'_abs.dat'
        # output_emi = path+'amax=%.2f'%(self.amax*1e4)+'_emi.dat'

        self.file_save()

    def file_save(self):
        f=open(self.filename,'w')
        f.write('U=%.3f \n'%self.U)
        f.write('ngas=%.3e (cm-3) \n'%self.ngas)
        f.write('mean_lam=%.3f (um) \n'%(self.mean_lam*1e4))
        f.write('gamma=%.3f \n'%self.gamma)
        f.write('! \n')
        f.write('Av= ')
        f.write(",".join(str("{:.3f}".format(iAv)) for iAv in self.Av_array) + "\n")
        f.write('! \n')
        #keys=sorted(data_save.keys())
        keys=list(self.data.keys())
        print('   '.join(keys), end="\n",file=f)
        for i in range(len(self.data[keys[0]])):
            line=''
            for k in keys:
                # line=line+str(self.eformat(self.data[k][i],4,2))+'\t '
                line=line+str("{:.3e}".format(self.data[k][i]))+'\t '

            print(line,end="\n",file=f)
        f.close()

    def eformat(self,f, prec, exp_digits):
        s = "%.*e"%(prec, f)
        mantissa, exp = s.split('e')
        # add 1 to digits as 1 is taken by sign +/-
        return "%se%+0*d"%(mantissa, exp_digits+1, int(exp))
