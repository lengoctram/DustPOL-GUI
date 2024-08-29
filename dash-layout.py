import streamlit as st
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from astropy.io import ascii
import importlib
from scipy.signal import savgol_filter
# ----------------------------------------------------------------------------------------- #
# Execute DustPOL-py
# ----------------------------------------------------------------------------------------- #
@st.cache_data(persist="disk",max_entries=1000,show_spinner="Executing DustPOL-py ...")
def execute_DustPOL(U_rad, ngas, fmax, grain_type, grain_shape, amax, amin, rat_theory, ratd, Smax, Bfield, Ncl, p_plot_option):
    dir_dustpol = os.getcwd()+'/DustPOL-py/'
    sys.path.insert(1, dir_dustpol)
    
    input_file = dir_dustpol + 'input.dustpol'
    q = np.genfromtxt(input_file, skip_header=1, dtype=None, names=['names', 'params'],
                      comments='!', usecols=(0, 1), encoding='utf-8')
    
    # Update parameters
    param_updates = {
        1: ratd,
        7: U_rad,
        11: ngas,
        15: amin,
        16: amax,
        14: grain_type,
        19: grain_shape,
        20: Smax,
        24: rat_theory,
        25: fmax,
        26: Bfield,
        28: Ncl
    }
    for idx, value in param_updates.items():
        q['params'][idx]=value

    ascii.write(q, input_file, comment=True, overwrite=True)

    import DustPOL_class, align, DustPOL_io
    importlib.reload(DustPOL_io)

    args = DustPOL_class.DustPOL()
    A_per_Ngas=args.extinction()

    if p_plot_option == 'Starlight Polarization':
        w, psil, ptot = args.cal_pol_abs()
        return [w, psil, A_per_Ngas] if composition_plot_option in ['Silicate', 'Astrodust'] else [w, ptot, A_per_Ngas]
    elif p_plot_option == 'Thermal dust Polarization':
        w, I_list, p_list = args.cal_pol_emi()
        return [w, p_list[0], A_per_Ngas] if composition_plot_option in ['Silicate', 'Astrodust'] else [w,p_list[1],A_per_Ngas]
    else: #Both
        w, psil, ptot = args.cal_pol_abs()
        w, I_list, p_list = args.cal_pol_emi()
        return [w, psil, p_list[0], A_per_Ngas] if composition_plot_option in ['Silicate', 'Astrodust'] else [w, ptot, p_list[1], A_per_Ngas]

# ----------------------------------------------------------------------------------------- #
# Setup layout
# ----------------------------------------------------------------------------------------- #
st.set_page_config(page_title='Visualization DustPOL-py', page_icon=":shark:", layout='wide', initial_sidebar_state='expanded')

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

_,cen,_=st.sidebar.columns([1,3,1])
cen.image("dustpol-logo.png", use_column_width=True)

# st.sidebar.title('Visualization `DustPOL-py`')

# Header - plot option
st.sidebar.header('Degree of dust polarization')
p_plot_option = st.sidebar.selectbox('Please choose (starlight/emission polarizations)', ['Starlight Polarization', 'Thermal dust Polarization', 'Both']) 
st.sidebar.divider()

# Header - grain size
st.sidebar.header('Grain size')
col1, col2 = st.sidebar.columns(2)
amax = col1.selectbox('Maximum size ($\\mu$m):', [0.1, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0], index=9)
amin = col2.selectbox('Minimum size ($\\mu$m):', np.linspace(3.1e-4,amax+0.01*amax,20), index=0, format_func=lambda x: '{:.1e}'.format(x))
st.sidebar.divider()

# Header - grain composition
st.sidebar.header('Dust grains')
col1, col2 = st.sidebar.columns(2)
composition_plot_option = col1.selectbox('Dust composition', ('Silicate', 'Silicate+Carbon', 'Astrodust'), index=2)

grain_type_dict = {
    'Silicate': 'sil',
    'Silicate+Carbon': 'sil+car',
    'Astrodust': 'astro'
}
grain_type = grain_type_dict[composition_plot_option]

grain_shape_options = [0.3333] if composition_plot_option in ['Silicate', 'Silicate+Carbon'] else [1.4, 2.0, 3.0]
grain_shape = col2.selectbox('Grain shape', grain_shape_options)
st.sidebar.divider()

# Header - RAT theory
st.sidebar.header('Alignment Theory')
rat_theory = st.sidebar.selectbox('Select theory of RAT alignment', ('RAT', 'MRAT'))

# Sub-header -- MRAT
Bfield, Ncl = (np.nan, np.nan)
if rat_theory == 'MRAT':
    st.sidebar.subheader('B-field strength')
    Bfield = st.sidebar.number_input('B[$\\mu G$]', value=600.0, step=10.0)
    
    st.sidebar.subheader('Number of iron cluster (Ncl)')
    Ncl = st.sidebar.slider('Specify Ncl', 10., 1.e5, 10., format='%.1e')

# Sub-header -- RAT-D
ratd,Smax=(False,-1.e-99)
c1,c2=st.sidebar.columns(2)
with c1:
    ratd = c1.checkbox('RAT-D')
    if (ratd):
        Smax=c2.selectbox('Smax',[1e5,1e6,1e7,1e8,1e9,1e10],format_func=lambda x: '{:.1e}'.format(x),index=3)
c1,c2=st.sidebar.columns(2)
with c1.expander("Explanation"):
    st.write('''Turn on/off the rotational disruption -->> contrainning the $a_{\\rm max}$
    ''')
with c2.expander("Explanation"):
    st.write('''Maximum tensile strength of grain ($\\rm erg\\,cm^{-3}$) -- characterizing grain's porosity
    ''')
    
st.sidebar.divider()
button = st.sidebar.button("Clear All Caches")
if button:   
    st.cache_data.clear()
with st.sidebar.expander("explanation"):
    st.write('To clear all memory caches. Caches are on your disk and it is recommended to clear them all after a while!')

# Row A
st.markdown('### Parameters')
col1, col2, col3 = st.columns(3)
U_rads = col1.multiselect('Multiselect radiation field (U)', list(np.arange(0.,10.,1))+list(np.arange(10.,500.,20)))
with col1.expander("See explanation"):
    st.write('''
        $U=\\frac{\\int_{\\lambda}u_{\\lambda}d\\lambda}{8.64\\times 10^{-13}\\,\\rm erg\\,cm^{-3}}$
        with $u_{\\lambda}$ the radiation spectrum. For a typical aISRF, $U=1$.
    ''')

ngass = np.array(col2.multiselect('Multiselect gas volume density (ngas)', [1e1,1e2,1e3,1e4,1e5,1e6,1e7], format_func=lambda x: '{:.1e}'.format(x)))
with col2.expander("See explanation"):
    st.write('''
    n$_{\\rm gas}$ is in unit of $\\rm cm^{-3}$
    ''')
    
if rat_theory == 'RAT':
    fmaxs = col3.multiselect('Multiselect max. alignment efficiency (fmax)', [0.25, 0.5, 1.0])
else:
    fmaxs=[0.0]
    st.session_state.disable_opt = True
    col3.multiselect('Select maximum alignment efficiency (fmax)', [],disabled=st.session_state.disable_opt)
with col3.expander("See explanation"):
    st.write('''
    For MRAT theory, f$_{\\rm max}$ is estimated by the Bfield strength and Ncl
    ''')
        
st.divider()

def plot_figures():
    col_count = 10
    if p_plot_option == 'Both':
        c1, _ = st.columns((col_count, 1))
        c3, _ = st.columns((col_count, 1))
    else:
        c1, _ = st.columns((col_count, 1))
    
    fig1, ax1, fig2, ax2 = None, None, None, None
    if p_plot_option in ['Starlight Polarization', 'Both']:
        fig1, ax1 = plt.subplots(figsize=(12, 3))
        ax1.set_xlabel('$\\rm wavelength\\,(\\mu m)$')
        ax1.set_ylabel('$\\rm p_{ext}/N_{H}\\,(\\%/cm^{-2})$')
        ax1.set_title('$\\rm Starlight\\,Polarization$',pad=20)
        ax11=ax1.twinx()
        ax11.set_ylabel('$\\rm A_{\\lambda}/N_{\\rm H}$')

    if p_plot_option in ['Thermal dust Polarization', 'Both']:
        fig2, ax2 = plt.subplots(figsize=(12, 3))
        ax2.set_xlabel('$\\rm wavelength\\,(\\mu m)$')
        ax2.set_ylabel('$\\rm p_{em}\\,(\\%)$')
        ax2.set_title('$\\rm Thermal\\,Polarization$',pad=20)
        ax22=ax2.secondary_yaxis('right')
        ax22.set_ylabel('$\\rm p_{em}\\,(\\%)$')


    first=True
    ii=0
    for U_rad in U_rads:
        for n_gas in ngass: 
            for f_max in fmaxs:
                results = execute_DustPOL(U_rad, n_gas, f_max, grain_type, grain_shape, amax, amin, rat_theory, ratd, Smax, Bfield, Ncl,p_plot_option)
                if p_plot_option == 'Both':
                    w, pext, pem, A_per_Ngas = results
                    ax1.semilogx(w * 1e4, pext / n_gas, label=f'U={U_rad:.1f} -- n$_{{\\rm H}}$={n_gas:.1e} -- f$_{{\\rm max}}$={f_max:.2f}')
                    ax11.loglog(w * 1e4, A_per_Ngas,color='k',ls='--')
                 
                    if (ratd):
                        ax11.loglog(w * 1e4, A_per_Ngas,ls='--')
                    else:
                        ax11.loglog(w * 1e4, A_per_Ngas,color='k',ls='--')

                    if (first):
                        ax11.loglog(w*1e-4,np.ones(len(w)),color='k',ls='-',label='$\\rm pol.\\,spectrum$')
                        ax11.loglog(w*1e-4,np.ones(len(w)),color='k',ls='--',label='$\\rm Extinction\\, curve$')
                        
                    ax2.semilogx(w * 1e4, pem, label=f'U={U_rad:.1f} -- n$_{{\\rm H}}$={n_gas:.1e} -- f$_{{\\rm max}}$={f_max:.2f}')
                
                elif p_plot_option == 'Starlight Polarization':
                    w, pext,A_per_Ngas = results
                    ax1.semilogx(w * 1e4, pext / n_gas, label=f'U={U_rad:.1f} -- n$_{{\\rm H}}$={n_gas:.1e} -- f$_{{\\rm max}}$={f_max:.2f}')
                
                    if (ratd):
                        ax11.loglog(w * 1e4, A_per_Ngas,ls='--')
                    else:
                        ax11.loglog(w * 1e4, A_per_Ngas,color='k',ls='--')

                    if (first):
                        ax11.loglog(w*1e-4,np.ones(len(w)),color='k',ls='-',label='$\\rm pol.\\,spectrum$')
                        ax11.loglog(w*1e-4,np.ones(len(w)),color='k',ls='--',label='$\\rm Extinction\\, curve$')
                    ii+=1
                    bar = st.progress(ii)
                elif p_plot_option == 'Thermal dust Polarization':
                    w, pem, A_per_Ngas = results
                    ax2.semilogx(w * 1e4, pem, label=f'U={U_rad:.1f} -- n$_{{\\rm H}}$={n_gas:.1e} -- f$_{{\\rm max}}$={f_max:.2f}')
                first=False
                
    # A_per_Ngas_pre=0.0
    # for U_rad in U_rads:
    #     for n_gas in ngass: 
    #         for f_max in fmaxs:
    #             results = execute_DustPOL(U_rad, n_gas, f_max, grain_type, grain_shape, amax, amin, rat_theory, ratd, Smax, Bfield, Ncl,p_plot_option)
    #             if p_plot_option == 'Both':
    #                 w, pext, pem, A_per_Ngas = results
    #                 # smooth pext (for visualization) -- not physically affected
    #                 pext = savgol_filter(pext,20,2)
    #                 if rat_theory == 'RAT':
    #                     ax1.semilogx(w * 1e4, pext / n_gas, label=f'U={U_rad:.1f} -- n$_{{\\rm H}}$={n_gas:.1e} -- f$_{{\\rm max}}$={f_max:.2f}')
    #                 else:
    #                     ax1.semilogx(w * 1e4, pext / n_gas, label=f'U={U_rad:.1f} -- n$_{{\\rm H}}$={n_gas:.1e}')                        
    #                 if np.sum(abs(A_per_Ngas/A_per_Ngas.max()-A_per_Ngas_pre))<=1e-19:
    #                     ax11.loglog(w * 1e4, A_per_Ngas,color='k',ls='--')
    #                 else:
    #                     ax11.loglog(w*1e-4,np.ones(len(w)),label='pol. spectrum')
    #                     ax11.loglog(w * 1e4, A_per_Ngas,color='k',ls='--',label='Extinction curve')
    #                 if rat_theory == 'RAT':
    #                     ax2.semilogx(w * 1e4, pem, label=f'U={U_rad:.1f} -- n$_{{\\rm H}}$={n_gas:.1e} -- f$_{{\\rm max}}$={f_max:.2f}')
    #                 else:
    #                     ax2.semilogx(w * 1e4, pem, label=f'U={U_rad:.1f} -- n$_{{\\rm H}}$={n_gas:.1e}')
    #             elif p_plot_option == 'Starlight Polarization':
    #                 w, pext,A_per_Ngas = results
    #                 # smooth pext (for visualization) -- not physically effect
    #                 pext = savgol_filter(pext,20,2)
    #                 if rat_theory == 'RAT':
    #                     ax1.semilogx(w * 1e4, pext / n_gas, label=f'U={U_rad:.1f} -- n$_{{\\rm H}}$={n_gas:.1e} -- f$_{{\\rm max}}$={f_max:.2f}')
    #                 else:
    #                     ax1.semilogx(w * 1e4, pext / n_gas, label=f'U={U_rad:.1f} -- n$_{{\\rm H}}$={n_gas:.1e}')
    #                 if np.sum(abs(A_per_Ngas/A_per_Ngas.max()-A_per_Ngas_pre))<=1e-19:
    #                     ax11.loglog(w * 1e4, A_per_Ngas,color='k',ls='--')
    #                 else:
    #                     ax11.loglog(w*1e-4,np.ones(len(w)),label='$\\rm pol.\\,spectrum$')
    #                     ax11.loglog(w * 1e4, A_per_Ngas,color='k',ls='--',label='$\\rm Extinction\\, curve$')
    #             elif p_plot_option == 'Thermal dust Polarization':
    #                 w, pem, A_per_Ngas = results
    #                 if rat_theory == 'RAT':
    #                     ax2.semilogx(w * 1e4, pem, label=f'U={U_rad:.1f} -- n$_{{\\rm H}}$={n_gas:.1e} -- f$_{{\\rm max}}$={f_max:.2f}')
    #                 else:
    #                     ax2.semilogx(w * 1e4, pem, label=f'U={U_rad:.1f} -- n$_{{\\rm H}}$={n_gas:.1e}')
    #             A_per_Ngas_pre = A_per_Ngas/A_per_Ngas.max()

    if ax1:
        ax1.legend(frameon=False)
        ax11.legend(bbox_to_anchor=(0.95,1.35),frameon=False)
        ax11.set_ylim([1e-23,1e-20])
        ax1.set_xlim([0.05, 5e4])
        st.pyplot(fig1)
    if ax2:
        ax2.legend(frameon=False)
        ax2.set_xlim([0.05, 5e4])
        st.pyplot(fig2)

plot_figures()

st.sidebar.markdown('''
---
Model details: please refer to \\
https://ui.adsabs.harvard.edu/abs/2020ApJ...896...44L \\
https://ui.adsabs.harvard.edu/abs/2021ApJ...906..115T \\
https://arxiv.org/abs/2403.17088
''')
# st.sidebar.divider()
st.sidebar.markdown('''
---
Created with ❤️ by [Le N. Tram](The DustPOL-py is at: https://github.com/lengoctram/DustPOL-py).
''')
