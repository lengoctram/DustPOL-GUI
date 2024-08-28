import streamlit as st
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from astropy.io import ascii
import importlib

# ----------------------------------------------------------------------------------------- #
# Execute DustPOL-py
# ----------------------------------------------------------------------------------------- #
@st.cache_data
def execute_DustPOL(U_rad, ngas, fmax, grain_type, grain_shape, amax, amin, rat_theory, Bfield, Ncl, p_plot_option):
    dir_dustpol = os.getcwd()+'/DustPOL-py/'
    sys.path.insert(1, dir_dustpol)
    
    input_file = dir_dustpol + 'input.dustpol'
    q = np.genfromtxt(input_file, skip_header=1, dtype=None, names=['names', 'params'],
                      comments='!', usecols=(0, 1), encoding='utf-8')
    
    # Update parameters
    param_updates = {
        7: U_rad,
        11: ngas,
        15: amin,
        16: amax,
        14: grain_type,
        19: grain_shape,
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
        return [w, p_list[0]] if composition_plot_option in ['Silicate', 'Astrodust'] else [w,p_list[1]]
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

st.sidebar.title('Visualization `DustPOL-py`')

# Header - plot option
st.sidebar.header('Degree of dust polarization')
p_plot_option = st.sidebar.selectbox('Please choose (starlight/emission polarizations)', ['Starlight Polarization', 'Thermal dust Polarization', 'Both']) 
st.sidebar.divider()

# Header - grain size
st.sidebar.header('Grain size')
col1, col2 = st.sidebar.columns(2)
amax = col1.selectbox('Maximum size (um):', [0.1, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0], index=9)
amin = col2.selectbox('Minimum size (um):', np.linspace(3.1e-4,amax+0.01*amax,20), index=0, format_func=lambda x: '{:.1e}'.format(x))
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

Bfield, Ncl = (np.nan, np.nan)
if rat_theory == 'MRAT':
    st.sidebar.subheader('B-field strength')
    Bfield = st.sidebar.number_input('B[$uG$]', value=600.0, step=10.0)
    
    st.sidebar.subheader('Number of iron cluster (Ncl)')
    Ncl = st.sidebar.slider('Specify Ncl', 10., 1.e5, 10., format='%.1e')

# Row A
st.markdown('### Parameters')
col1, col2, col3 = st.columns(3)
U_rads = col1.multiselect('Select radiation field', list(np.arange(0.,10.,1))+list(np.arange(10.,500.,20)))
ngass = np.array(col2.multiselect('Select gas volume density', [1e1,1e2,1e3,1e4,1e5,1e6,1e7], format_func=lambda x: '{:.1e}'.format(x)))
fmaxs = col3.multiselect('Select maximum alignment efficiency (fmax)', [0.25, 0.5, 1.0])

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
        ax1.set_xlabel('$\\rm wavelength\\,(um)$')
        ax1.set_ylabel('$\\rm p_{ext}/N_{H}\\,(\\%/cm^{-2})$')
        ax1.set_title('$\\sf Starlight\\,Polarization$',pad=20)
        ax11=ax1.twinx()
        ax11.set_ylabel('$\\rm A_{\\lambda}/N_{\\rm H}$')

    if p_plot_option in ['Thermal dust Polarization', 'Both']:
        fig2, ax2 = plt.subplots(figsize=(12, 3))
        ax2.set_xlabel('$\\rm wavelength\\,(um m)$')
        ax2.set_ylabel('$\\rm p_{em}\\,(\\%)$')
        ax2.set_title('$\\sf Thermal\\,Polarization$',pad=20)
        ax22=ax2.secondary_yaxis('right')
        ax22.set_ylabel('$\\rm p_{em}\\,(\\%)$')
        
    A_per_Ngas_pre=0.0
    for U_rad in U_rads:
        for n_gas in ngass: 
            for f_max in fmaxs:
                results = execute_DustPOL(U_rad, n_gas, f_max, grain_type, grain_shape, amax, amin, rat_theory, Bfield, Ncl,p_plot_option)
                if p_plot_option == 'Both':
                    w, pext, pem, A_per_Ngas = results
                    ax1.semilogx(w * 1e4, pext / n_gas, label=f'U={U_rad:.1f} -- n$_{{\\rm H}}$={n_gas:.1e} -- f$_{{\\rm max}}$={f_max:.2f}')
                    if np.sum(abs(A_per_Ngas/A_per_Ngas.max()-A_per_Ngas_pre))<=1e-19:
                        ax11.loglog(w * 1e4, A_per_Ngas,color='k',ls='--')
                    else:
                        ax11.loglog(w*1e-4,np.ones(len(w)),label='$\\sf pol.\\,spectrum$')
                        ax11.loglog(w * 1e4, A_per_Ngas,color='k',ls='--',label='$\\sf Extinction\\, curve$')
                    ax2.semilogx(w * 1e4, pem, label=f'U={U_rad:.1f} -- n$_{{\\rm H}}$={n_gas:.1e} -- f$_{{\\rm max}}$={f_max:.2f}')
                elif p_plot_option == 'Starlight Polarization':
                    w, pext,A_per_Ngas = results
                    ax1.semilogx(w * 1e4, pext / n_gas, label=f'U={U_rad:.1f} -- n$_{{\\rm H}}$={n_gas:.1e} -- f$_{{\\rm max}}$={f_max:.2f}')
                    if np.sum(abs(A_per_Ngas/A_per_Ngas.max()-A_per_Ngas_pre))<=1e-19:
                        ax11.loglog(w * 1e4, A_per_Ngas,color='k',ls='--')
                    else:
                        ax11.loglog(w*1e-4,np.ones(len(w)),label='$\\rm pol.\\,spectrum$')
                        ax11.loglog(w * 1e4, A_per_Ngas,color='k',ls='--',label='$\\rm Extinction\\, curve$')
                elif p_plot_option == 'Thermal dust Polarization':
                    w, pem = results
                    ax2.semilogx(w * 1e4, pem, label=f'U={U_rad:.1f} -- n$_{{\\rm H}}$={n_gas:.1e} -- f$_{{\\rm max}}$={f_max:.2f}')
                A_per_Ngas_pre = A_per_Ngas/A_per_Ngas.max()

    if ax1:
        ax1.legend(frameon=False)
        ax11.legend(bbox_to_anchor=(0.95,1.3))
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
Created with ❤️ by [Le N. Tram](The DustPOL-py is at: https://github.com/lengoctram/DustPOL-py).
''')
