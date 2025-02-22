import streamlit as st
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from astropy.io import ascii
import importlib
import pandas as pd
from scipy.signal import savgol_filter
# ----------------------------------------------------------------------------------------- #
# Execute DustPOL-py
# ----------------------------------------------------------------------------------------- #
@st.cache_data(persist="disk",max_entries=100,show_spinner="Executing DustPOL-py ...")
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
# cen.image("dustpol-logo.png", use_column_width=True)
cen.image("dustpol-logo.png", use_container_width=True)

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
if (ratd):
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
U_rads = col1.multiselect('Multiselect radiation field (U)', list(np.around(np.arange(0.1,1,0.1),2))+list(np.arange(1,10.,1))+list(np.arange(10.,500.,20))+list(np.arange(500.,1020.,20)))
with col1.expander("See explanation"):
    st.write('''
        $U=\\frac{\\int_{\\lambda}u_{\\lambda}d\\lambda}{8.64\\times 10^{-13}\\,\\rm erg\\,cm^{-3}}$
        with $u_{\\lambda}$ the radiation spectrum. For a typical interstellar medium, $U=1$.
    ''')

ngass = np.array(col2.multiselect('Multiselect gas volume density (ngas)', [1e1,1e2,1e3,1e4,1e5,1e6,1e7], format_func=lambda x: '{:.1e}'.format(x)))
with col2.expander("See explanation"):
    st.write('''
    n$_{\\rm gas}=n(H) + 2n(H2)+...$ is in unit of $\\rm cm^{-3}$
    ''')
    
if rat_theory == 'RAT':
    fmaxs = col3.multiselect('Multiselect max. alignment efficiency (fmax)', [0.25, 0.5, 1.0])
else:
    fmaxs=[0.0]
    st.session_state.disable_opt = True
    col3.multiselect('Select maximum alignment efficiency (fmax)', [],disabled=st.session_state.disable_opt)
with col3.expander("See explanation"):
    st.write('''
    $f_{\\rm max}$ is the maximum alignment efficiency. For MRAT theory, $f_{\\rm max}$ is estimated by the input values of Bfield strength and Ncl
    ''')
        
st.divider()

##declearing output file
output_abs={}
output_emi={}
output_ext={}

def plot_figures():
    col_count = 8
    if p_plot_option == 'Both':
        c1, _ = st.columns((col_count, 1))
        c3, _ = st.columns((col_count, 1))
    else:
        c1, _ = st.columns((col_count, 1))
    
    fig1, ax1, fig2, ax2 = None, None, None, None
    if p_plot_option in ['Starlight Polarization', 'Both']:
        fig1, ax1 = plt.subplots(figsize=(10, 3))
        ax1.set_xlabel('$\\rm wavelength\\,(\\mu m)$')
        ax1.set_ylabel('$\\rm p_{ext}/N_{H}\\,(\\%/cm^{-2})$')
        ax1.set_title('$\\rm Starlight\\,Polarization$',pad=20)
        ax11=ax1.twinx()
        ax11.set_ylabel('$\\rm A_{\\lambda}/N_{\\rm H}$')

    if p_plot_option in ['Thermal dust Polarization', 'Both']:
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.set_xlabel('$\\rm wavelength\\,(\\mu m)$')
        ax2.set_ylabel('$\\rm p_{em}\\,(\\%)$')
        ax2.set_title('$\\rm Thermal\\,Polarization$',pad=20)
        ax22=ax2.secondary_yaxis('right')
        ax22.set_ylabel('$\\rm p_{em}\\,(\\%)$')


    first=True
    for U_rad in U_rads:
        for n_gas in ngass: 
            for f_max in fmaxs:
                results = execute_DustPOL(U_rad, n_gas, f_max, grain_type, grain_shape, amax, amin, rat_theory, ratd, Smax, Bfield, Ncl,p_plot_option)
                if p_plot_option == 'Both':
                    w, pext, pem, A_per_Ngas = results
                    
                    pext = savgol_filter(pext,20,2) # smooth pext (for visualization) -- not physically affected
                    
                    output_abs['wavelength(micron)'] = w*1e4
                    output_abs['p/NH(U=%.1f,ngas=%.1e,fmax=%.1f)'%(U_rad,n_gas,f_max)] = pext/n_gas
                    output_emi['wavelength(micron)'] = w*1e4
                    output_emi['pem(U=%.1f,ngas=%.1e,fmax=%.1f)'%(U_rad,n_gas,f_max)] = pem
                    output_ext['wavelength(micron)'] = w*1e4
                    output_ext['A_per_Ngas(U=%.1f,ngas=%.1e,fmax=%.1f)'%(U_rad,n_gas,f_max)] = A_per_Ngas

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
                    
                    pext = savgol_filter(pext,20,2) # smooth pext (for visualization) -- not physically affected
                    
                    output_abs['wavelength(micron)'] = w*1e4
                    output_abs['p/NH(U=%.1f,ngas=%.1e,fmax=%.1f)'%(U_rad,n_gas,f_max)] = pext/n_gas
                    output_ext['wavelength(micron)'] = w*1e4
                    output_ext['A_per_Ngas(U=%.1f,ngas=%.1e,fmax=%.1f)'%(U_rad,n_gas,f_max)] = A_per_Ngas

                    ax1.semilogx(w * 1e4, pext / n_gas, label=f'U={U_rad:.1f} -- n$_{{\\rm H}}$={n_gas:.1e} -- f$_{{\\rm max}}$={f_max:.2f}')
                
                    if (ratd):
                        ax11.loglog(w * 1e4, A_per_Ngas,ls='--')
                    else:
                        ax11.loglog(w * 1e4, A_per_Ngas,color='k',ls='--')

                    if (first):
                        ax11.loglog(w*1e-4,np.ones(len(w)),color='k',ls='-',label='$\\rm pol.\\,spectrum$')
                        ax11.loglog(w*1e-4,np.ones(len(w)),color='k',ls='--',label='$\\rm Extinction\\, curve$')
                elif p_plot_option == 'Thermal dust Polarization':
                    w, pem, A_per_Ngas = results
                    output_emi['wavelength(micron)'] = w*1e4
                    output_emi['pem(U=%.1f,ngas=%.1e,fmax=%.1f)'%(U_rad,n_gas,f_max)] = pem
                    output_ext['wavelength(micron)'] = w*1e4
                    output_ext['A_per_Ngas(U=%.1f,ngas=%.1e,fmax=%.1f)'%(U_rad,n_gas,f_max)] = A_per_Ngas
                    ax2.semilogx(w * 1e4, pem, label=f'U={U_rad:.1f} -- n$_{{\\rm H}}$={n_gas:.1e} -- f$_{{\\rm max}}$={f_max:.2f}')
                first=False

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

# ----------------------------------------------------------------------------------------- #
# Plots polarization spectra
# ----------------------------------------------------------------------------------------- #
st.markdown('### Visualizations')
plot_figures()

# ----------------------------------------------------------------------------------------- #
# ASCII files for downloading
# ----------------------------------------------------------------------------------------- #
st.divider()
st.markdown('### ASCII files')

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode("utf-8")

my_large_df_abs = pd.DataFrame(data=output_abs)
csv_abs = convert_df(my_large_df_abs)

my_large_df_emi = pd.DataFrame(data=output_emi)
csv_emi = convert_df(my_large_df_emi)

my_large_df_ext = pd.DataFrame(data=output_ext)
csv_ext = convert_df(my_large_df_ext)

if p_plot_option=='Both':
    col_save1, col_save2, col_save3 = st.columns(3)
    col_save1.download_button(
        label="Download pext/NH as CSV",
        data=csv_abs,
        file_name="pabs.csv",
        mime="text/csv",
    )

    col_save2.download_button(
        label="Download pem as CSV",
        data=csv_emi,
        file_name="pemi.csv",
        mime="text/csv",
    )

    col_save3.download_button(
        label="Download ext_curve as CSV",
        data=csv_ext,
        file_name="ext_curve.csv",
        mime="text/csv",
    )
elif p_plot_option=='Starlight Polarization':
    col_save1, col_save2 = st.columns(2)
    col_save1.download_button(
        label="Download pext/NH as CSV",
        data=csv_abs,
        file_name="pabs.csv",
        mime="text/csv",
    )

    col_save2.download_button(
        label="Download ext_curve as CSV",
        data=csv_ext,
        file_name="ext_curve.csv",
        mime="text/csv",
    )

elif p_plot_option=='Thermal dust Polarization':
    col_save1, col_save2 = st.columns(2)
    col_save1.download_button(
        label="Download pem as CSV",
        data=csv_emi,
        file_name="pem.csv",
        mime="text/csv",
    )

    col_save2.download_button(
        label="Download ext_curve as CSV",
        data=csv_ext,
        file_name="ext_curve.csv",
        mime="text/csv",
    )

st.sidebar.markdown('''
---
Model details: please refer to \\
https://ui.adsabs.harvard.edu/abs/2020ApJ...896...44L \\
https://ui.adsabs.harvard.edu/abs/2021ApJ...906..115T \\
https://www.aanda.org/articles/aa/pdf/2024/09/aa50127-24.pdf
''')
# st.sidebar.divider()
st.sidebar.markdown('''
---
Created with ❤️ by `Le N. Tram`

Model: https://github.com/lengoctram/DustPOL-py (version 1.6) \\
Contact: nle 'at' strw 'dot' leidenuniv 'dot' nl
''')
