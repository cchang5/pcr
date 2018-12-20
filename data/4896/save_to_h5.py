import numpy as np
import pandas as pd
import h5py as h5
import os
import gvar as gv

list3pt = """
barff_s22_SS_g8_T8_tsrc_0.dat.re
barff_s22_SS_g8_T8_tsrc_48.dat.re
barff_s22_SS_g8_T10_tsrc_0.dat.re
barff_s22_SS_g8_T10_tsrc_48.dat.re
barff_s22_SS_down_g8_T8_tsrc_0.dat.re
barff_s22_SS_down_g8_T8_tsrc_48.dat.re
barff_s22_SS_down_g8_T10_tsrc_0.dat.re
barff_s22_SS_down_g8_T10_tsrc_48.dat.re
"""
list2pt = """
bar_s22_SS_tsrc_0.dat.re
bar_s22_SS_tsrc_48.dat.re
"""

list2pt = list2pt.split('\n')[1:-1]
list3pt = list3pt.split('\n')[1:-1]

data = dict()
data3d = dict()
for c in list2pt:
    df = pd.read_csv(c,delimiter=' ',header=None)
    tag = df[0][0]
    corr = np.array([np.array(df.loc[df[0] == tag.replace('t0','t%s' %t)].values)[:,1:-1] for t in range(96)])
    corr = np.swapaxes(corr,0,1) # swap axis to get [cfg,t,z]
    data3d[c] = corr
    nl = len(corr[0,0])
    z = np.roll(np.arange(nl)-nl/2+1,nl//2+1)
    print("tag:", tag, "nl:", nl, "shape:", np.shape(corr), "type:", type(corr), type(corr[0]), type(corr[0,0]), type(corr[0,0,0]))
    for n in [0,1,2]:
        k = 2.*np.pi*n/nl
        data['nucleon_q%s' %str(n)] = np.sum(corr*np.cos(k*z),axis=2)
        if n == 0:
            D = -0.5*z**2
        else:
            D = -0.5*z*np.sin(k*z)/k
        # project to 3 momentum
        data['dnucleon_q%s' %str(n)] = np.sum(D*corr,axis=2)
    data['znucleon'] = 0.5*(np.sum(corr,axis=1)+np.roll(np.sum(corr,axis=1)[:,::-1],1,axis=1))
for c in list3pt:
    df = pd.read_csv(c,delimiter=' ',header=None)
    tag = df[0][0]
    corr = np.array([np.array(df.loc[df[0] == tag.replace('t0','t%s' %t)].values)[:,1:-1] for t in range(96)])
    corr = np.swapaxes(corr,0,1) # swap axis to get [cfg,t,z]
    data3d[c] = corr
    nl = len(corr[0,0])
    z = np.roll(np.arange(nl)-nl/2+1,nl//2+1)
    print("tag:", tag, "nl:", nl, "shape:", np.shape(corr), "type:", type(corr), type(corr[0]), type(corr[0,0]), type(corr[0,0,0]))
    for n in [0,1,2]:
        k = 2.*np.pi*n/nl
        data['gV_q%s' %str(n)] = np.sum(corr*np.cos(k*z),axis=2)
        if n == 0:
            D = -0.5*z**2
        else:
            D = -0.5*z*np.sin(k*z)/k
        T=c.split('T')[1].split('_')[0]
        if len(c.split('_')) == 7:
            i = 'u'
        else:
            i = 'd'
        data['dgV%s_q%s_T%s' %(i,str(n),str(T))] = np.sum(D*corr,axis=2)
try: os.remove('../fast_load/4896_1d.h5')
except: pass
hf = h5.File('../fast_load/4896_1d.h5','w')
for k in data.keys():
    hf.create_dataset(k,data=data[k].astype(float))
hf.close()

try: os.remove('../fast_load/4896_3d.h5')
except: pass
hf = h5.File('../fast_load/4896_3d.h5','w')
for k in data3d.keys():
    hf.create_dataset(k,data=data3d[k].astype(float))
hf.close()
