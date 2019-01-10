import gvar as gv
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt

def load_switches():
    s = dict()
    # PLOTTING OPTIONS
    s['plot'] = dict()
    s['plot']['correlator'] = True # data (effective mass, Zs, etc...)
    s['plot']['stability'] = True # correlator stability after fit
    # FITTING OPTIONS
    s['fit'] = dict()
    s['fit']['switch'] = True # turn fitting on/off
    s['fit']['simultaneous'] = False
    s['fit']['chain'] = True
    # choose dataset
    s['fit']['clist']   = ['nucleon','gV'] #,'dnucleon','dgV'] # the derivatives should be fit with the correlators or else there's no handel on the parameters
    s['fit']['snk_src'] = ['SS']
    s['fit']['mom']     = [0,1] # n in 2*n*pi/L
    s['fit']['nstate']  = 2
    s['fit']['ens'] = '4896' # 3296 or 4896
    s['fit']['tail_correction'] = True
    # time region
    s['fit']['T_3pt']   = [8,10] #,14] # t_snk for 3pt: only 8, 10 or 14
    s['fit']['trange'] = dict()
    if s['fit']['ens'] in ['3296']:
        s['fit']['trange']['nucleon_tmin']   = [4,4] # snk location 4
        s['fit']['trange']['nucleon_tmax']   = [12,12] # snk location 10
        s['fit']['trange']['dnucleon_tmin']   = [4,4] # snk location 4
        s['fit']['trange']['dnucleon_tmax']   = [8,8] # snk location 8
        s['fit']['trange']['gV_T8_tmin']   = [2,2] # current insertion time 2
        s['fit']['trange']['gV_T8_tmax']   = [5,5] # current insertion time 5 
        s['fit']['trange']['gV_T10_tmin']   = [2,2] # current insertion time 2 
        s['fit']['trange']['gV_T10_tmax']   = [6,6] # current insertion time 6 
        s['fit']['trange']['dgV_T8_tmin']   = [1,1] # current insertion time 1
        s['fit']['trange']['dgV_T8_tmax']   = [4,4] # current insertion time 4 
        s['fit']['trange']['dgV_T10_tmin']   = [2,2] # current insertion time 2
        s['fit']['trange']['dgV_T10_tmax']   = [6,6] # current insertion time 6
    elif s['fit']['ens'] in ['4896']:
        s['fit']['trange']['nucleon_tmin']   = [5,5] # snk location 4
        s['fit']['trange']['nucleon_tmax']   = [8,8] # snk location 10
        s['fit']['trange']['dnucleon_tmin']   = [3,3] # snk location 4
        s['fit']['trange']['dnucleon_tmax']   = [7,7] # snk location 8
        s['fit']['trange']['gV_T8_tmin']   = [2,2] # current insertion time 2
        s['fit']['trange']['gV_T8_tmax']   = [6,6] # current insertion time 5 
        s['fit']['trange']['gV_T10_tmin']   = [3,3] # current insertion time 2
        s['fit']['trange']['gV_T10_tmax']   = [7,7] # current insertion time 6
        s['fit']['trange']['dgV_T8_tmin']   = [0,0] # current insertion time 1
        s['fit']['trange']['dgV_T8_tmax']   = [5,5] # current insertion time 4
        s['fit']['trange']['dgV_T10_tmin']   = [0,0] # current insertion time 1
        s['fit']['trange']['dgV_T10_tmax']   = [5,5] # current insertion time 4
    s['L'] = int(s['fit']['ens'][:2])
    return s

def load_priors():
    # PRIORS
    p = dict()
    # nucleon q = 0
    p['E0_q0'] = gv.gvar(0.7,0.14)
    p['Z0s_q0'] = gv.gvar(2.2E-4,1.1E-4) # z = Z/sqrt(2E)
    p['E1_q0'] = gv.gvar(-0.7,1.5) # log splitting to two pions
    p['Z1s_q0'] = gv.gvar(0.0,1.1E-4)
    p['E2_q0'] = gv.gvar(-0.7,1.5) # log splitting to two pions
    p['Z2s_q0'] = gv.gvar(0.0,0.55E-4)
    p['E3_q0'] = gv.gvar(-0.7,1.5) # log splitting to two pions
    p['Z3s_q0'] = gv.gvar(0.0,0.275E-4)
    # nucleon q = 1
    p['E0_q1'] = gv.gvar(0.75,0.15)
    p['Z0s_q1'] = gv.gvar(2E-4,1E-4)
    p['E1_q1'] = gv.gvar(-0.8,1.6) # splitting to zero momentum 2 pion and 1 pion
    p['Z1s_q1'] = gv.gvar(0.0,1E-4)
    p['E2_q1'] = gv.gvar(-0.8,1.6) # splitting to zero momentum 2 pion and 1 pion
    p['Z2s_q1'] = gv.gvar(0.0,0.5E-4)
    p['E3_q1'] = gv.gvar(-0.8,1.6) # splitting to zero momentum 2 pion and 1 pion
    p['Z3s_q1'] = gv.gvar(0.0,0.25E-4)
    # dnucleon q = 0 (Y comes before Z... so it's a derivative? something like that)
    p['Y0s_q0'] = gv.gvar(-1.5,0.5) # z' = Z'/[z*sqrt(2E)]
    p['Y1s_q0'] = gv.gvar(-0,0.5)
    p['Y2s_q0'] = gv.gvar(-0,0.5)
    p['Y3s_q0'] = gv.gvar(-0,0.5)
    # dnucleon q = 1
    p['Y0s_q1'] = gv.gvar(-1.5,0.5)
    p['Y1s_q1'] = gv.gvar(-0,0.5)
    p['Y2s_q1'] = gv.gvar(-0,0.5)
    p['Y3s_q1'] = gv.gvar(-0,0.5)
    # gV q = 0
    p['V00_q0'] = gv.gvar(0.6,0.6)
    p['V10_q0'] = gv.gvar(0.0,0.6)
    p['V11_q0'] = gv.gvar(0.0,0.6)
    p['V20_q0'] = gv.gvar(0.0,0.6)
    p['V21_q0'] = gv.gvar(0.0,0.6)
    p['V22_q0'] = gv.gvar(0.0,0.6)
    p['V30_q0'] = gv.gvar(0.0,0.6)
    p['V31_q0'] = gv.gvar(0.0,0.6)
    p['V32_q0'] = gv.gvar(0.0,0.6)
    p['V33_q0'] = gv.gvar(0.0,0.6)
    # gV q = 1
    p['V00_q1'] = gv.gvar(0.5,0.5)
    p['V10_q1'] = gv.gvar(0.0,0.5)
    p['V11_q1'] = gv.gvar(0.0,0.5)
    p['V20_q1'] = gv.gvar(0.0,0.5)
    p['V21_q1'] = gv.gvar(0.0,0.5)
    p['V22_q1'] = gv.gvar(0.0,0.5)
    p['V30_q1'] = gv.gvar(0.0,0.5)
    p['V31_q1'] = gv.gvar(0.0,0.5)
    p['V32_q1'] = gv.gvar(0.0,0.5)
    p['V33_q1'] = gv.gvar(0.0,0.5)
    # dgV q = 0
    p['U00_q0'] = gv.gvar(-5,5)
    p['U10_q0'] = gv.gvar(-0,5)
    p['U11_q0'] = gv.gvar(-0,5)
    p['U20_q0'] = gv.gvar(-0,5)
    p['U21_q0'] = gv.gvar(-0,5)
    p['U22_q0'] = gv.gvar(-0,5)
    p['U30_q0'] = gv.gvar(-0,5)
    p['U31_q0'] = gv.gvar(-0,5)
    p['U32_q0'] = gv.gvar(-0,5)
    p['U33_q0'] = gv.gvar(-0,5)
    # dgV q = 1
    p['U00_q1'] = gv.gvar(-5,5)
    p['U10_q1'] = gv.gvar(-0,5)
    p['U11_q1'] = gv.gvar(-0,5)
    p['U20_q1'] = gv.gvar(-0,5)
    p['U21_q1'] = gv.gvar(-0,5)
    p['U22_q1'] = gv.gvar(-0,5)
    p['U30_q1'] = gv.gvar(-0,5)
    p['U31_q1'] = gv.gvar(-0,5)
    p['U32_q1'] = gv.gvar(-0,5)
    p['U33_q1'] = gv.gvar(-0,5)
    
    return p

def load_data():
    d = h5.File('./data/fast_load/4896_3d.h5','r')
    data = {k: d[k] for k in d.keys()}
    return data

def contract_current(d):
    mom = [0,1]
    tlist = [8,10]
    nsrc = len(np.unique([i.split('tsrc_')[1].split('.')[0] for i in d.keys()]))
    # initiate dictionary
    data = dict()
    for n in mom:
        data['nucleon_q%s' %str(n)] = []
        data['dnucleon_q%s' %str(n)] = []
        for T in tlist:
            data['gV_q%s_T%s' %(str(n),str(T))] = []
            data['dgV_q%s_T%s' %(str(n),str(T))] = []
            data['zgV_T%s' %str(T)] = []
    data['znucleon'] = []
    # make data
    list3pt = [i for i in d.keys() if i.split('_')[0] == 'barff']
    list2pt = [i for i in d.keys() if i.split('_')[0] == 'bar']
    for c in list2pt:
        corr = d[c]
        nl = len(corr[0,0])
        z = np.roll(np.arange(nl)-nl/2+1,nl//2+1)
        #print("tag:", c, "nl:", nl, "shape:", np.shape(corr), "type:", type(corr), type(corr[0]), type(corr[0,0]), type(corr[0,0,0]))
        for n in mom:
            k = 2.*np.pi*n/nl
            data['nucleon_q%s' %str(n)].append(np.sum(corr*np.cos(k*z),axis=2))
            if n == 0:
                D = -0.5*z**2
            else:
                D = -0.5*z*np.sin(k*z)/k
            # project to 3 momentum
            data['dnucleon_q%s' %str(n)].append(np.sum(D*corr,axis=2))
        data['znucleon'].append(0.5*(np.sum(corr,axis=1)+np.roll(np.sum(corr,axis=1)[:,::-1],1,axis=1)))
    for c in list3pt:
        corr = d[c]
        nl = len(corr[0,0])
        T=c.split('T')[1].split('_')[0]
        z = np.roll(np.arange(nl)-nl/2+1,nl//2+1)
        #print("tag:", c, "nl:", nl, "shape:", np.shape(corr), "type:", type(corr), type(corr[0]), type(corr[0,0]), type(corr[0,0,0]))
        for n in mom:
            if len(c.split('_')) == 7: phase = 1.0
            else: phase = -1.0
            k = 2.*np.pi*n/nl
            data['gV_q%s_T%s' %(str(n),str(T))].append(phase*np.sum(corr*np.cos(k*z),axis=2))
            if n == 0: D = -0.5*z**2
            else: D = -0.5*z*np.sin(k*z)/k
            data['dgV_q%s_T%s' %(str(n),str(T))].append(phase*np.sum(D*corr,axis=2))
        data['zgV_T%s' %str(T)].append(phase*0.5*(np.sum(corr,axis=1)+np.roll(np.sum(corr,axis=1)[:,::-1],1,axis=1)))
    # source average
    data = {k:np.sum(data[k],axis=0)/float(nsrc) for k in data.keys()}
    #for k in data.keys():
    #    print(k,np.shape(data[k]))
    return data

def plot_meff(d,x=[1,15]):
    fig = plt.figure('effective mass',figsize=(7,4.3))
    ax = plt.axes([0.14,0.155,0.825,0.825])
    x = np.arange(x[0],x[1]+1)
    for k in d.keys():
        if k.split('_')[0] == 'nucleon':
            meff = np.log(d[k]/np.roll(d[k],-1))[x]
            ax.errorbar(x=x,y=[i.mean for i in meff],yerr=[i.sdev for i in meff],ls='None',marker='o',capsize=3,fillstyle='none',label=k)
    plt.title('effective mass')
    ax.legend()
    plt.draw()

def plot_zeff(d,x=[1,15]):
    fig = plt.figure('scaled correlator',figsize=(7,4.3))
    ax = plt.axes([0.14,0.155,0.825,0.825])
    x = np.arange(x[0],x[1]+1)
    for k in d.keys():
        if k.split('_')[0] == 'nucleon':
            meff = np.log(d[k]/np.roll(d[k],-1))
            scor = (d[k]*np.exp(meff*np.arange(len(d[k]))))[x]
            ax.errorbar(x=x,y=[i.mean for i in scor],yerr=[i.sdev for i in scor],ls='None',marker='o',capsize=3,fillstyle='none',label=k)
    plt.title('scaled correlator')
    ax.legend()
    plt.draw()

def plot_zcor(d,x=[1,15]):
    fig = plt.figure('z-correlator effective mass',figsize=(7,4.3))
    ax = plt.axes([0.14,0.155,0.825,0.825])
    x = np.arange(x[0],x[1]+1)
    for k in d.keys():
        if k.split('_')[0] == 'znucleon':
            meff = np.log(d[k]/np.roll(d[k],-1))[x]
            ax.errorbar(x=x,y=[i.mean for i in meff],yerr=[i.sdev for i in meff],ls='None',marker='o',capsize=3,fillstyle='none',label=k)
    plt.title('z-correlator effective mass')
    ax.legend()
    plt.draw()

def plot_gV(d):
    fig = plt.figure('isovector vector matrix element',figsize=(7,4.3))
    ax = plt.axes([0.14,0.155,0.825,0.825])
    meff = dict()
    scor = dict()
    for k in d.keys():
        if k.split('_')[0] == 'nucleon':
            meff[k.split('_')[1]] = np.log(d[k]/np.roll(d[k],-1))
            scor[k.split('_')[1]] = np.sqrt(d[k]*np.exp(meff[k.split('_')[1]]*np.arange(len(d[k]))))
        else: pass
    for k in d.keys():
        if k.split('_')[0] == 'gV':
            T = int(k.split('_')[2][1:])
            x = np.arange(0,T+1)
            elem = (d[k]*(np.exp(meff['q0'][3]*float(T))/(scor['q0'][T]*scor[k.split('_')[1]][3])))[x]
            ax.errorbar(x=x,y=[i.mean for i in elem],yerr=[i.sdev for i in elem],ls='None',marker='o',capsize=3,fillstyle='none',label=k)
        else: pass
    plt.title('isovector vector matrix element')
    ax.legend()
    plt.draw()

def plot_zgV(d,x=[1,15]):
    fig = plt.figure('z-gV effective mass',figsize=(7,4.3))
    ax = plt.axes([0.14,0.155,0.825,0.825])
    x = np.arange(x[0],x[1]+1)
    for k in d.keys():
        if k.split('_')[0] == 'zgV':
            meff = np.log(d[k]/np.roll(d[k],-1))[x]
            ax.errorbar(x=x,y=[i.mean for i in meff],yerr=[i.sdev for i in meff],ls='None',marker='o',capsize=3,fillstyle='none',label=k)
    plt.title('z-gV effective mass')
    ax.legend()
    plt.draw()

def plot_dgV(d):
    fig = plt.figure('isovector vector derivative matrix element',figsize=(7,4.3))
    ax = plt.axes([0.14,0.155,0.825,0.825])
    meff = dict()
    scor = dict()
    for k in d.keys():
        if k.split('_')[0] == 'nucleon':
            meff[k.split('_')[1]] = np.log(d[k]/np.roll(d[k],-1))
            scor[k.split('_')[1]] = np.sqrt(d[k]*np.exp(meff[k.split('_')[1]]*np.arange(len(d[k]))))
        else: pass
    for k in d.keys():
        if k.split('_')[0] == 'dgV':
            T = int(k.split('_')[2][1:])
            q = k.split('_')[1]
            x = np.arange(0,T+1)
            elem = (d[k]/d[k[1:]])[x] + (x/(2.*meff[q][T]))[x]
            ax.errorbar(x=x,y=[i.mean for i in elem],yerr=[i.sdev for i in elem],ls='None',marker='o',capsize=3,fillstyle='none',label=k)
        else: pass
    plt.title('isovector derivative vector matrix element')
    ax.legend()
    plt.draw()
