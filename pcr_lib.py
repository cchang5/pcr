import numpy as np
import pandas as pd
import gvar as gv
import lsqfit
import matplotlib.pyplot as plt
import tqdm

# some plotting functions
def plot_2pt_z(data,title='2pt z-meff'):
    # two point
    z = np.arange(1,32)
    trange = [2,10]
    fig = plt.figure(title,figsize=(7,4.326237))
    ax = plt.axes([0.15,0.15,0.8,0.8])
    for t in range(trange[0],trange[1]+1):
        pltdata = gv.dataset.avg_data(data[:,t,:])
        #pltdata = (pltdata + np.roll(pltdata[::-1],1))/2 # fold data
        meff = pltdata[z] #np.arccosh((np.roll(pltdata,1)+np.roll(pltdata,-1))[z]/(2*pltdata[z]))
        ax.errorbar(z,y=[i.mean for i in meff],yerr=[i.sdev for i in meff])
    ax.set_yscale("log")
    plt.draw()

def plot_3pt_z(data,title='gV'):
    # three point
    z = np.arange(1,32)
    trange = [2,10]
    fig = plt.figure(title,figsize=(7,4.326237))
    ax = plt.axes([0.15,0.15,0.8,0.8])
    for t in range(trange[0],trange[1]+1):
        pltdata = gv.dataset.avg_data(data[:,t,:])[z]
    ax.errorbar(z,y=[i.mean for i in pltdata],yerr=[i.sdev for i in pltdata])
    ax.set_yscale("log")
    plt.draw()

# read csv files
def read_csv(switches):
    zdata = dict()
    # read two point
    if len(set(switches['fit']['clist']).intersection(['nucleon','dnucleon'])) > 0:
        for snk_src in switches['fit']['snk_src']:
            dataset2 = 'bar_s22_%s_tsrc_SRC.dat.re' %str(snk_src)
            dataset = list()
            for s in [0,48]: # automatically read both sources
                rd = dataset2.replace('SRC',str(s))
                df = pd.read_csv('./data/%s' %rd,delimiter=' ',header=None)
                tag = df[0][0]
                key =tag.replace('t0_','')
                # read zero momentum spatial correlator
                corr = np.array([np.array(df.loc[df[0] == tag.replace('t0','t%s' %t)].as_matrix())[:,1:-1] for t in range(96)])
                corr = np.swapaxes(corr,0,1) # swap axis to get [cfg,t,z]
                # delete missing config 1470 from other datasets
                if len(corr) == 196:
                    corr = np.delete(corr,47,axis=0)
                dataset.append(corr)
                # average sources
            zdata[key] = (dataset[0]+dataset[1])/2
            print('2pt:',key,np.shape(zdata[key]))
    # read three point
    if len(set(switches['fit']['clist']).intersection(['gV','dgV'])) > 0:
        for snk_src in switches['fit']['snk_src']:
            dataset3 = 'barff_s22_%s_g8_TSNK_tsrc_SRC.dat.re' %str(snk_src)
            for snk in switches['fit']['T_3pt']:
                dataset=list()
                for s in [0,48]:
                    rd = dataset3.replace('SRC',str(s)).replace('SNK',str(snk))
                    df = pd.read_csv('./data/%s' %rd,delimiter=' ',header=None)
                    tag = df[0][0]
                    key = tag.replace('t0_','')
                    # read zero momentum spatial correlator
                    corr = np.array([np.array(df.loc[df[0] == tag.replace('t0','t%s' %t)].as_matrix())[:,1:-1] for t in range(96)])
                    corr = np.swapaxes(corr,0,1) # swap axis to get [cfg,t,z]
                    # delete missing config 1470 from other datasets
                    if len(corr) == 196:
                        corr = np.delete(corr,47,axis=0)
                    dataset.append(corr)
                # average sources
                zdata[key] = (dataset[0]+dataset[1])/2
                print('3pt:',key,np.shape(zdata[key]))
    return zdata
    
# fitter class
class fitter_class():
    # INSTANTIATE CLASS
    def __init__(self,clist,snk_src,mom,nstate):
        # nested for loop in the following order
        self.clist = clist # list of correlators [nucleon, dnucleon, gV, dgV]
        self.snk_src = snk_src # smearing list ['SS','PS',...]
        self.mom = mom # momentum list [0,1,2,...]
        self.nstate = nstate # n-states in ansatz
    # FORMAT DATA FOR FITTER
    # get independent variables
    def x(self,switches,data,trange):
        x = dict()
        for k in data.keys():
            cla = k.split('_')[0]
            x[k] = dict()
            if cla in ['nucleon','dnucleon']:
                x[k]['t'] = trange[cla]
            elif cla in ['gV','dgV']:
                T = int(k.split('_')[2][1:])
                x[k]['T'] = T
                x[k]['cur'] = cla.split('d')[-1]
                x[k]['t'] = trange['%s_T%s' %(cla,T)]
            x[k]['q'] = int(k[-1])
            x[k]['snk'] = k.split('_')[1][0].lower()
            x[k]['src'] = k.split('_')[1][1].lower()
        return x
    # splices correct t-region for dependent variables
    def y(self,x,data):
        y = dict()
        for k in data.keys():
            y[k] = data[k][x[k]['t']]
        return y
    # FORMAT PRIORS
    def p(self,switches):
        # all priors
        ap = switches['p']
        # parameter dependences
        pdict = dict()
        pdict['nucleon'] = ['E','Z']
        pdict['dnucleon'] = ['E','Z','Y']
        pdict['gV'] = ['E','Z','V']
        pdict['dgV'] = ['E','Z','V','Y','U']
        p = dict()
        plist = np.array([])
        for c in self.clist:
            plist = np.unique(np.concatenate((plist,pdict[c])))
        for k in ap.keys():
            # filter prior, state, momentum
            if k[0] in plist and int(k[1]) <= int(self.nstate) and int(k[-1]) in self.mom:
                # create smearing list from self.snk_src
                smears = ''
                smears = [i.lower() for i in np.unique(list(smears.join(self.snk_src)))]
                # for energy and matrix elements
                if k[0] in ['E','V','U']:
                    p[k] = ap[k]
                # for smearing dependent quantities (overlaps)
                elif k[2] in smears:
                    p[k] = ap[k]
        return p
    # FIT FUNCTIONS
    # two point fit functions
    def E(self,p,q,n):
        En = p['E0_q%s' %str(q)]
        for i in range(n):
            En += np.exp(p['E%s_q%s' %(str(i+1),str(q))])
        return En
    def Z(self,p,q,n,s):
        return p['Z%s%s_q%s' %(str(n),str(s),str(q))]
    def dZ(self,p,q,n,s):
        return p['Y%s%s_q%s' %(str(n),str(s),str(q))]
    def c2pt(self,T,En,Zn_snk,Zn_src):
        return Zn_snk*Zn_src*np.exp(-En*T)
    def twopt(self,x,p):
        # unpack x
        T = x['t']
        q = x['q']
        snk = x['snk']
        src = x['src']
        # fit function
        r = 0
        for n in range(self.nstate):
            En = self.E(p,q,n)
            Zn_snk = self.Z(p,q,n,snk)
            Zn_src = self.Z(p,q,n,src)
            r += self.c2pt(T,En,Zn_snk,Zn_src)
        return r
    def dtwopt(self,x,p):
        # unpack x
        T = x['t']
        q = x['q']
        snk = x['snk']
        src = x['src']
        # fit function
        r = 0
        for n in range(self.nstate):
            En = self.E(p,q,n)
            Zn_snk = self.Z(p,q,n,snk)
            Zn_src = self.Z(p,q,n,src)
            dZn_snk = self.dZ(p,q,n,snk)
            dZn_src = self.dZ(p,q,n,src)
            c2ptn = self.c2pt(T,En,Zn_snk,Zn_src)
            r += c2ptn * (dZn_snk/Zn_snk + dZn_src/Zn_src - 0.5/En**2 - 0.5*T/En)
        return r
    # three point fit functions
    def G(self,p,q,m,n,cur):
        if n > m:
            if cur in ['gV']:
                return p['V%s%s_q%s' %(str(n),str(m),str(q))]
            else:
                return 0
        else:
            if cur in ['gV']:
                return p['V%s%s_q%s' %(str(m),str(n),str(q))]
            else:
                return 0
    def dG(self,p,q,m,n,cur):
        if cur in ['gV']:
            return p['U%s%s_q%s' %(str(m),str(n),str(q))]
        else:
            return 0
    def c3pt(self,t,T,Em,En,Zm_snk,Zn_src,Gmn):
        return Zm_snk*Gmn*Zn_src*np.exp(-Em*T-(En-Em)*t)
    def threept(self,x,p):
        # unpack x
        t = x['t']
        T = x['T']
        q = x['q']
        snk = x['snk']
        cur = x['cur']
        src = x['src']
        # fit function
        r = 0
        for m in range(self.nstate):
            for n in range(self.nstate):
                Em = self.E(p,0,m)
                En = self.E(p,q,n)
                Zm_snk = self.Z(p,0,m,snk)
                Zn_src = self.Z(p,q,n,src)
                Gmn = self.G(p,q,m,n,cur)
                r += self.c3pt(t,T,Em,En,Zm_snk,Zn_src,Gmn)
        return r
    def dthreept(self,x,p):
        # unpack x
        t = x['t']
        T = x['T']
        q = x['q']
        snk = x['snk']
        cur = x['cur']
        src = x['src']
        # fit function
        r = 0
        for m in range(self.nstate):
            for n in range(self.nstate):
                Em = self.E(p,0,m)
                En = self.E(p,q,n)
                Zm_snk = self.Z(p,0,m,snk)
                Zn_src = self.Z(p,q,n,src)
                dZn_src = self.dZ(p,q,n,src)
                Gmn = self.G(p,q,m,n,cur)
                dGmn = self.dG(p,q,m,n,cur)
                c3ptmn = self.c3pt(t,T,Em,En,Zm_snk,Zn_src,Gmn)
                r += c3ptmn * (dGmn/Gmn + dZn_src/Zn_src - 0.5/Em**2 - 0.5*t/Em)
        return r
    # simultaneous fits
    def fit_function(self,x,p):
        r = dict()
        for k in x.keys():
            cla = k.split('_')[0]
            if cla in ['nucleon']:
                r[k] = self.twopt(x[k],p)
            elif cla in ['dnucleon']:
                r[k] = self.dtwopt(x[k],p)
            elif cla in ['gV']:
                r[k] = self.threept(x[k],p)
            elif cla in ['dgV']:
                r[k] = self.dthreept(x[k],p)
        return r
