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
    def __init__(self,clist,mom,nstate):
        # nested for loop in the following order
        self.clist = clist # list of correlators [nucleon, dnucleon, gV, dgV]
        self.mom = mom # momentum list [0,1,2,...]
        self.nstate = nstate # n-states in ansatz
    # FORMAT DATA FOR FITTER
    # get independent variables
    def x(self,switches,data,trange):
        x = dict()
        for k in data.keys():
            cla = k.split('_')[0]
            x[k] = dict()
            x[k]['t'] = trange[cla]
            if cla in ['gV','dgV']:
                x[k]['T'] = int(k.split('_')[2][1:])
        return x
    # splices correct t-region for dependent variables
    def y(self,x,data):
        y = dict()
        for k in data.keys():
            y[k] = data[k][x[k]['t']]
        return y
    # FORMAT PRIORS
    def p(self,switches):
        p = dict()
        pass
        return p
    # FIT FUNCTIONS
    # two point fit functions
    def E(self,p,q,n):
        En = p['E0_q%s' %str(q)]
        for i in range(n):
            En += np.exp(p['E%s_q%s' %(str(i+1),str(q))])
        return En
    def Z(self,p,q,n,s):
        return p['Z%s%s_q%s' %(str(s),str(n),str(q))]
    def dZ(self,p,q,ns):
        return p['dZ%s%s_q%s' %(str(s),str(n),str(q))]
    def c2pt(self,T,En,Zn_snk,Zn_src):
        return Zn_snk*Zn_src*np.exp(-En*T) / (2.*En)
    def twopt(T,p,q,snk,src):
        r = 0
        for n in range(self.nstate):
            En = self.E(p,q,n)
            Zn_snk = self.Z(p,n,q,snk)
            Zn_src = self.Z(p,n,q,src)
            r += self.c2pt(T,En,Zn_snk,Zn_src)
        return r
    def dtwopt(T,p,q,snk,src):
        r = 0
        for n in range(self.nstate):
            En = self.E(p,q,n)
            Zn_snk = self.Z(p,n,q,snk)
            Zn_src = self.Z(p,n,q,src)
            dZn_snk = self.dZ(p,n,q,snk)
            dZn_src = self.dZ(p,n,q,src)
            c2ptn = self.c2pt(T,En,Zn_snk,Zn_src)
            r += c2ptn * (dZn_snk/Zn_snk + dZn_src/Zn_src - 0.5/En**2 - 0.5*T/En)
        return r
    # three point fit functions
    def G(self,p,m,n,q,cur):
        return p['G%s%s%s_q%s' %(str(cur),str(m),str(n),str(q))]
    def dG(self,p,n,m,q,cur):
        return p['dG%s%s%s_q%s' %(str(cur),str(m),str(n),str(q))]
    def c3pt(self,t,T,Em,En,Zm_snk,Zn_src,Gmn):
        return Zm_snk*Gmn*Zn_src*np.exp(-Em*T-(En-Em)*t)/(4.*Em*En)
    def threept(t,T,p,q,snk,cur,src):
        r = 0
        for m in range(self.nstate):
            for n in range(self.nstate):
                Em = self.E(p,q,m)
                En = self.E(p,0,n)
                Zm_snk = self.Z(p,m,0,snk)
                Zn_src = self.Z(p,n,q,src)
                Gmn = self.G(p,m,n,q,cur)
                r += self.c3pt(t,T,Em,En,Zm_snk,Zn_src,Gmn)
    def dthreept(t,T,p,q,snk,cur,src):
        r = 0
        for m in range(self.nstate):
            for n in range(self.nstate):
                Em = self.E(p,q,m)
                En = self.E(p,0,n)
                Zm_snk = self.Z(p,m,0,snk)
                Zn_src = self.Z(p,n,q,src)
                dZn_src = self.dZ(p,n,q,src)
                Gmn = self.G(p,m,n,q,cur)
                dGmn = self.dG(p,m,n,q,cur)
                c3ptmn = self.c3pt(t,T,Em,En,Zm_snk,Zn_src,Gmn)
                r += c3ptmn * (dGmn/Gmn + dZn_src/Zn_src - 0.5/Em**2 - 0.5*t/Em)
    # simultaneous fits
    def fit_function(x,p):
        pass
