import numpy as np
import pandas as pd

# concatenate three-point correlator
if True:
    head = 'barff_s22_SS'
    quark = ['','down_']
    operator = 'g8'
    tsnk = ['T8','T10']
    tsrc = [0,48]
    comp = ['re','im']
    
    cfgs = dict()
    for q in quark:
        for T in tsnk:
            for t in tsrc:
                for c in comp:
                    string = ''
                    for stream in ['a','b']:
                        fname = '%s_%s%s_%s_tsrc_%s_%s.dat.%s' %(head,q,operator,T,t,stream,c)
                        fopen = open(fname,'r')
                        cfgs['%s_%s_%s_%s_%s' %(q,T,t,c,stream)] = []
                        for line in fopen:
                            sidx = line.index('_')+1
                            string += line[sidx:]
                            cfgs['%s_%s_%s_%s_%s' %(q,T,t,c,stream)].append(int(line.split('_')[0]))
                        cfgs['%s_%s_%s_%s_%s' %(q,T,t,c,stream)] = np.unique(cfgs['%s_%s_%s_%s_%s' %(q,T,t,c,stream)])
                        fopen.close()
                    fwrite ='%s_%s%s_%s_tsrc_%s.dat.%s' %(head,q,operator,T,t,c)
                    fopen = open('../4896/%s' %fwrite, 'w')
                    fopen.write(string)
                    fopen.flush()
                    fopen.close()


# concat two-point correlator
head = 'bar_s22_SS_tsrc'
tsrc = [0,48]
comp = ['re','im']
for t in tsrc:
    for c in comp:
        string = ''
        for stream in ['a','b']:
            fname = '%s_%s_%s.dat.%s' %(head,t,stream,c)
            fopen = open(fname,'r')
            cfgs['%s_%s_%s' %(t,c,stream)] = []
            for line in fopen:
                sidx = line.index('_')+1
                string += line[sidx:]
                cfgs['%s_%s_%s' %(t,c,stream)].append(int(line.split('_')[0]))
            cfgs['%s_%s_%s' %(t,c,stream)] = np.unique(cfgs['%s_%s_%s' %(t,c,stream)])
            fopen.close()
        fwrite = '%s_%s.dat.%s' %(head,t,c)
        fopen = open('../4896/%s' %fwrite, 'w')
        fopen.write(string)
        fopen.flush()
        fopen.close()

# count configs
cfgcnt = dict()
for k in cfgs:
    for i in cfgs[k]:
        try:
            cfgcnt[i] += 1
        except:
            cfgcnt[i] = 1
print(cfgcnt)
#for k in cfgcnt:
#    if cfgcnt[k] != 40:
#        print(k)
