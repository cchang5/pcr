import numpy as np
import pandas as pd

# concatenate three-point correlator
if True:
    head = 'barff_s22_SS'
    quark = ['','down_']
    operator = 'g8'
    tsnk = ['T8','T10'] #['T8','T10']
    tsrc = ['0','48'] #[0,24,48,72]
    comp = ['re','im']
    
    for q in quark:
        for T in tsnk:
            for t in tsrc:
                for c in comp:
                    string = ''
                    fname = '%s_%s%s_%s_tsrc_%s.dat.%s' %(head,q,operator,T,t,c)
                    fopen = open(fname,'r')
                    for line in fopen:
                        string += line[5:]
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
        fname = '%s_%s.dat.%s' %(head,t,c)
        fopen = open(fname,'r')
        for line in fopen:
            string += line[5:]
        fopen.close()
        fwrite = '%s_%s.dat.%s' %(head,t,c)
        fopen = open('../4896/%s' %fwrite, 'w')
        fopen.write(string)
        fopen.flush()
        fopen.close()
