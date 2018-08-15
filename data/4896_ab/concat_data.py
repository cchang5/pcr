import numpy as np
import pandas as pd

# concatenate three-point correlator
if False:
    head = 'barff_s22_SS'
    quark = ['','down_']
    operator = 'g8'
    tsnk = ['T8','T10']
    tsrc = [0,48]
    comp = ['re','im']
    
    for q in quark:
        for T in tsnk:
            for t in tsrc:
                for c in comp:
                    string = ''
                    for stream in ['a','b']:
                        fname = '%s_%s%s_%s_tsrc_%s_%s.dat.%s' %(head,q,operator,T,t,stream,c)
                        fopen = open(fname,'r')
                        for line in fopen:
                            string += line
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
            for line in fopen:
                string += line
            fopen.close()
        fwrite = '%s_%s.dat.%s' %(head,t,c)
        fopen = open('../4896/%s' %fwrite, 'w')
        fopen.write(string)
        fopen.flush()
        fopen.close()
