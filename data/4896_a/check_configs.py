import numpy as np

c0 = open('bar_s22_SS_tsrc_0.dat.re')
c24 = open('bar_s22_SS_tsrc_24.dat.re')

cfg0 = []
for line in c0:
    cfg0.append(int(line.split('_')[0]))

cfg24 = []
for line in c24:
    cfg24.append(int(line.split('_')[0]))

ucfg0 = np.unique(cfg0)
ucfg24 = np.unique(cfg24)

d0 = {key:0 for key in ucfg0}
d24 = {key:0 for key in ucfg24}

for i in cfg0:
    d0[i] += 1

for i in cfg24:
    d24[i] += 1

print(cfg0)
print(cfg24)
print(len(cfg0))
print(len(cfg24))
print(d0)
print(d24)
