import numpy as np
import matplotlib.pyplot as plt

def fv_0(E,L):
    geom = np.exp(-E*L)/(1.-np.exp(-E*L))
    d0 = -1.*(L**2/(4.*E) + L/E**2 + 2/E**3) * np.exp(-0.5*E*L)
    d1 = geom*(np.exp(0.5*E*L)*(-L**2/(4.*E)+L/E**2-2./E**3)+2./E**3)
    d2 = geom*(np.exp(-0.5*E*L)*(L**2/(4.*E)+L/E**2+2./E**3)-2./E**3)
    d = d0-d1-d2
    gs = -2./E**3
    return d/gs

E = np.arange(0.1,1.6,0.01)
L = np.arange(1.0,10.0,0.01)

relerr = []
for e in E:
    temp = []
    for l in L:
        temp.append(fv_0(e,l))
    relerr.append(temp)

plt_axes = [0.14,0.14,0.825,0.825]
figsize = (3.50394*2,2*2.1655535534)
fig = plt.figure('relerr',figsize=figsize)
ax = plt.axes(plt_axes)
im = ax.imshow(relerr)
ax.set_xticks(E)
ax.set_yticks(L)
plt.show()
