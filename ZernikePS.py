
# -*- coding: utf-8 -*-
"""
生成任意直径的Kolmogorov大气随机相位屏
采用Zernike法, Atmospheric wavefront simulation using zernike polynomials_Nicolas Roddier 1990
"""

import math
import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
from scipy import optimize
import time
from astropy.io import fits
import os
import matplotlib


A = 2*(gamma(6/5)*24/5)**(5/6) #
B=(gamma(11/6))**2/2/(np.pi)**(11/3)
C=A*B/2


#根据文件，储存路径，和储存名称来讲文件保存为fits
def save(file,saving_path,saving_name): 
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    file=file.astype(np.float32)
    hdu=fits.PrimaryHDU(file)
    hdulist=fits.HDUList([hdu])
    path=os.path.join(saving_path,saving_name)
    hdulist.writeto(path,overwrite=True)

def jnmlist(N):
    #预先计算好j,n,m对应列表
    #N是个整数
    ini=[]
    count = 1
    for ii in range(N):
        n = ii
        for mm in range(n+1):
            m = mm
            if np.mod(n-m,2)==0:
                ini.append([count,n,m])
                if m!=0:
                    count=count+1
                    ini.append([count,n,m])
                    
                count=count+1
    return(ini)
    


def aj0j1(j0,j1,jnm):
    #计算协方差,j0和j1表示项数，从1开始
    n0 = jnm[j0-1][1]
    m0 = jnm[j0-1][2]
    n1 = jnm[j1-1][1]
    m1 = jnm[j1-1][2]
    if (m0 == m1 and np.mod(j0-j1,2)==0) or (m0==0 and m1==0):
        #print(n0,n1,m0)
        #----------N Roddier文章的协方差-------------------实际是有一丢丢问题的----主要是相位的功率谱的0.0229他计算的有点问题,eq(6) in Nicolas's paper is slightly wrong, we give a corrected version here.
        #K = gamma(14/3)*((24/5)*gamma(6/5))**(5/6)*(gamma(11/6))**2*(-1)**((n0+n1-2*m0)/2)*np.sqrt((n0+1)*(n1+1))/2/np.pi**2
        #cov = K*gamma((n0+n1-5/3)/2)/gamma((n0-n1+17/3)/2)/gamma((n1-n0+17/3)/2)/gamma((n0+n1+23/3)/2)
        #----------Noll文章里的式(25)加上Sasiela的Mellin变换------------------------------------
        coeffront = C*(2*np.pi)**(11/3)*gamma(7/3)*gamma(17/6)/np.pi/2**(5/3)/np.sqrt(np.pi)
        cov = coeffront*(-1)**((n0+n1-2*m0)/2)*np.sqrt((n0+1)*(n1+1))*gamma(-11/6+(n0+n1+2)/2)/gamma(17/6+(n0+n1+2)/2)/gamma(17/6+(n0-n1)/2)/gamma(17/6+(n1-n0)/2)
    else:
        cov=0
    return(cov)

def Rmn(n,m,r,R):
    summ = np.zeros(r.shape,dtype=np.float64)
    for s in range((n-m)//2+1):
        summ = summ + (-1)**s*np.float64(math.factorial(n-s))*(r/R)**(n-2*s)/math.factorial(s)/np.float64(math.factorial((n+m)//2-s))/np.float64(math.factorial((n-m)//2-s))
    return(summ)

def zernikeps(jj,jnm,r0,NN,R=1):
    #jj:希望用jj项zernike多项式来生成相位屏从第二项开始 因为第一项是piston
    #R:半径，默认为1m
    #jnm提前计算好的j,n,m列表
    #NN:相位屏的采样点数
    iniarr = np.ones((jj-1,jj-1)) #不包括第一项
    for i in range(2,jj+1): #从第2项到第jj项
        for j in range(2,jj+1):
            iniarr[i-2][j-2] = aj0j1(i,j,jnm)
    Carr = np.array(iniarr)
    u,s,v=np.linalg.svd(Carr)
    Brandom = np.random.normal(0,np.sqrt(s)).reshape(jj-1,1)
    Arandom = np.dot(u,Brandom)*(2*R/r0)**(5/6)

    inips = np.zeros((NN+1,NN+1),dtype=np.float64)
    y,x=np.mgrid[-NN//2:NN//2+1,-NN//2:NN//2+1]/(NN//2)
    r=np.sqrt(y**2+x**2)*R
    theta = np.arctan2(y,x)
    
    for term in range(2,jj+1):
        n = jnm[term-1][1]
        m = jnm[term-1][2]
        if m == 0:
            Z = np.sqrt(n+1)*Rmn(n,m,r,R)
        elif np.mod(term,2) == 0:
            Z = np.sqrt(n+1)*np.sqrt(2)*np.cos(m*theta)*Rmn(n,m,r,R)
        elif np.mod(term,2) == 1:
            Z = np.sqrt(n+1)*np.sqrt(2)*np.sin(m*theta)*Rmn(n,m,r,R)

        inips = inips+ Arandom[term-2]*Z
    inips[r>R]=0
    return(Carr,inips)
        
def pstest(N,jnm):
    #生成N个相位屏测试
    
    Dr=np.zeros(128)
    for i in range(N):
        #print(i)
        ps = zernikeps(50,jnm,0.1,256,R=0.12)[1]
        yy,xx=ps.shape
        pscut = ps[64:192,64:192]
        pscut1pad = np.tile(pscut[:,0].reshape(128,1),(1,128))
        
        Dr = Dr + (pscut[10,:]-pscut1pad[10,:])**2
    Dr = Dr/N
    return(Dr)
    
        


if __name__ == '__main__':
    start = time.perf_counter()
    jnm=jnmlist(200) #precalculated j,n,m list
    
    jj=120 #zernike多项式的项数
    r0=0.1 #Fried's parameter
    NN=256 #Number of grid points
    R=1 #Radius 1m
    
    Carr,ps = zernikeps(jj,jnm,r0,NN,R=R)
    plt.imshow(ps)
    plt.colorbar()
    plt.show()
    

    
   
    
    
    
    
    
    
    
    
    
    
    end=time.perf_counter()
    print('running time:',end-start)
    
    
    
    
    
    
    
