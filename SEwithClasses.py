# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 12:22:51 2019

@author: phill
"""

""" Schrodinger Equation Using Classes """

from math import pi, sqrt, cos, sin, exp
import numpy as np
import matplotlib.pyplot as plt

def main(NN, del_x, dt):
    
    N2 = int(NN/2)
    hbar = 1.054e-34
    h_nobar = hbar/(2*pi)
    meff = 0.067*9.11e-31 #This material is GaAs
    epsz = 8.85e-12 
    epsr = 13.1 #This material is GaAs
    eps = epsz*epsr
    ecoul = 1.6e-19
    eV2J = 1.6e-19
    J2eV = 1/eV2J
    J2meV = 1e3*J2eV
    
    ra = (0.5*hbar/meff)*(dt/del_x**2)
    rd = dt/hbar
    print("ra = ", ra, "rd = ", rd)
    DX = del_x*1e9
    XX = np.linspace(DX, DX*NN, NN)
    gamma = ecoul/meff
    gamdt = 0.5*gamma*dt
    
# --- HO potential and ground state ---
    
    E0 = 0.003
    w0 = (E0*eV2J)/hbar
    a_init = 100
    
    EREF = 0.001
    V = np.zeros(NN)
    prla = np.zeros(NN)
    Eref = EREF*eV2J
    kref = (meff*Eref/hbar**2)*del_x**2
    for n in range(0,NN):
        V[n] = 0.5*meff*((Eref/hbar)**2)*(del_x**2)*(n-N2)**2
        prla[n] = exp(-0.5*kref*(n-N2)**2)
        
    freq = Eref/h_nobar
    T_period = 1/freq
    T_dt = T_period/dt
    print("T_dt = ", T_dt)
    
    plt.subplot(2,1,1)
    plt.plot(XX, J2eV*V, 'k')
    plt.text(100, 0.007, "EREF = {} eV".format(EREF), fontsize=12)
    plt.grid()
    plt.ylabel('V (eV)')
    plt.show()
    
    psi1 = Psi(NN)
    psi1 = find_direction(psi1,prla)
    ptot1 = psi1.normalization()
    print("Ptot = ", round(ptot1,3))
    
    T = 0 
    a = a_init
    B = np.array([0,0,0])
    B[0] = input("B: X --> ")
    B[1] = input("B: Y --> ")
    B[2] = input("B: Z --> ")
    print("B: ", B[0], " ", B[1], " ", B[2])
    
    plotem(psi1, V, NN, del_x, T, dt, B)
    
    n_steps = 0
    while True:
        
        n_steps = int(input("How many time steps: --> "))
        
        if n_steps == 0:
            break
        
        for _ in range(n_steps):
            T = T + 1
            psi1 = run_fdtd_loop(NN, psi1, ra, rd, V, gamdt, B)
            
        plotem(psi1, V, NN, del_x, T, dt, B)
    
        ptot1, ptot1_up, ptot1_down = psi1.cal_ptot()
    
        print("ptot_up = ", round(ptot1_up,3), " ptot_down = ", 
               round(ptot1_down, 3), " ptot1 = ", round(ptot1,3))
    
        Bmax = np.sqrt(B[0]**2 + B[1]**2 + B[2]**2)
        omega_L = gamma*Bmax
        T_rev = 2*pi/omega_L
        T_rev_dt = round(T_rev/dt,2)
        T_rev4 = round(0.25*T_rev/dt,2)
        print("T_rev_dt = ",T_rev_dt," T_rev4 = ",T_rev4)
    
""" Creation of the state variables using classes """

class Psi_Complex:
    """ Creates the complex functions """
    
    def __init__(self,NN: int):
        self.real = np.zeros(NN)
        self.imag = np.zeros(NN)
        
    def ptot(self, pnorm):
        return round(np.sum((self.real/pnorm)**2 + (self.imag/pnorm)**2),3)

class Psi:
    """ Creates the state variable (up and down) """
    
    def __init__(self, NN):
        self.up = Psi_Complex(NN)
        self.down = Psi_Complex(NN)
        
    def normalization(self):
        pnorm = sqrt(np.sum(self.up.real**2 + self.up.imag**2 +
                            self.down.real**2 + self.down.imag**2))
        
        for n in range(100):
            self.up.real[n] = self.up.real[n]/pnorm
            self.up.imag[n] = self.up.imag[n]/pnorm
            self.down.real[n] = self.down.real[n]/pnorm
            self.down.imag[n] = self.down.imag[n]/pnorm
            
        ptot = (np.sum(self.up.real**2 + self.up.imag**2 +
                      self.down.real**2 + self.down.imag**2))
        
        return ptot
    
    def cal_ptot(self):
        
         ptot = (np.sum(self.up.real**2 + self.up.imag**2 +
                      self.down.real**2 + self.down.imag**2))
         ptot_up = np.sum(self.up.real**2 + self.up.imag**2)
         ptot_down = np.sum(self.down.real**2 + self.down.imag**2)
         
         return ptot, ptot_up, ptot_down
     
    def cal_observ(self, V, NN, del_x):
        
        hbar = 1.054e-34
        meff = 0.067*9.11e-31 #GaAs
        
        PEup = 0.
        psiup = np.zeros(NN,dtype = complex)
        for n in range(NN):
            psiup[n] = self.up.real[n] + 1j*self.up.imag[n]
            PEup = PEup + V[n]*psiup[n]*np.conjugate(psiup[n])
            
        PEdown = 0.
        psidown = np.zeros(NN,dtype = complex)
        for n in range(NN):
            psidown[n] = self.down.real[n] + 1j*self.down.imag[n]
            PEdown = PEdown + V[n]*psidown[n]*np.conjugate(psidown[n])
            
        PE = PEup + PEdown
        
        keup = 0 + 1j*0
        for n in range(1,NN-1):
            lap_p = psiup[n+1] - 2*psiup[n] + psiup[n-1]
            keup = keup + lap_p*np.conjugate(psiup[n])
            
        kedown = 0 + 1j*0
        for n in range(1,NN-1):
            lap_p = psidown[n+1] - 2*psidown[n] + psidown[n-1]
            kedown = kedown + lap_p*np.conjugate(psidown[n])   
            
        KE = -((hbar/del_x)**2/(2*meff))*(keup + kedown)
        
        return PE.real, KE.real
    
    def cal_S(self):
        
        psiu = self.up.real + 1j*self.up.imag
        psid = self.down.real + 1j*self.down.imag
        Sx = round(np.sum(psid*np.conj(psiu)) + np.sum(psiu*np.conj(psid)),2)
        Sy = round(np.sum(-1j*psid*np.conj(psiu)) + 
                   1j*np.sum(psiu*np.conj(psid)),2)
        Sz = round(np.sum(psiu*np.conj(psiu)) - np.sum(psid*np.conj(psid)),2)
        
        return Sx.real, Sy.real, Sz.real
    
# -------------------------------------------------------------------------
        
""" Functions for the FDTD calculations """

def run_fdtd_loop(NN, psi, ra, rd, V, gamdt, B):
    for n in range(1,NN-1):
        psi.up.real[n] = (psi.up.real[n]-ra*(psi.up.imag[n-1] - 
                   2*psi.up.imag[n]+psi.up.imag[n+1])+rd*V[n]*psi.up.imag[n]
            + gamdt*(-B[0]*psi.down.imag[n] + B[1]*psi.down.real[n] - 
                     B[2]*psi.up.imag[n])) 
        
    for n in range(1,NN-1):
        psi.up.imag[n] = (psi.up.imag[n] + ra*(psi.up.real[n] - 
                   2*psi.up.real[n]+psi.up.real[n+1])-rd*V[n]*psi.up.real[n]
            + gamdt*(B[0]*psi.down.real[n] + B[1]*psi.down.imag[n] + 
                     B[2]*psi.up.real[n]))
            
    for n in range(1,NN-1):
        psi.down.real[n] = (psi.down.real[n] - ra*(psi.down.imag[n-1] - 
                   2*psi.down.imag[n]+psi.down.imag[n+1])
            +rd*V[n]*psi.down.imag[n] + gamdt*(-B[0]*psi.up.imag[n] -
                 B[1]*psi.up.real[n] + B[2]*psi.down.imag[n]))

    for n in range(1,NN-1):
        psi.down.imag[n] = (psi.down.imag[n] + ra*(psi.down.real[n-1] - 
                   2*psi.down.real[n] + psi.down.real[n+1]) - 
            rd*V[n]*psi.down.real[n] + gamdt*(B[0]*psi.up.real[n] - 
                          B[1]*psi.up.imag[n] - B[2]*psi.down.real[n]))
            
    return psi

def plotem(psi1, V, NN, del_x, T, dt, B):
    
    J2meV = 1e3/1.6e-19
    N2 = int(NN/2)
    DX = del_x*1e9
    XX = np.linspace(DX, DX*NN, NN)
    hbar = 1.054e-34
    ecoul = 1.6e-19
    meff = 0.067*9.11e-31 #GaAs
    gamma = ecoul/meff
    
    plt.subplot(3,1,1)
    plt.plot(psi1.up.real,'b')
    plt.plot(psi1.up.imag,'r--')
    
    plt.ylabel('up')
    plt.title('Harmonic Oscillator')
    plt.grid()
    
    plt.subplot(3,1,2)
    plt.plot(psi1.down.real,'b')
    plt.plot(psi1.down.imag,'r--')
    
    plt.grid()
    plt.ylabel('down')
    
    PE1, KE1 = psi1.cal_observ(V, NN, del_x)
    Etot = KE1 + PE1
    
    KE1 = round(J2meV*KE1, 3)
    PE1 = round(J2meV*PE1, 3)
    Sx1, Sy1, Sz1 = psi1.cal_S()
    
    Bmax = np.sqrt(B[0]**2 + B[1]**2 + B[2]**2)
    omega_L = gamma*Bmax
    T_rev = 2*pi/omega_L
    T_rev_dt = round(T_rev/dt, 3)
    T_rev4 = round(0.25*T_rev/dt, 3)
    
    time = round(T*dt*1e12, 2)
    VM = 15
    
    plt.subplot(3,1,3)
    plt.plot(XX, J2meV*V)
    plt.grid()
    plt.axis([0,NN*DX,0,VM])
    plt.text(20,.85*VM,"T = {}".format(T))
    plt.text(60,.85*VM," = {} ps".format(time))
    
    plt.text(150,.85*VM,"KE1:   {}".format(KE1),fontsize=10)
    plt.text(220,.85*VM,"PE1:     {}".format(PE1),fontsize=10)
   
    plt.text(10,.25*VM,"S1: {}".format(Sx1))
    plt.text(60,.25*VM," {}".format(Sy1))
    plt.text(110,.25*VM," {}".format(Sz1))
        
    plt.text(190,.25*VM,"B:  {}".format(B[0]))
    plt.text(230,.25*VM," {}".format(B[1]))
    plt.text(270,.25*VM," {}".format(B[2]))
    plt.text(100,.05*VM," B_rev = {}".format(T_rev_dt))
    plt.text(180,.05*VM," / {}".format(T_rev4))
    plt.savefig('dat.png')
    plt.show()   
    
    print("<S1>: ",Sx1," ",Sy1," ",Sz1)
    
    T_rev = 2*pi*hbar/Etot
    print("Etot = ", round(J2meV*Etot, 3), " T_rev = ", round(1e12*T_rev, 4),
          " ps")
    
    T_iter = round(int(T_rev/dt), 1)
    T4 = round(int(0.25*T_rev/dt), 1)
    print("T_iter = ", T_iter, " T4 = ", T4)
    
def find_direction(psi, ps_eig):
    
    theta = int(input("theta (degrees) ---> "))
    phi = int(input("phi (degress) ---> "))
    print("theta = ", theta, " phi = ", phi)
    theta = (pi/180)*theta
    phi = (pi/180)*phi
    
    psi.up.real = cos(theta/2)*ps_eig
    psi.up.imag = 0*ps_eig
    psi.down.real = sin(theta/2)*cos(phi)*ps_eig
    psi.down.imag = sin(theta/2)*sin(phi)*ps_eig
    
    return psi

if __name__ == '__main__':
    main(NN = 100, del_x = 3e-9, dt = 1e-15)