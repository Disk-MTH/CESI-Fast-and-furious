## Importation des bibliothèques nécessaires au calcul

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

## Déclaration des constantes

g = 9.81
L = 0.115
theta0_deg = 270   # angle initial en degrés
theta0 = theta0_deg*np.pi/180   # angle initial en radians
vAV_init = [theta0, 4.24/L]    # conditions initiales : angle initial = theta0, vitesse angulaire initiale = 0

t0 = 0
tfinal = 10
t = np.linspace(t0, tfinal,100)

## Déclaration de la fonction renvoyant le vecteur VitesseAccélération en fonction du vecteur AngleVitesse pour (E1)

def vecteurVitesseAcceleration1(vecteurAngleVitesse1, t):
    dvecteurAngleVitesse1dt = [vecteurAngleVitesse1[1], - (g/L)*np.sin(vecteurAngleVitesse1[0])]
    return dvecteurAngleVitesse1dt

## Déclaration de la fonction renvoyant le vecteur VitesseAccélération en fonction du vecteur AngleVitesse pour (E2)

def vecteurVitesseAcceleration2(vecteurAngleVitesse2, t):
    dvecteurAngleVitesse2dt = [vecteurAngleVitesse2[1], - (g/L)*vecteurAngleVitesse2[0]]
    return dvecteurAngleVitesse2dt

sol1 = odeint(vecteurVitesseAcceleration1, vAV_init, t)
sol2 = odeint(vecteurVitesseAcceleration2, vAV_init, t)

plt.plot(t, sol1[:, 0]*180/np.pi, color = "red")
plt.title("Position angulaire du pendule (E1)")
plt.xlabel("Temps (s)")
plt.ylabel("Position en degrés")
plt.show()
plt.plot(t, sol2[:, 0]*180/np.pi, color = "green")
plt.title("Position angulaire du pendule (E2)")
plt.xlabel("Temps (s)")

