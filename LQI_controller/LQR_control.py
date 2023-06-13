import math
import random
import gym
import numpy as np
import mujoco_py
import glfw
import time
import os
from scipy.linalg import solve_discrete_are
from state import SmallestSignedAngleBetween
from state import state_matrix
from state import MGIv2
from scipy.linalg import expm
from control.matlab import *







# Para visualizar que es lo que hace el agente
glfw.init()
test_steps = 1000
env = gym.make("Reacher-v4", render_mode="human") # brazo 2 DOF
observation, info = env.reset()


# recuperation des X et Y de la cible
x_star=observation[4]
y_star=observation[5]

#calcul des q1,q2 desirée
q1_star,q2_star=MGIv2(x_star,y_star)
#creation de notre rep d'etat souhaite X_star=(q1_star,q2_star,q1point_star,q2point_star)^t
X_star=np.array([[q1_star],[q2_star],[0],[0]])

g=0 # gravity
Ts=0.002
action=[0,0]

#Creation de nos matrice Q et R  qui doivent être diagonales
Q=np.identity(4)
R=np.diag([0.001,0.001])
#boucle de contrôle 
while(1):
	# calcul des q1 et q2 actuel
	cosq1=observation[0]
	cosq2=observation[1]
	sinq1=observation[2]
	sinq2=observation[3]
	q1_point=-observation[6]
	q2_point=-observation[7]
	
	q1=np.arctan2(sinq1,cosq1)
	q2=np.arctan2(sinq2,cosq2)
	
	#creation de notre rep d'etat courant current_X
	current_X=np.array([[q1],[q2],[q1_point],[q2_point]])
	#creation de notre rep d'etat ( dans notre cas X est l'erreur entre X_star et current_X)
	X=X_star-current_X

	#calcul de nos matrice A,B,C de notre systeme X_point =AX+Bu et Y=CX
	A,B,C=state_matrix(q1,q2,q1_point,q2_point,action[0],action[1],g)
	
	#discrétisation Ad=exp^A*t et Bd=A^-1*(Ad-I)*B
	I=np.identity(4)
	Ad=expm(A*Ts)+0*I*1e-6
	
	#Bd=np.dot(np.linalg.pinv(A),np.dot(Ad-I,B))
	Bd=np.dot(np.dot(Ad,(I-A*Ts/2))*Ts,B)
	A=Ad
	B=Bd
	
	#résolution équatio de Ricciti pour connaitre notre matrice P
	#P=solve_are(A,B,Q,R)
	#calcul de la matrice de retour d'etat K tel que K minimise la fonction de cout 
	#K= np.dot(np.linalg.inv(R),np.dot(np.transpose(B),P))
	K, S, E = dlqr(A, B, Q, R)
	#definir nouvelle valeur/controle a appliquer au torque u=-K*x
	U=np.dot(-K,X)
	#conversion de notre numpy array en commande valide pour notre environnement
	action= np.reshape(U,(1,2))
	action=[action[0,0] ,action[0,1]]
	#envoie de la commande dans la simulation
	observation, reward, terminated, truncated, info = env.step(action)
	time.sleep(0.1)
     
#fin affichage de la situation final avant de fermer le programme  
print(" valeurs final : error_q1 :",error_q1," error_q2 :", error_q2,"terminated:",terminated)
glfw.terminate()
env.close()
