import math
import random
import gym
import numpy as np
import mujoco_py
import glfw
import time
import os
import matplotlib.pyplot as plt
from jacobienne import MGIv2
from jacobienne import SmallestSignedAngleBetween
from data import  impedance
from impDouble import  impedanceTotal
# Para visualizar que es lo que hace el agente
glfw.init()
test_steps = 1000
env = gym.make("Double_Reacher", render_mode="human") # brazo 2 DOF
observation, info = env.reset(random=3)
Kp=0.04
Kd=0.02
contact=False
score=0;
# recuperation des X et Y de la cible
x_star_target,y_star_target=observation[18:20]

tmp = open("contact_log.txt", "w")
i=200
# calcul des q1 et q2 actuel
print(observation)

# boucle de controle jusqu'à avoir une erreur proche de zéro
while(i!=0):

	cosq1R,cosq2R,sinq1R,sinq2R,cosq1L,cosq2L,sinq1L,sinq2L=observation[:8]
	error_q1R_point,error_q2R_point,error_q1L_point,error_q2L_point=observation[8:12]
	xR_error,yR_error,zR_error,xL_error,yL_error,zL_error=observation[12:18]
	x_current_target,y_current_target,z_current_target=observation[20:23]
	xdot_current_target,ydot_current_target,zdot_current_target=observation[23:26]
	Rxdot_current_target,Rydot_current_target,Rzdot_current_target=observation[26:29]
	xdd_current_target,ydd_current_target,zdd_current_target=observation[29:32]
	Rxdd_current_target,Rydd_current_target,Rzdd_current_target=observation[32:]
	
	observationR=[cosq1R,cosq2R,sinq1R,sinq2R,x_current_target,y_current_target,error_q1R_point,error_q2R_point,xR_error,yR_error,zR_error]
	observationL=[cosq1L,cosq2L,sinq1L,sinq2L,x_current_target,y_current_target,error_q1L_point,error_q2L_point,xL_error,yL_error,zL_error]
	observationObject=[x_star_target,y_star_target,x_current_target,y_current_target,z_current_target,xdot_current_target,ydot_current_target,zdot_current_target]

	#definir nouvelle valeur/controle a appliquer au torque
	#UR=impedance(observationR)
	#UL=impedance(observationL)
	#actionR= np.reshape(UR,(1,2))
	#actionL= np.reshape(UL,(1,2))
	#action=[actionR[0,0] ,actionR[0,1],actionL[0,0],actionL[0,1]]
	
	U=impedanceTotal(observationR,observationL)
	
	q1R=np.arctan2(sinq1R,cosq1R)
	q2R=np.arctan2(sinq2R,cosq2R)
	q1L=np.arctan2(sinq1L,cosq1L)
	q2L=np.arctan2(sinq2L,cosq2L)
	actionT= np.reshape(U,(1,4))
	actionT=[actionT[0,0] ,actionT[0,1],actionT[0,2],actionT[0,3]]
	#actionT=[0 ,0,actionT[0,2],actionT[0,3]]
	
	contact_R,contact_L=env.contact()
	tmp.write("\n contact droite:"+ str(contact_R))
	tmp.write("\n contact gauche"+str(contact_L))
	tmp.write("\n ################")
	#i-=1
	#if(i==0):
	#	force=[1000,0,0,0,0,0]
	#	body_id=7
	#s	env.add_external_force(force,body_id)
	
		

	#envoie de la commande dans l'environnement 
	i-=1
	observation, reward, terminated, truncated, info = env.step(actionT)
	time.sleep(0.1)
tmp.close()
glfw.terminate()
env.close()
