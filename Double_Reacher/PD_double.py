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
#calcul des q1,q2 desirée
q1R_star,q2R_star,q1L_star,q2L_star=MGIv2(x_star_target,y_star_target)


# calcul des q1 et q2 actuel
cosq1R,cosq2R,sinq1R,sinq2R,cosq1L,cosq2L,sinq1L,sinq2L=observation[:8]
error_q1R_point,error_q2R_point,error_q1L_point,error_q2L_point=observation[8:12]
xR_error,yR_error,zR_error,xL_error,yL_error,zL_error=observation[12:18]
x_current_target,y_current_target,z_current_target=observation[20:23]
xdot_current_target,ydot_current_target,zdot_current_target=observation[23:26]
Rxdot_current_target,Rydot_current_target,Rzdot_current_target=observation[26:29]
xdd_current_target,ydd_current_target,zdd_current_target=observation[29:32]
Rxdd_current_target,Rydd_current_target,Rzdd_current_target=observation[32:]

# calcul des q1 et q2 actuel
q1R=np.arctan2(sinq1R,cosq1R)
q2R=np.arctan2(sinq2R,cosq2R)
q1L=np.arctan2(sinq1L,cosq1L)
q2L=np.arctan2(sinq2L,cosq2L)

#calcul de l'erreur sur chaque articulation
error_q1R=SmallestSignedAngleBetween(q1R,q1R_star)
error_q2R=SmallestSignedAngleBetween(q2R,q2R_star)
error_q1L=SmallestSignedAngleBetween(q1L,q1L_star)
error_q2L=SmallestSignedAngleBetween(q2L,q2L_star)


# boucle de controle jusqu'à avoir une erreur proche de zéro
while(abs(error_q1R)>1e-4 or abs(error_q2R)>1e-4 or abs(error_q1L)>1e-4 or abs(error_q2L)>1e-4):
	cosq1R,cosq2R,sinq1R,sinq2R,cosq1L,cosq2L,sinq1L,sinq2L=observation[:8]
	error_q1R_point,error_q2R_point,error_q1L_point,error_q2L_point=observation[8:12]
	xR_error,yR_error,zR_error,xL_error,yL_error,zL_error=observation[12:18]
	x_current_target,y_current_target,z_current_target=observation[20:23]
	xdot_current_target,ydot_current_target,zdot_current_target=observation[23:26]
	Rxdot_current_target,Rydot_current_target,Rzdot_current_target=observation[26:29]
	xdd_current_target,ydd_current_target,zdd_current_target=observation[29:32]
	Rxdd_current_target,Rydd_current_target,Rzdd_current_target=observation[32:]

	# calcul des q1 et q2 actuel
	q1R=np.arctan2(sinq1R,cosq1R)
	q2R=np.arctan2(sinq2R,cosq2R)
	q1L=np.arctan2(sinq1L,cosq1L)
	q2L=np.arctan2(sinq2L,cosq2L)

	#calcul de l'erreur sur chaque articulation
	error_q1R=SmallestSignedAngleBetween(q1R,q1R_star)
	error_q2R=SmallestSignedAngleBetween(q2R,q2R_star)
	error_q1L=SmallestSignedAngleBetween(q1L,q1L_star)
	error_q2L=SmallestSignedAngleBetween(q2L,q2L_star)
	#definir nouvelle valeur/controle a appliquer au torque
	U1R=Kp*error_q1R+Kd*-error_q1R_point
	U2R=Kp*error_q2R+Kd*-error_q2R_point
	U1L=Kp*error_q1L+Kd*-error_q1L_point
	U2L=Kp*error_q2L+Kd*-error_q2L_point
	action= [U1R ,U2R ,U1L ,U2L]
	#envoie de la commande dans l'environnement 
	observation, reward, terminated, truncated, info = env.step(action)
	time.sleep(0.1)

glfw.terminate()
env.close()


