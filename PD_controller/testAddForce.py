import math
import random
import gym
import numpy as np
import mujoco_py
import glfw
import time
import os
from jacobienne import MGIv2
from jacobienne import SmallestSignedAngleBetween









# Para visualizar que es lo que hace el agente
glfw.init()
test_steps = 1000
env = gym.make("Reacher-v4_perso", render_mode="human") # brazo 2 DOF
observation, info = env.reset(random=1)

Kp=0.04
Kd=0.02
# recuperation des X et Y de la cible
x_star=observation[4]
y_star=observation[5]
#calcul des q1,q2 desirée
q1_star,q2_star=MGIv2(x_star,y_star)

# calcul des q1 et q2 actuel
cosq1=observation[0]
cosq2=observation[1]
sinq1=observation[2]
sinq2=observation[3]

q1=np.arctan2(sinq1,cosq1)
q2=np.arctan2(sinq2,cosq2)

#calcul de l'erreur sur chaque articulation
error_q1=SmallestSignedAngleBetween(q1_star,q1)
error_q2=SmallestSignedAngleBetween(q2_star,q2)
score=0;
add_force=False
# boucle de controle jusqu'à avoir une erreur proche de zéro
while(True):
	# calcul des q1 et q2 actuel
	cosq1,cosq2,sinq1,sinq2,x_star,y_star,error_q1_point,error_q2_point,x_error,y_error,z_error,sensor=observation
	q1=np.arctan2(sinq1,cosq1)
	q2=np.arctan2(sinq2,cosq2)
	#calcul de l'erreur sur chaque articulation
	error_q1=q1_star-q1
	error_q2=q2_star-q2
	#definir nouvelle valeur/controle a appliquer au torque
	U1=Kp*error_q1+Kd*-error_q1_point
	U2=Kp*error_q2+Kd*-error_q2_point
	action= [U1 ,U2]
	#envoie de la commande dans l'environnement 
	observation, reward, terminated, truncated, info = env.step(action)
	time.sleep(0.1)
	score+=reward
	if((abs(error_q1)<1e-4 or abs(error_q2)<1e-4)):
		force=[0,10,0,0,0,0]
		body_id=3
		env.add_external_force(force,body_id)
		add_force=True
		print(env.data.xfrc_applied)
	if(sensor>0):
		env.contact()	
	if(add_force):
		body_id=3
		force=[0,-0.5,0,0,0,0]
		#print(action)
		#print("actuator_force:",env.data.actuator_force)
		env.data.xfrc_applied[body_id]+=force
		
	
     
  
print(" valeurs final : error_q1 :",error_q1," error_q2 :", error_q2,"score",score)
glfw.terminate()
env.close()




   # Control (con red ya entrenada)
    # (obs) -->NN--> accion
    
    # Training 
    # (obs, reward) -->NN--> accion
    #print(observation)
    
