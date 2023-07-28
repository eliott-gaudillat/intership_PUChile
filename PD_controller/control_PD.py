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
env = gym.make("Reacher-v4", render_mode="human") # brazo 2 DOF
observation, info = env.reset()

Kp=0.04
Kd=0.02

Kp=2
Kd=0.04

# recuperation des X et Y de la cible
x_star=observation[4]
y_star=observation[5]
#calcul des q1,q2 desirée
q1_star,q2_star=MGIv2(x_star,y_star)
# calcul des q1 et q2 actuel
cosq1,cosq2,sinq1,sinq2,x_star,y_star,error_q1_point,error_q2_point,x_error,y_error,z_error=observation

q1=np.arctan2(sinq1,cosq1)
q2=np.arctan2(sinq2,cosq2)

#calcul de l'erreur sur chaque articulation
error_q1=SmallestSignedAngleBetween(q1_star,q1)
error_q2=SmallestSignedAngleBetween(q2_star,q2)
score=0;
#calcul de l'erreur sur chaque articulation
error_q1=q1_star-q1
error_q2=q2_star-q2
#definir nouvelle valeur/controle a appliquer au torque
U1_log=Kp*error_q1+Kd*-error_q1_point
U2_log=Kp*error_q2+Kd*-error_q2_point
error_q1_log=error_q1
error_q2_log=error_q2
step=1
# boucle de controle jusqu'à avoir une erreur proche de zéro
while((abs(error_q1)>1e-3 or abs(error_q2)>1e-3)):
	# calcul des q1 et q2 actuel
	cosq1,cosq2,sinq1,sinq2,x_star,y_star,error_q1_point,error_q2_point,x_error,y_error,z_error=observation
	q1=np.arctan2(sinq1,cosq1)
	q2=np.arctan2(sinq2,cosq2)

	#calcul de l'erreur sur chaque articulation
	error_q1=q1_star-q1
	error_q2=q2_star-q2
	#definir nouvelle valeur/controle a appliquer au torque
	U1=Kp*error_q1+Kd*-error_q1_point
	U2=Kp*error_q2+Kd*-error_q2_point
	U1_log=np.append(U1_log,U1)
	U2_log=np.append(U2_log,U2)
	error_q1_log=np.append(error_q1_log,error_q1)
	error_q2_log=np.append(error_q2_log,error_q2)
	action= [U1 ,U2]
	#envoie de la commande dans l'environnement 
	observation, reward, terminated, truncated, info = env.step(action)
	time.sleep(0.1)
	score+=reward
	step+=1
 
    
print(step)
time=np.arange(step)
plt.figure(1)
plt.plot(time, U1_log)
plt.title('Data log U1')
plt.savefig("plot_U1")
plt.figure(2)
plt.plot(time, U2_log)
plt.title('Data log U2')
plt.savefig("plot_U2")
plt.figure(3)
plt.plot(time, error_q1_log)
plt.title('Data log error_q1')
plt.savefig("plot_eq1")
plt.figure(4)
plt.plot(time, error_q2_log)
plt.title('Data log error_q2')
plt.savefig("plot_eq2")

print(" valeurs final : error_q1 :",error_q1," error_q2 :", error_q2,"score",score)
glfw.terminate()
env.close()
