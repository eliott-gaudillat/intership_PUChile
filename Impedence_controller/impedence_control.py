import math
import random
import gym
import numpy as np
import mujoco_py
import glfw
import time
import os
import matplotlib.pyplot as plt
from data import  impedance

# Para visualizar que es lo que hace el agente
glfw.init()
test_steps = 1000
env = gym.make("Reacher-v4_perso", render_mode="human") # brazo 2 DOF
observation, info = env.reset(random=3)
add_force=False
nb_step=0
tmp = open("sensor_log.txt", "w")
cosq1,cosq2,sinq1,sinq2,x_star,y_star,q1_point,q2_point,x_error,y_error,z_error,sensor=observation
U=impedance(observation)
action= np.reshape(U,(1,2))
U1_log=action[0,0]
U2_log=action[0,1]
step=1
# boucle de controle 
while((abs(x_error)>5e-4 or abs(y_error)>5e-4)):
	#definir nouvelle valeur/controle a appliquer au torque
	U=impedance(observation)
	cosq1,cosq2,sinq1,sinq2,x_star,y_star,q1_point,q2_point,x_error,y_error,z_error,sensor=observation
	#envoie de la commande dans l'environnement 
	#conversion de notre numpy array en commande valide pour notre environnement
	action= np.reshape(U,(1,2))
	U1_log=np.append(U1_log,action[0,0])
	U2_log=np.append(U2_log,action[0,1])
	action=[action[0,0] ,action[0,1]]
	#action=[0,0]
	observation, reward, terminated, truncated, info = env.step(action)
	step+=1
	time.sleep(0.1)
	#if((abs(x_error)<1e-3 and abs(y_error)<1e-3)):
	#	print("cible atteinte")
	#	force=[700,300,0,0,0,0]
	#	body_id=1
	#	env.add_external_force(force,body_id)
	#	add_force=True
	#	nb_step=50
	if(add_force):
		nb_step-=1
		if(nb_step==0):
			body_id=1
			force=[0,0,0,0,0,0]
			env.data.xfrc_applied[body_id]=force
			add_force=False
	if(sensor>0):
		value=env.contact()
		#tmp.write("\n"+value)


time=np.arange(step)
plt.figure(1)
plt.plot(time, U1_log)
plt.title('Data log U1')
plt.savefig("plot_U1")
plt.figure(2)
plt.plot(time, U2_log)
plt.title('Data log U2')
plt.savefig("plot_U2")				
tmp.close()
glfw.terminate()
env.close()


