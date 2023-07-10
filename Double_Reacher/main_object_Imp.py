import math
import random
import gym
import numpy as np
import mujoco_py
import glfw
import time
import os
import matplotlib.pyplot as plt
from object_impedance import object_impedance
from object_impedance_v2 import object_impedance_2





tmp = open("contact_log.txt", "w")
step=0
Erreur=False
# Para visualizar que es lo que hace el agente
glfw.init()
test_steps = 1000
env = gym.make("Double_Reacher", render_mode="human") # brazo 2 DOF
observation, info = env.reset(random=3)
contact=False
score=0;
#force=[-100,0,0,0,0,0]
#body_id=2
#env.add_external_force(force,body_id)
i=5
# recuperation des X et Y de la cible
x_star_target,y_star_target=observation[18:20]
x_star_target,y_star_target=0,0
Xdd_previous=np.array([[0],[0],[0],[0],[0],[0]])
F_previous=np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]])
# boucle de controle jusqu'à avoir une erreur proche de zéro
while(not Erreur):
	step+=1
	tmp.write("\n step :"+ str(step))
	contact_R,contact_L=env.contact()
	U,Xdd_previous,F_previous=object_impedance_2(observation,x_star_target,y_star_target,Xdd_previous,F_previous,contact_R,contact_L)
	action= np.reshape(U,(1,4))
	action=[action[0,0] ,action[0,1],action[0,2],action[0,3]]
	contact_R,contact_L=env.contact()
	tmp.write("\n contact droite:"+ str(contact_R))
	tmp.write("\n contact gauche"+str(contact_L))
	tmp.write("\n ################")
	if(action[0]<1 and action[0]!=0):
	#envoie de la commande dans l'environnement 
		observation, reward, terminated, truncated, info = env.step(action)
		time.sleep(0.1)
	else:
		Erreur=True
	i-=1
	if(i==0):
		force=[0,0,0,0,0,0]
		body_id=7
		env.add_external_force(force,body_id)

	
tmp.close()
glfw.terminate()
env.close()
