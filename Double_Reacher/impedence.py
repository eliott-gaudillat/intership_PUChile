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

# calcul des q1 et q2 actuel


# boucle de controle jusqu'à avoir une erreur proche de zéro
while(1):
	#definir nouvelle valeur/controle a appliquer au torque
	U=impedance(observation)
	action= np.reshape(U,(1,4))
	action=[action[0,0] ,action[0,1],action[0,2],action[0,3]]
	#envoie de la commande dans l'environnement 
	observation, reward, terminated, truncated, info = env.step(action)
	time.sleep(0.1)

glfw.terminate()
env.close()
