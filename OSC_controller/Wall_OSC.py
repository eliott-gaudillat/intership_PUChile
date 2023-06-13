import math
import random
import gym
import numpy as np
import mujoco_py
import glfw
import time
import os
from jacobienne import osc
from jacobienne import filtre
from jacobienne import add_mesure







# Para visualizar que es lo que hace el agente
glfw.init()
test_steps = 1000
env = gym.make("Reacher-v4_perso", render_mode="human") # brazo 2 DOF
observation, info = env.reset(random=3)
cosq1,cosq2,sinq1,sinq2,x_star,y_star,q1_point,q2_point,x_error,y_error=observation[0:10]
sample_q1_point=np.array([[q1_point,q1_point,q1_point]])
sample_q2_point=np.array([[q2_point,q2_point,q2_point]])
N=3
i=0
# boucle de controle jusqu'à avoir une erreur proche de zéro
while((abs(x_error)>5e-4 or abs(y_error)>5e-4)):

	# calcul des q1 et q2 actuel
	cosq1,cosq2,sinq1,sinq2,x_star,y_star,q1_point,q2_point,x_error,y_error=observation[0:10]
	sample_q1_point=add_mesure(sample_q1_point,q1_point)
	sample_q2_point=add_mesure(sample_q2_point,q2_point)

	
	q1=np.arctan2(sinq1,cosq1)
	q2=np.arctan2(sinq2,cosq2)
	q1_point_filtered,q2_point_filtered=filtre(sample_q1_point,sample_q2_point,N)
	#print("q1 fitre:",q1_point_filtered,"mesure:",q1_point)
	#print("q2 fitre:",q2_point_filtered,"mesure:",q2_point)
	
	
	U=osc(q1,q2,q1_point,q2_point,-x_error,-y_error)
	U2=osc(q1,q2,q1_point_filtered,q2_point_filtered,-x_error,-y_error)
	#conversion de notre numpy array en commande valide pour notre environnement
	action= np.reshape(U,(1,2))
	action=[action[0,0] ,action[0,1]]
	action2= np.reshape(U2,(1,2))
	action2=[action2[0,0] ,action2[0,1]]
	
	if(action[0]>1):
		action[0]=1
	elif(action[0]<-1):
		action[0]=-1	
	if(action[1]>1):
		action[1]=1
	elif(action[1]<-1):
		action[1]=-1	
	print(action)
	print("error",x_error,y_error)
	#envoie de la commande dans l'environnement 
	observation, reward, terminated, truncated, info = env.step(action)
	time.sleep(0.1)
     
  
print(" valeurs final : error_x :",x_error," error_y :", y_error,"terminated:",terminated)
glfw.terminate()
env.close()
