import math
import random
import gym
import numpy as np
import mujoco_py
import glfw
import time
import os
import matplotlib.pyplot as plt








# Para visualizar que es lo que hace el agente
glfw.init()
test_steps = 1000
env = gym.make("Double_Reacher", render_mode="human") # brazo 2 DOF
observation, info = env.reset()

# boucle de controle jusqu'à avoir une erreur proche de zéro
while(1):

	action= [0 ,0,0,0]
	#envoie de la commande dans l'environnement 
	observation, reward, terminated, truncated, info = env.step(action)
	cosq1R,cosq2R,sinq1R,sinq2R,cosq1L,cosq2L,sinq1L,sinq2L=observation[:8]
	error_q1R_point,error_q2R_point,error_q1L_point,error_q2L_point=observation[8:12]
	xR_error,yR_error,zR_error,xL_error,yL_error,zL_error=observation[12:18]
	x_star_target,y_star_target=observation[18:20]
	x_current_target,y_current_target,z_current_target=observation[20:23]
	xdot_current_target,ydot_current_target,zdot_current_target=observation[23:26]
	Rxdot_current_target,Rydot_current_target,Rzdot_current_target=observation[26:29]
	xdd_current_target,ydd_current_target,zdd_current_target=observation[29:32]
	Rxdd_current_target,Rydd_current_target,Rzdd_current_target=observation[32:]
	#print('x:',x_current_target,'y:',y_current_target,'z:',z_current_target)
	#print('xd:',xdot_current_target,'yd:',ydot_current_target,'zd:',zdot_current_target)
	#print('Rxd:',Rxdot_current_target,'Ryd:',Rydot_current_target,'Rzd:',Rzdot_current_target)
	#print('xdd:',xdd_current_target,'ydd:',ydd_current_target,'zdd:',zdd_current_target)
	#print('Rxdd:',Rxdd_current_target,'Rydd:',Rydd_current_target,'Rzdd:',Rzdd_current_target)
	#print('xstar:',x_star_target,'ystar:',y_star_target)
 
   
glfw.terminate()
env.close()


