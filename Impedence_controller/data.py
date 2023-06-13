
import numpy as np
import math


def impedance(observation):

	#data robot
	L1=0.1
	L2=0.11
	r=0.01
	mv=1000
	V_tube=np.pi*r*r*L1
	V_sphere=4/3*np.pi*r*r*r
	m1=(V_tube+V_sphere)*mv
	m2=m1
	
	#affectations des valeurs d'observation
	cosq1,cosq2,sinq1,sinq2,x_star,y_star,q1_point,q2_point,x_error,y_error,z_error,sensor=observation

	#matrice mass-spring-damper
	#Mm=np.diag([70,10])
	#Dm=np.diag([8,20])
	#Km=np.diag([30,100])
	
	#Mm=np.diag([10,55])
	#Dm=np.diag([20,8])
	#Km=np.diag([100,12])
	
	#Mm=np.diag([5,5])
	#Dm=np.diag([8,8])
	#Km=np.diag([70,70])
	
	#plutot pas mal
	Mm=np.diag([70,70])
	Dm=np.diag([6,6])
	Km=np.diag([15,15])
	
	#Mm=np.diag([52,54])
	#Dm=np.diag([11,11])
	#Km=np.diag([12,12])
	
	Mm_inv=np.linalg.inv(Mm)

	#calcul des coordonnes articulaire a l'aide des valeurs d'observation
	q1=np.arctan2(sinq1,cosq1)
	q2=np.arctan2(sinq2,cosq2)
	
	#creation des matrices d'etat
	Q=np.array([[q1],[q2]])
	W=np.array([[q1_point],[q2_point]])
	
	#jacobienne analytique ?
	Jee=np.array([[-L1*np.sin(q1)-L2*np.sin(q1+q2) , -L2*np.sin(q1+q2)],
	[L1*np.cos(q1)+L2*np.cos(q1+q2) , L2*np.cos(q1+q2)]])
	#son inverse et sa transposéé
	Jee_inv=np.linalg.inv(Jee)
	J_T=np.transpose(Jee)	
	#ainsi que sa dérivée
	J11=-L2*np.cos(q1+q2)*(q1_point + q2_point)-L1*np.cos(q1)*q1_point
	J12=-L2*np.cos(q1+q2)*(q1_point+ q2_point)
	J21=-L2*np.sin(q1+q2)*(q1_point+ q2_point)-L1*np.sin(q1)*q1_point
	J22=-L2*np.sin(q1+q2)*(q1_point+ q2_point)
	J_dot=np.array([[J11,J12],[J21,J22]])	
	
	#creation matrice des erreur 
	Xe=np.array([[-x_error],[-y_error]])
	X_dot=np.dot(Jee,W) 
	X_dot_des=np.array([[0],[0]])
	Xe_dot=-X_dot
	X_dd_des=np.array([[0],[0]])


	
	#matrice d'inertie du robot M
	M=np.array([[m1*L1**2+m2*(L1**2+2*L1*L2*np.cos(q2)+L2**2),m2*(L1*L2*np.cos(q2)+L2**2)],
	[m2*(L1*L2*np.cos(q2)+L2**2),m2*L2**2]])

	
	# Matrice force coreolis
	C=np.array([[-m2*L1*L2*np.sin(q2)*q2_point,-m2*L1*L2*np.sin(q2)*(q1_point+q2_point)],
	[m2*L1*L2*np.sin(q2)*q1_point,0]])
	#matrice gravité
	G=np.array([[0],[0]])
	#calcul loi de commande
	# on calcul chaque morceau de l'equation séparement
	UA1=np.dot(M,Jee_inv)
	UA2=X_dd_des-np.dot(J_dot,np.dot(Jee_inv,X_dot_des))
	UA=np.dot(UA1,UA2)
	UB=np.dot(C,np.dot(Jee_inv,X_dot_des))
	UC1=np.dot(Dm,Xe_dot)+np.dot(Km,Xe)
	UC=np.dot(J_T,UC1)
	
	U=UA+UB+G+UC
	
	return U
	
	    
		
