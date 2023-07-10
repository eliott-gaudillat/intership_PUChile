
import numpy as np
import math


def impedanceTotal(observationR,observationL):

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
	cosq1R,cosq2R,sinq1R,sinq2R,x_current_target,y_current_target,q1R_point,q2R_point,xR_error,yR_error,zR_error=observationR
	cosq1L,cosq2L,sinq1L,sinq2L,x_current_target,y_current_target,q1L_point,q2L_point,xL_error,yL_error,zL_error=observationL

	
	#plutot pas mal
	Mm=np.diag([70,70,70,70])
	Dm=np.diag([6,6,6,6])
	Km=np.diag([15,15,15,15])

	
	Mm_inv=np.linalg.inv(Mm)

	#calcul des coordonnes articulaire a l'aide des valeurs d'observation
	q1R=np.arctan2(sinq1R,cosq1R)
	q2R=np.arctan2(sinq2R,cosq2R)
	q1L=np.arctan2(sinq1L,cosq1L)
	q2L=np.arctan2(sinq2L,cosq2L)

	
	#creation des matrices d'etat
	Q=np.array([[q1R],[q2R],[q1L],[q2L]])
	W=np.array([[q1R_point],[q2R_point],[q1L_point],[q2L_point]])
	
	#jacobienne analytique ?
	Jee=np.array([[-L1*np.sin(q1R)-L2*np.sin(q1R+q2R) , -L2*np.sin(q1R+q2R),0,0],
	[L1*np.cos(q1R)+L2*np.cos(q1R+q2R) , L2*np.cos(q1R+q2R),0,0],
	[0,0,-L1*np.sin(q1L)-L2*np.sin(q1L+q2L) , -L2*np.sin(q1L+q2L)],
	[0,0,L1*np.cos(q1L)+L2*np.cos(q1L+q2L) , L2*np.cos(q1L+q2L)]])
	#son inverse et sa transposéé
	Jee_inv=np.linalg.inv(Jee)
	J_T=np.transpose(Jee)	
	#ainsi que sa dérivée
	J11=-L2*np.cos(q1R+q2R)*(q1R_point + q2R_point)-L1*np.cos(q1R)*q1R_point
	J12=-L2*np.cos(q1R+q2R)*(q1R_point+ q2R_point)
	J21=-L2*np.sin(q1R+q2R)*(q1R_point+ q2R_point)-L1*np.sin(q1R)*q1R_point
	J22=-L2*np.sin(q1R+q2R)*(q1R_point+ q2R_point)
	J33=-L2*np.cos(q1L+q2L)*(q1L_point + q2L_point)-L1*np.cos(q1L)*q1L_point
	J34=-L2*np.cos(q1L+q2L)*(q1L_point+ q2L_point)
	J43=-L2*np.sin(q1L+q2L)*(q1L_point+ q2L_point)-L1*np.sin(q1L)*q1L_point
	J44=-L2*np.sin(q1L+q2L)*(q1L_point+ q2L_point)
	
	J_dot=np.array([[J11,J12,0,0],[J21,J22,0,0],[0,0,J33,J34],[0,0,J43,J44]])	
	
	#creation matrice des erreur 
	Xe=np.array([[-xR_error],[-yR_error],[-xL_error],[-yL_error]])
	X_dot=np.dot(Jee,W) 
	X_dot_des=np.array([[0],[0],[0],[0]])
	Xe_dot=-X_dot
	X_dd_des=np.array([[0],[0],[0],[0]])


	
	#matrice d'inertie du robot M
	M=np.array([[m1*L1**2+m2*(L1**2+2*L1*L2*np.cos(q2R)+L2**2),m2*(L1*L2*np.cos(q2R)+L2**2),0,0],
	[m2*(L1*L2*np.cos(q2R)+L2**2),m2*L2**2,0,0],
	[0,0,m1*L1**2+m2*(L1**2+2*L1*L2*np.cos(q2L)+L2**2),m2*(L1*L2*np.cos(q2L)+L2**2)],
	[0,0,m2*(L1*L2*np.cos(q2L)+L2**2),m2*L2**2]])

	
	# Matrice force coreolis
	C=np.array([[-m2/2*L1*L2*np.sin(q2R)*q2R_point,-m2*L1*L2*np.sin(q2R)*(q1R_point/2-q2R_point),0,0],
	[m2*L1*L2*np.sin(q2R)*q1R_point,0,0,0],
	[0,0,-m2/2*L1*L2*np.sin(q2L)*q2L_point,-m2*L1*L2*np.sin(q2L)*(q1L_point/2-q2L_point)],
	[0,0,m2*L1*L2*np.sin(q2L)*q1L_point,0]])
	#matrice gravité
	G=np.array([[0],[0],[0],[0]])
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
	
	    
		
