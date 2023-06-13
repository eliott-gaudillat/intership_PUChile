import numpy as np
import math


def osc(q1,q2,w1,w2,x_error,y_error):
	#data robot
	L1=0.1
	L2=0.11
	r=0.01
	mv=1000
	V_tube=np.pi*r*r*L1
	V_sphere=4/3*np.pi*r*r*r
	m1=(V_tube+V_sphere)*mv
	m2=m1
	#Kp=25
	Kp1=10.8
	Kp2=6
	Kd1=5
	Kd2=2
	Kp=np.array([[Kp1],[Kp2]])
	Kd=np.array([[Kd1],[Kd2]])
	W=np.array([[w1],[w2]])
	
	#jacobienne a l' OT
	
	Jee=np.array([[-L1*np.sin(q1)-L2*np.sin(q1+q2) , -L2*np.sin(q1+q2)],
	[L1*np.cos(q1)+L2*np.cos(q1+q2) , L2*np.cos(q1+q2)]])
	
	
	#position et vitesse du robot
	X=np.array([[L1*np.cos(q1)+L2*np.cos(q1+q2)],[L1*np.sin(q1)+L2*np.sin(q1+q2)]])
	
	X_point=np.dot(Jee,W) 
	
	#matrice d'inertie du robot
	M=np.array([[m1*L1**2+m2*(L1**2+2*L1*L2*np.cos(q2)+L2**2),m2*(L1*L2*np.cos(q2)+L2**2)],
	[m2*(L1*L2*np.cos(q2)+L2**2),m2*L2**2]])
	
	#matrice d'inertie au niveau de l'OT
	#Mxee=np.linalg.inv(np.dot(Jee,np.dot(np.linalg.inv(M),np.transpose(Jee))))
	J_T=np.transpose(Jee)
	#J_inv=np.linalg.inv(Jee)
	#J_t_inv=np.transpose(J_inv)
	#Mxee=np.dot(J_t_inv,np.dot(M,J_inv)) 
	#version matlab
	Mxee2=np.array([[(L2**4*np.cos(q1 + q2)**2 + L2**2*m2*np.cos(q1 + q2)**2 - L2**4*m2*np.cos(q1 + q2)**2 + L1**2*m2*np.cos(q1)**2 + L1**2*L2**2*np.cos(q1)**2 + L1**2*L2**2*m1*np.cos(q1 + q2)**2 + L1**2*L2**2*m2*np.cos(q1 + q2)**2 + 2*L1*L2**3*np.cos(q1 + q2)*np.cos(q1) - 2*L1*L2**3*m2*np.cos(q1 + q2)*np.cos(q1) + 2*L1*L2*m2*np.cos(q1 + q2)*np.cos(q1) - 2*L1**2*L2**2*m2*np.cos(q1 + q2)*np.cos(q1)*np.cos(q2))/(L1**2*L2**2*np.cos(q1 + q2)**2*np.sin(q1)**2 + L1**2*L2**2*np.sin(q1 + q2)**2*np.cos(q1)**2 - 2*L1**2*L2**2*np.cos(q1 + q2)*np.sin(q1 + q2)*np.cos(q1)*np.sin(q1)),(L2**4*np.cos(q1 + q2)*np.sin(q1 + q2) + L1**2*m2*np.cos(q1)*np.sin(q1) + L1**2*L2**2*np.cos(q1)*np.sin(q1) + L2**2*m2*np.cos(q1 + q2)*np.sin(q1 + q2) - L2**4*m2*np.cos(q1 + q2)*np.sin(q1 + q2) + L1*L2**3*np.cos(q1 + q2)*np.sin(q1) + L1*L2**3*np.sin(q1 + q2)*np.cos(q1) - L1*L2**3*m2*np.cos(q1 + q2)*np.sin(q1) - L1*L2**3*m2*np.sin(q1 + q2)*np.cos(q1) + L1**2*L2**2*m1*np.cos(q1 + q2)*np.sin(q1 + q2) + L1**2*L2**2*m2*np.cos(q1 + q2)*np.sin(q1 + q2) + L1*L2*m2*np.cos(q1 + q2)*np.sin(q1) + L1*L2*m2*np.sin(q1 + q2)*np.cos(q1) - L1**2*L2**2*m2*np.cos(q1 + q2)*np.cos(q2)*np.sin(q1) - L1**2*L2**2*m2*np.sin(q1 + q2)*np.cos(q1)*np.cos(q2))/(L1**2*L2**2*np.cos(q1 + q2)**2*np.sin(q1)**2 + L1**2*L2**2*np.sin(q1 + q2)**2*np.cos(q1)**2 - 2*L1**2*L2**2*np.cos(q1 + q2)*np.sin(q1 + q2)*np.cos(q1)*np.sin(q1))],[(L2**4*np.cos(q1 + q2)*np.sin(q1 + q2) + L1**2*m2*np.cos(q1)*np.sin(q1) + L1**2*L2**2*np.cos(q1)*np.sin(q1) + L2**2*m2*np.cos(q1 + q2)*np.sin(q1 + q2) - L2**4*m2*np.cos(q1 + q2)*np.sin(q1 + q2) + L1*L2**3*np.cos(q1 + q2)*np.sin(q1) + L1*L2**3*np.sin(q1 + q2)*np.cos(q1) - L1*L2**3*m2*np.cos(q1 + q2)*np.sin(q1) - L1*L2**3*m2*np.sin(q1 + q2)*np.cos(q1) + L1**2*L2**2*m1*np.cos(q1 + q2)*np.sin(q1 + q2) + L1**2*L2**2*m2*np.cos(q1 + q2)*np.sin(q1 + q2) + L1*L2*m2*np.cos(q1 + q2)*np.sin(q1) + L1*L2*m2*np.sin(q1 + q2)*np.cos(q1) - L1**2*L2**2*m2*np.cos(q1 + q2)*np.cos(q2)*np.sin(q1) - L1**2*L2**2*m2*np.sin(q1 + q2)*np.cos(q1)*np.cos(q2))/(L1**2*L2**2*np.cos(q1 + q2)**2*np.sin(q1)**2 + L1**2*L2**2*np.sin(q1 + q2)**2*np.cos(q1)**2 - 2*L1**2*L2**2*np.cos(q1 + q2)*np.sin(q1 + q2)*np.cos(q1)*np.sin(q1)),(L2**4*np.sin(q1 + q2)**2 + L2**2*m2*np.sin(q1 + q2)**2 - L2**4*m2*np.sin(q1 + q2)**2 + L1**2*m2*np.sin(q1)**2 + L1**2*L2**2*np.sin(q1)**2 + L1**2*L2**2*m1*np.sin(q1 + q2)**2 + L1**2*L2**2*m2*np.sin(q1 + q2)**2 + 2*L1*L2**3*np.sin(q1 + q2)*np.sin(q1) - 2*L1*L2**3*m2*np.sin(q1 + q2)*np.sin(q1) + 2*L1*L2*m2*np.sin(q1 + q2)*np.sin(q1) - 2*L1**2*L2**2*m2*np.sin(q1 + q2)*np.cos(q2)*np.sin(q1))/(L1**2*L2**2*np.cos(q1 + q2)**2*np.sin(q1)**2 + L1**2*L2**2*np.sin(q1 + q2)**2*np.cos(q1)**2 - 2*L1**2*L2**2*np.cos(q1 + q2)*np.sin(q1 + q2)*np.cos(q1)*np.sin(q1))]])
	
	
	#calcul error position and velocity
	error_position=np.array([[x_error],[y_error]])
	error_velocity=-X_point

	# x_point_point defini par un controlleur PD  kp(x_des-x)+kv(x_point_des-x_point)
	X_point_point=Kp*error_position+Kd*error_velocity
	#X_point_point=Kp*error_position
	
	#F=Meex*x_point_point
	Fx=np.dot(Mxee2,X_point_point)
	
	#u=JeeT*Fx
	
	U=np.dot(J_T,Fx)
	
	return U
	

def add_mesure(previous_mesure,mesure):
	#decaler fenetre de mesure
	previous_mesure=np.roll(previous_mesure,-1)
	#ajout de la  nouvelle mesure a la fin de la fenetre
	previous_mesure[0,-1]=mesure
	
	return previous_mesure
	
def filtre(q1_point_noisy,q2_point_noisy,N):
	q1_point_sum=np.sum(q1_point_noisy)
	q2_point_sum=np.sum(q2_point_noisy)
	
	q1_point_filtered=q1_point_noisy[0,2]*0.8+q1_point_noisy[0,1]*0.1+q1_point_noisy[0,0]*0.1
	q2_point_filtered=q2_point_noisy[0,2]*0.8+q2_point_noisy[0,1]*0.1+q2_point_noisy[0,0]*0.1
	
		
	return q1_point_filtered,q2_point_filtered

	
	

	
	

		
	
