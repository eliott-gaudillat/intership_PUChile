import numpy as np
import math


def object_impedance(observation,x_star_target,y_star_target,Xdd_previous):
	#data robot et object
	V_object=.01*.01*.01
	mV_object=1000	
	m=V_object*mV_object
	
	L1=0.1
	L2=0.11
	r=0.01
	mv=1000
	V_tube=np.pi*r*r*L1
	V_sphere=4/3*np.pi*r*r*r
	m1=(V_tube+V_sphere)*mv
	m2=m1
	
	#data observation
	cosq1R,cosq2R,sinq1R,sinq2R,cosq1L,cosq2L,sinq1L,sinq2L=observation[:8]
	q1R_point,q2R_point,q1L_point,q2L_point=observation[8:12]
	xR_error,yR_error,zR_error,xL_error,yL_error,zL_error=observation[12:18]
	x_current_target,y_current_target,z_current_target=observation[20:23]
	xdot_current_target,ydot_current_target,zdot_current_target=observation[23:26]
	Rxdot_current_target,Rydot_current_target,Rzdot_current_target=observation[26:29]
	xdd_current_target,ydd_current_target,zdd_current_target=observation[29:32]
	Rxdd_current_target,Rydd_current_target,Rzdd_current_target=observation[32:]
	
	q1R=np.arctan2(sinq1R,cosq1R)
	q2R=np.arctan2(sinq2R,cosq2R)
	q1L=np.arctan2(sinq1L,cosq1L)
	q2L=np.arctan2(sinq2L,cosq2L)
	
	Qd=np.array([[q1R_point],[q2R_point],[q1L_point],[q2L_point]])


	e=np.array([[x_star_target-x_current_target],[y_star_target-y_current_target]])
	ed=np.array([[-xdot_current_target],[-ydot_current_target]])


#Xdd_cmd=Iinv * [Fext+Fimp-B]
#ici B=0
#I inertia matrix of the object relative to it c.m for {bi}
	
	Md=np.array([[15,0],[0,15]])
	Kp=400
	Kd=200
	F_ext_estime=Xdd_previous
	F_imp=Kp*e+Kd*ed
	F=F_ext_estime+F_imp
	Md_inv=np.linalg.inv(Md)
	Xdd_cmd=np.dot(Md_inv,F)


#F_cmd
	G=np.array([[0],[0]])
	Mo=np.array([[m,0],[0,m]])
	Q=np.diag([m,m,m,m])
	Q_inv=np.linalg.inv(Q)
	W=np.array([[1,0,1,0],[0,1,0,1]])
	W_T=np.transpose(W)
	
	W3=np.linalg.inv(np.dot(W,np.dot(Q_inv,W_T)))
	W_pseudo_inv=np.dot(np.dot(Q_inv,W_T),W3)
	F_cmd=np.dot(W_pseudo_inv,(G-F_ext_estime+np.dot(Mo,(np.dot(Md_inv,F)))))

#a_cmd
#ici absence de rotation donc
	A_cmd=np.vstack((Xdd_cmd,Xdd_cmd))

#qdd_cmd
	Jee=np.array([[-L1*np.sin(q1R)-L2*np.sin(q1R+q2R) , -L2*np.sin(q1R+q2R),0,0],
	[L1*np.cos(q1R)+L2*np.cos(q1R+q2R) , L2*np.cos(q1R+q2R),0,0],
	[0,0,-L1*np.sin(q1L)-L2*np.sin(q1L+q2L) , -L2*np.sin(q1L+q2L)],
	[0,0,L1*np.cos(q1L)+L2*np.cos(q1L+q2L) , L2*np.cos(q1L+q2L)]])
	# sa transposéé
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
	A=np.dot(J_dot,Qd)
	qdd_cmd=np.dot(Jee_inv,(A_cmd-A))


#torque_i=Mi*qdd_icmd+C(qi,qdi)+Ji^t*fi_cmd
	
	
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
	
	C=np.dot(C,Qd)
	
	U=np.dot(M,qdd_cmd)+C+np.dot(J_T,F_cmd)
	print("U:",U)
	return [U,Xdd_cmd]
