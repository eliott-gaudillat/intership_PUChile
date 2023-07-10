import numpy as np
import math


def object_impedance_2(observation,x_star_target,y_star_target,Xdd_previous,F_previous,contact_R,contact_L):
	#data robot et object
	V_object=.01*.01*.01
	mV_object=1000	
	m=V_object*mV_object
	g=-9.81
	
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
	
	#contact_R,contact_L forme (pos,frame,res)
	
	q1R=np.arctan2(sinq1R,cosq1R)
	q2R=np.arctan2(sinq2R,cosq2R)
	q1L=np.arctan2(sinq1L,cosq1L)
	q2L=np.arctan2(sinq2L,cosq2L)
	
	Qd=np.array([[q1R_point],[q2R_point],[q1L_point],[q2L_point]])

	#contact point
	if(len(contact_R)==0):
		p1=np.array([[0],[0],[0]])
	else:
		p1=np.array([[contact_R[0][0][0]-x_current_target],[contact_R[0][0][1]-y_current_target],[contact_R[0][0][2]-z_current_target]])
	if(len(contact_L)==0):
		p2=np.array([[0],[0],[0]])
	else:
		p2=np.array([[contact_L[0][0][0]-x_current_target],[contact_L[0][0][1]-y_current_target],[contact_L[0][0][2]-z_current_target]])
	e=np.array([[x_star_target-x_current_target],[y_star_target-y_current_target],[0.01-z_current_target],[0],[0],[0]])
	ed=np.array([[-xdot_current_target],[-ydot_current_target],[-zdot_current_target],[-Rxdot_current_target],[-Rydot_current_target],[-Rzdot_current_target]])
	
	W_object=np.array([[Rxdot_current_target],[Rydot_current_target],[Rzdot_current_target]])
	Wd_object=np.array([[Rxdd_current_target],[Rydd_current_target],[Rzdd_current_target]])
	P1=np.array([[0,-p1[2,0],p1[1,0]],[p1[2,0],0,-p1[0,0]],[-p1[1,0],-p1[0,0],0]])
	P2=np.array([[0,-p2[2,0],p2[1,0]],[p2[2,0],0,-p2[0,0]],[-p2[1,0],-p2[0,0],0]])
	
#usefull matrix
	I3=np.eye(3)
	O3=np.zeros((3,3))
	Wl1=np.hstack((I3,O3))
	
	J_cube = np.diag([m/6*r*r,m/6*r*r,m/6*r*r])
	
		
	#Mo=[ mI3 mR
	#    O3   J_cube]
	#here R =[0,0,0] center of mass are the same
	Mo=np.array([[m,0,0,0,0,0],
	[0,m,0,0,0,0],
	[0,0,m,0,0,0],
	[0,0,0,m/6*r*r,0,0],
	[0,0,0,0,m/6*r*r,0],
	[0,0,0,0,0,m/6*r*r]])
	
	#Bo
	b=np.dot(J_cube,W_object)
	b=np.cross(W_object,b,axis=0)
	Bo=np.array([[0],[0],[-m*g]])
	Bo=np.vstack((Bo,b))
	
	#W=[ I3 O3 ...
	#   P1  I3 ...] bloc repeat for the n arm manipulator
	Q=np.diag([m,m,m,m,m,m,m,m,m,m,m,m])
	Q_inv=np.linalg.inv(Q)
	W1l2=np.hstack((P1,np.eye(3)))
	W2l2=np.hstack((P2,np.eye(3)))
	W1=np.vstack((Wl1,W1l2))
	W2=np.vstack((Wl1,W2l2))
	W=np.hstack((W1,W2))
	W_T=np.transpose(W)
	
	W3=np.linalg.inv(np.dot(W,np.dot(Q_inv,W_T)))
	W_pseudo_inv=np.dot(np.dot(Q_inv,W_T),W3)
#Xdd_cmd=Iinv * [Fext+Fimp-B]
#ici B=0
#I inertia matrix of the object relative to it c.m for {bi}
#matrice inertie cube J_cube = diag(m/6*l²)
	km=120
	Md=np.array([[km,0,0,0,0,0],
	[0,km,0,0,0,0],
	[0,0,km,0,0,0],
	[0,0,0,km/6*r*r,0,0],
	[0,0,0,0,km/6*r*r,0],
	[0,0,0,0,0,km/6*r*r]])
	Md_inv=np.linalg.inv(Md)
	
	#B_od=[0,0,0, w X Jdes*w]t
	J_des=J_cube
	b=np.dot(J_des,W_object)
	b=np.cross(W_object,b,axis=0)
	B_od=np.array([[0],[0],[0]])
	B_od=np.vstack((B_od,b))
	
	Kp=400
	Kd=200
	F_imp=-Kp*e-Kd*ed  # terme md*xdd_des=0
	
	#Estimation de F_ext
	A=np.dot(Mo,Xdd_previous)
	WFprev=np.dot(W,F_previous)
	F_ext_estime=A+Bo-WFprev
	F_ext_estime=Xdd_previous
	F=F_ext_estime+F_imp-B_od
	
	Xdd_cmd=np.dot(Md_inv,F)


#F_cmd

	
	F_cmd=np.dot(W_pseudo_inv,(Bo-F_ext_estime+np.dot(Mo,(np.dot(Md_inv,F)))))
#a_i_cmd=xdd_cmd+wdxpi+wx(wxpi)
#ici absence de rotation donc
	xdd_cmd=Xdd_cmd[:3,:]
	
	a1_cmd=xdd_cmd+np.cross(Wd_object,p1,axis=0)+np.cross(W_object,np.cross(W_object,p1,axis=0),axis=0)
	a2_cmd=xdd_cmd+np.cross(Wd_object,p2,axis=0)+np.cross(W_object,np.cross(W_object,p2,axis=0),axis=0)	
	A_cmd=np.vstack((a1_cmd[:2,:],a2_cmd[:2,:]))

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
	f1=F_cmd[:2,:]
	f2=F_cmd[6:8,:]
	print("f1:",f1)
	print("f2:",f2)
	f_cmd=np.vstack((f1,f2))
	U=np.dot(M,qdd_cmd)+C+np.dot(J_T,f_cmd)
	if np.all(np.isnan(U)):
		U=np.array([[0,0,0,0]])
	print("U:",U)
	return [U,Xdd_cmd,F_cmd]
