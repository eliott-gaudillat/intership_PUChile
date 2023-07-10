
import numpy as np
import math


def Dual_arm_impedance(observation):

	#External/Object Impedance ( T_ed -> T_er)
	#T_ed=(x_ed,R_ed,xd_ed,w_ed,xdd_ed,wd_ed) position/orientation/vitesse(lin et angulaire)/acceleration desiré
	#ici
	X_ed=np.array([x_star_target],[y_star_target],[z_star_target]])
	R_ed=np.diag([1,1,1])
	Q_ed=quaternion.from_rotation_vector(R_ed)
	xd_ed=np.array([[0],[0],[0],[0]])
	wd_ed=np.array([[0],[0],[0],[0]])
	xdd_ed=np.array([[0],[0],[0],[0]])
	wdd_ed=np.array([[0],[0],[0],[0]])
	
	#integrating differencial equation
	#(A-I)*Me*vd_er+De*v_er+h_delta-e=-her_e-C(v_e)*v_e-g_e+A*Me*vd_ed+De*v_ed
	#object Mass matrix
	Me=np.array([[m,0,0,0,0,0],
	[0,m,0,0,0,0],
	[0,0,m,0,0,0],
	[0,0,0,1,0,0],
	[0,0,0,0,1,0],
	[0,0,0,0,0,1]])
	A=np.array([[alpha_p,0,0,0,0,0],
	[0,alpha_p,0,0,0,0],
	[0,0,alpha_p,0,0,0],
	[0,0,0,alpha_o,0,0],
	[0,0,0,0,alpha_o,0],
	[0,0,0,0,0,alpha_0]])
	De=np.diag([Dx,Dy,Dz,Drx,Dry,Drz])
	I=np.eye(6)
	
	he=np.dot(W,h)
	her_e=np.dot(np.transpose(R_er),he)
	
	h_delta_e=
	
	#closed-chain constraints (T_er -> T_kd)
	#T_rd
	X_r=Xe+Re*e_e
	R_r=R_e
	Xd_r=Xd_e-S(Re*e_e)*w_e
	w_r=w_e
	Xdd_r=Xdd_e-S(w_e)*S(Re*e_e)*w_e-S(Re*e_e)*wd_e
	wd_r=wd_e
	#T_ld
	X_l=Xe+Re*e_e
	R_l=R_e
	Xd_l=Xd_e-S(Re*e_e)*w_e
	w_l=w_e
	Xdd_l=Xdd_e-S(w_e)*S(Re*e_e)*w_e-S(Re*e_e)*wd_e
	wd_l=wd_e
	
	#Internal/Manipulators Impedance (T_Kd -> T_kr)
	#T_1r
	M1=np.array([[m,0,0,0,0,0],
	[0,m,0,0,0,0],
	[0,0,m,0,0,0],
	[0,0,0,1,0,0],
	[0,0,0,0,1,0],
	[0,0,0,0,0,1]])
	D1=np.diag([Dx,Dy,Dz,Drx,Dry,Drz])
	K1=np.diag([Kx,Ky,Kz,Krx,Kry,Krz])
	h_delta_I1=
	hi1=np.dot(W,h)
	h1r_i1=np.dot(np.transpose(R_1r),hi1)
	
	#T_2r
	M2=np.array([[m,0,0,0,0,0],
	[0,m,0,0,0,0],
	[0,0,m,0,0,0],
	[0,0,0,1,0,0],
	[0,0,0,0,1,0],
	[0,0,0,0,0,1]])
	D2=np.diag([Dx,Dy,Dz,Drx,Dry,Drz])
	K2=np.diag([Kx,Ky,Kz,Krx,Kry,Krz])
	h_delta_I2=
	hi2=np.dot(W,h)
	h2r_i2=np.dot(np.transpose(R_2r),hi2)
	#Motion control (T_kr-> torque)
	#PID
	Uk=Pk(q_kr-qk)+Dk(w_kr-wk)+Ik*integration(qkr-qk)
