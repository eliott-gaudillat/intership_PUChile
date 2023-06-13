import math
import numpy as np


	
def SmallestSignedAngleBetween(theta1, theta2):     # Para que el angulo se mantenga en el rango [-pi, pi]
    a = (theta1 - theta2) % (2.*np.pi)
    b = (theta2 - theta1) % (2.*np.pi)
    return -a if a < b else b
 
# fonction MGI renvoie q1 et q2 en fonction de la position courant de l'OT en x et y   
def MGIv2(x,y):
	#longeur de nos corps en m
	l1=0.1
	l2=0.11
	#Matrice de Passage Rl a Ro
	RlR=np.array([[1,0,0.1],[0,1,0],[0,0,1]])
	#Matrice de Passage Rr a Ro
	RrR=np.array([[1,0,-0.1],[0,1,0],[0,0,1]])
	# passage de x et y dans les reperes des deux robots
	X=np.array([[x],[y],[1]])
	Xr=np.dot(RrR,X)
	Xl=np.dot(RlR,X)
	#calcul de theta Rr
	x_R=Xr[0,0]
	y_R=Xr[1,0]
	gamma_R=np.arctan2(y_R,x_R)
	alpha_R=np.arccos((l1*l1+l2*l2-x_R*x_R-y_R*y_R)/(2*l1*l2))
	beta_R=np.arccos((x_R*x_R+y_R*y_R+l1*l1-l2*l2)/(2*l1*np.sqrt(x_R*x_R+y_R*y_R)))
	theta1_R=gamma_R+beta_R
	theta2_R=alpha_R-np.pi
	
	#calcul de theta Rl
	x_L=Xl[0,0]
	y_L=Xl[1,0]
	gamma_L=np.arctan2(y_L,x_L)
	alpha_L=np.arccos((l1*l1+l2*l2-x_L*x_L-y_L*y_L)/(2*l1*l2))
	beta_L=np.arccos((x_L*x_L+y_L*y_L+l1*l1-l2*l2)/(2*l1*np.sqrt(x_L*x_L+y_L*y_L)))
	theta1_L=gamma_L+beta_L
	theta2_L=alpha_L-np.pi
	
	return (theta1_R,theta2_R,theta1_L,theta2_L)
