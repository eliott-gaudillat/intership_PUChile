import math
import numpy as np

# fonction MGI renvoie q1 et q2 en fonction de la position courant de l'OT en x et y
def MGI(x,y):
	#longeur de nos corps en m
	l1=0.1
	l2=0.1
	
	#calcul de q2
	cosq2=(x*x+y*y-l1*l1-l2*l2)/2*l1*l2
	sinq2=math.sqrt(1-cosq2*cosq2)
	q2=np.arctan2(sinq2,cosq2)
	
	#calcul q1
	B1=l1+l2*cosq2
	B2=l2*sinq2
	sinq1=(B1*y-B2*x)/(B1*B1+B2*B2)
	cosq1=(B1*x+B2*y)/(B1*B1+B2*B2)
	q1=np.arctan2(sinq1,cosq1)
	
	return (q1,q2)
	
def SmallestSignedAngleBetween(theta1, theta2):     # Para que el angulo se mantenga en el rango [-pi, pi]
    a = (theta1 - theta2) % (2.*np.pi)
    b = (theta2 - theta1) % (2.*np.pi)
    return -a if a < b else b
 
# variante du MGI    
def MGIv2(x,y):
	#longeur de nos corps en m
	l1=0.1
	l2=0.11
	#calcul de q2
	gamma=np.arctan2(y,x)
	alpha=np.arccos((l1*l1+l2*l2-x*x-y*y)/(2*l1*l2))
	beta=np.arccos((x*x+y*y+l1*l1-l2*l2)/(2*l1*np.sqrt(x*x+y*y)))
	
	theta1=gamma+beta
	theta2=alpha-np.pi
	
	return (theta1,theta2)
