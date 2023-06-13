import numpy as np

def SmallestSignedAngleBetween(theta1, theta2):     # Para que el angulo se mantenga en el rango [-pi, pi]
    a = (theta1 - theta2) % (2.*np.pi)
    b = (theta2 - theta1) % (2.*np.pi)
    return -a if a < b else b

# fonction MGI renvoie le q1 et q2 en fonction de la position de l'OT en x et y
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
	
#fonction renvoie nos matrices A,B et C de notre rep d'état de la forme X_point=AX+Bu, Y=CX	
def state_matrix(q1,q2,w1,w2,tau,tau2,g):
	L1=0.1
	L2=0.11
	r=0.01
	mv=1000
	V_tube=np.pi*r*r*L1
	V_sphere=4/3*np.pi*r*r*r
	m1=(V_tube+V_sphere)*mv
	m2=m1
	A=np.array([[0,0,1,0],
	[0,0,0,1],
	[(- g*np.sin(q1 + q2)*L2**3*m2**2 + g*np.sin(q1 + q2)*L2**3*m2 - L1*g*np.sin(q1 + q2)*np.cos(q2)*L2**2*m2**2 + L1*g*np.sin(q1)*L2**2*m2 + L1*g*m1*np.sin(q1)*L2**2 + g*np.sin(q1 + q2)*L2*m2**2 + L1*g*np.sin(q1)*m2**2 + L1*g*m1*np.sin(q1)*m2)/(L2**4*m2 + L1**2*m2**2 + L2**2*m2**2 - L2**4*m2**2 + L1**2*L2**2*m1 + L1**2*L2**2*m2 + L1**2*m1*m2 + 2*L1*L2*m2**2*np.cos(q2) + 2*L1*L2**3*m2*np.cos(q2) - L1**2*L2**2*m2**2*np.cos(q2)**2 - 2*L1*L2**3*m2**2*np.cos(q2)),(L2*g*m2**2*np.sin(q1 + q2) + L2**3*g*m2*np.sin(q1 + q2) - L2**3*g*m2**2*np.sin(q1 + q2) + L1*L2*m2**2*w2**2*np.cos(q2) + L1*L2**3*m2*w2**2*np.cos(q2) + L1**2*L2**2*m2**2*w1**2*np.cos(q2)**2 - L1**2*L2**2*m2**2*w1**2*np.sin(q2)**2 + L1*L2*m2*tau2*np.sin(q2) + L1*L2**3*m2**2*w1**2*np.cos(q2) - L1*L2**2*g*m2**2*np.cos(q1 + q2)*np.sin(q2) - L1*L2**2*g*m2**2*np.sin(q1 + q2)*np.cos(q2) + 2*L1*L2*m2**2*w1*w2*np.cos(q2) + 2*L1*L2**3*m2*w1*w2*np.cos(q2))/(L2**4*m2 + L1**2*m2**2 + L2**2*m2**2 - L2**4*m2**2 + L1**2*L2**2*m1 + L1**2*L2**2*m2 + L1**2*m1*m2 + 2*L1*L2*m2**2*np.cos(q2) + 2*L1*L2**3*m2*np.cos(q2) - L1**2*L2**2*m2**2*np.cos(q2)**2 - 2*L1*L2**3*m2**2*np.cos(q2)) - ((2*np.cos(q2)*np.sin(q2)*L1**2*L2**2*m2**2 + 2*np.sin(q2)*L1*L2**3*m2**2 - 2*np.sin(q2)*L1*L2**3*m2 - 2*np.sin(q2)*L1*L2*m2**2)*(np.cos(q2)*np.sin(q2)*L1**2*L2**2*m2**2*w1**2 + np.sin(q2)*L1*L2**3*m2**2*w1**2 + 2*np.sin(q2)*L1*L2**3*m2*w1*w2 + np.sin(q2)*L1*L2**3*m2*w2**2 + g*np.cos(q1 + q2)*np.cos(q2)*L1*L2**2*m2**2 - g*np.cos(q1)*L1*L2**2*m2 - g*m1*np.cos(q1)*L1*L2**2 + 2*np.sin(q2)*L1*L2*m2**2*w1*w2 + np.sin(q2)*L1*L2*m2**2*w2**2 - tau2*np.cos(q2)*L1*L2*m2 - g*np.cos(q1)*L1*m2**2 - g*m1*np.cos(q1)*L1*m2 + g*np.cos(q1 + q2)*L2**3*m2**2 - g*np.cos(q1 + q2)*L2**3*m2 - tau2*L2**2*m2 + tau*L2**2 - g*np.cos(q1 + q2)*L2*m2**2 + tau*m2))/(L2**4*m2 + L1**2*m2**2 + L2**2*m2**2 - L2**4*m2**2 + L1**2*L2**2*m1 + L1**2*L2**2*m2 + L1**2*m1*m2 + 2*L1*L2*m2**2*np.cos(q2) + 2*L1*L2**3*m2*np.cos(q2) - L1**2*L2**2*m2**2*np.cos(q2)**2 - 2*L1*L2**3*m2**2*np.cos(q2))**2,(2*w1*np.cos(q2)*np.sin(q2)*L1**2*L2**2*m2**2 + 2*w1*np.sin(q2)*L1*L2**3*m2**2 + 2*w2*np.sin(q2)*L1*L2**3*m2 + 2*w2*np.sin(q2)*L1*L2*m2**2)/(L2**4*m2 + L1**2*m2**2 + L2**2*m2**2 - L2**4*m2**2 + L1**2*L2**2*m1 + L1**2*L2**2*m2 + L1**2*m1*m2 + 2*L1*L2*m2**2*np.cos(q2) + 2*L1*L2**3*m2*np.cos(q2) - L1**2*L2**2*m2**2*np.cos(q2)**2 - 2*L1*L2**3*m2**2*np.cos(q2)),(2*L1*L2*m2**2*w1*np.sin(q2) + 2*L1*L2*m2**2*w2*np.sin(q2) + 2*L1*L2**3*m2*w1*np.sin(q2) + 2*L1*L2**3*m2*w2*np.sin(q2))/(L2**4*m2 + L1**2*m2**2 + L2**2*m2**2 - L2**4*m2**2 + L1**2*L2**2*m1 + L1**2*L2**2*m2 + L1**2*m1*m2 + 2*L1*L2*m2**2*np.cos(q2) + 2*L1*L2**3*m2*np.cos(q2) - L1**2*L2**2*m2**2*np.cos(q2)**2 - 2*L1*L2**3*m2**2*np.cos(q2))],
	[-(L1*L2**2*g*m2**2*np.sin(q1) - L1**2*L2*g*m2**2*np.sin(q1 + q2) - L1*L2**2*g*m2**2*np.sin(q1 + q2)*np.cos(q2) + L1**2*L2*g*m2**2*np.cos(q2)*np.sin(q1) - L1**2*L2*g*m1*m2*np.sin(q1 + q2) + L1*L2**2*g*m1*m2*np.sin(q1) + L1**2*L2*g*m1*m2*np.cos(q2)*np.sin(q1))/(L2**4*m2 + L1**2*m2**2 + L2**2*m2**2 - L2**4*m2**2 + L1**2*L2**2*m1 + L1**2*L2**2*m2 + L1**2*m1*m2 + 2*L1*L2*m2**2*np.cos(q2) + 2*L1*L2**3*m2*np.cos(q2) - L1**2*L2**2*m2**2*np.cos(q2)**2 - 2*L1*L2**3*m2**2*np.cos(q2)),((2*np.cos(q2)*np.sin(q2)*L1**2*L2**2*m2**2 + 2*np.sin(q2)*L1*L2**3*m2**2 - 2*np.sin(q2)*L1*L2**3*m2 - 2*np.sin(q2)*L1*L2*m2**2)*(L2**2*m2*tau - L1**2*m1*tau2 - L1**2*m2*tau2 - L2**2*m2*tau2 + L1**2*L2*g*m2**2*np.cos(q1 + q2) - L1*L2**2*g*m2**2*np.cos(q1) + L1*L2*m2*tau*np.cos(q2) - 2*L1*L2*m2*tau2*np.cos(q2) + L1*L2**3*m2**2*w1**2*np.sin(q2) + L1**3*L2*m2**2*w1**2*np.sin(q2) + L1*L2**3*m2**2*w2**2*np.sin(q2) + L1*L2**2*g*m2**2*np.cos(q1 + q2)*np.cos(q2) + 2*L1**2*L2**2*m2**2*w1**2*np.cos(q2)*np.sin(q2) + L1**2*L2**2*m2**2*w2**2*np.cos(q2)*np.sin(q2) - L1**2*L2*g*m2**2*np.cos(q1)*np.cos(q2) + L1**2*L2*g*m1*m2*np.cos(q1 + q2) - L1*L2**2*g*m1*m2*np.cos(q1) + L1**3*L2*m1*m2*w1**2*np.sin(q2) + 2*L1*L2**3*m2**2*w1*w2*np.sin(q2) + 2*L1**2*L2**2*m2**2*w1*w2*np.cos(q2)*np.sin(q2) - L1**2*L2*g*m1*m2*np.cos(q1)*np.cos(q2)))/(L2**4*m2 + L1**2*m2**2 + L2**2*m2**2 - L2**4*m2**2 + L1**2*L2**2*m1 + L1**2*L2**2*m2 + L1**2*m1*m2 + 2*L1*L2*m2**2*np.cos(q2) + 2*L1*L2**3*m2*np.cos(q2) - L1**2*L2**2*m2**2*np.cos(q2)**2 - 2*L1*L2**3*m2**2*np.cos(q2))**2 - (2*L1**2*L2**2*m2**2*w1**2*np.cos(q2)**2 - L1**2*L2*g*m2**2*np.sin(q1 + q2) + L1**2*L2**2*m2**2*w2**2*np.cos(q2)**2 - 2*L1**2*L2**2*m2**2*w1**2*np.sin(q2)**2 - L1**2*L2**2*m2**2*w2**2*np.sin(q2)**2 - L1*L2*m2*tau*np.sin(q2) + 2*L1*L2*m2*tau2*np.sin(q2) + L1*L2**3*m2**2*w1**2*np.cos(q2) + L1**3*L2*m2**2*w1**2*np.cos(q2) + L1*L2**3*m2**2*w2**2*np.cos(q2) - L1*L2**2*g*m2**2*np.cos(q1 + q2)*np.sin(q2) - L1*L2**2*g*m2**2*np.sin(q1 + q2)*np.cos(q2) + L1**2*L2*g*m2**2*np.cos(q1)*np.sin(q2) - L1**2*L2*g*m1*m2*np.sin(q1 + q2) + 2*L1**2*L2**2*m2**2*w1*w2*np.cos(q2)**2 - 2*L1**2*L2**2*m2**2*w1*w2*np.sin(q2)**2 + L1**3*L2*m1*m2*w1**2*np.cos(q2) + 2*L1*L2**3*m2**2*w1*w2*np.cos(q2) + L1**2*L2*g*m1*m2*np.cos(q1)*np.sin(q2))/(L2**4*m2 + L1**2*m2**2 + L2**2*m2**2 - L2**4*m2**2 + L1**2*L2**2*m1 + L1**2*L2**2*m2 + L1**2*m1*m2 + 2*L1*L2*m2**2*np.cos(q2) + 2*L1*L2**3*m2*np.cos(q2) - L1**2*L2**2*m2**2*np.cos(q2)**2 - 2*L1*L2**3*m2**2*np.cos(q2)),-(2*L1*L2**3*m2**2*w1*np.sin(q2) + 2*L1**3*L2*m2**2*w1*np.sin(q2) + 2*L1*L2**3*m2**2*w2*np.sin(q2) + 2*L1**3*L2*m1*m2*w1*np.sin(q2) + 4*L1**2*L2**2*m2**2*w1*np.cos(q2)*np.sin(q2) + 2*L1**2*L2**2*m2**2*w2*np.cos(q2)*np.sin(q2))/(L2**4*m2 + L1**2*m2**2 + L2**2*m2**2 - L2**4*m2**2 + L1**2*L2**2*m1 + L1**2*L2**2*m2 + L1**2*m1*m2 + 2*L1*L2*m2**2*np.cos(q2) + 2*L1*L2**3*m2*np.cos(q2) - L1**2*L2**2*m2**2*np.cos(q2)**2 - 2*L1*L2**3*m2**2*np.cos(q2)),-(2*L1*L2**3*m2**2*w1*np.sin(q2) + 2*L1*L2**3*m2**2*w2*np.sin(q2) + 2*L1**2*L2**2*m2**2*w1*np.cos(q2)*np.sin(q2) + 2*L1**2*L2**2*m2**2*w2*np.cos(q2)*np.sin(q2))/(L2**4*m2 + L1**2*m2**2 + L2**2*m2**2 - L2**4*m2**2 + L1**2*L2**2*m1 + L1**2*L2**2*m2 + L1**2*m1*m2 + 2*L1*L2*m2**2*np.cos(q2) + 2*L1*L2**3*m2*np.cos(q2) - L1**2*L2**2*m2**2*np.cos(q2)**2 - 2*L1*L2**3*m2**2*np.cos(q2))]])
	B=np.array([[0,0],
	[0,0],
	[(L2**2 + m2)/(L2**4*m2 + L1**2*m2**2 + L2**2*m2**2 - L2**4*m2**2 + L1**2*L2**2*m1 + L1**2*L2**2*m2 + L1**2*m1*m2 + 2*L1*L2*m2**2*np.cos(q2) + 2*L1*L2**3*m2*np.cos(q2) - L1**2*L2**2*m2**2*np.cos(q2)**2 - 2*L1*L2**3*m2**2*np.cos(q2)),-(m2*L2**2 + L1*m2*np.cos(q2)*L2)/(L2**4*m2 + L1**2*m2**2 + L2**2*m2**2 - L2**4*m2**2 + L1**2*L2**2*m1 + L1**2*L2**2*m2 + L1**2*m1*m2 + 2*L1*L2*m2**2*np.cos(q2) + 2*L1*L2**3*m2*np.cos(q2) - L1**2*L2**2*m2**2*np.cos(q2)**2 - 2*L1*L2**3*m2**2*np.cos(q2))],
	[-(m2*L2**2 + L1*m2*np.cos(q2)*L2)/(L2**4*m2 + L1**2*m2**2 + L2**2*m2**2 - L2**4*m2**2 + L1**2*L2**2*m1 + L1**2*L2**2*m2 + L1**2*m1*m2 + 2*L1*L2*m2**2*np.cos(q2) + 2*L1*L2**3*m2*np.cos(q2) - L1**2*L2**2*m2**2*np.cos(q2)**2 - 2*L1*L2**3*m2**2*np.cos(q2)),(L1**2*m1 + L1**2*m2 + L2**2*m2 + 2*L1*L2*m2*np.cos(q2))/(L2**4*m2 + L1**2*m2**2 + L2**2*m2**2 - L2**4*m2**2 + L1**2*L2**2*m1 + L1**2*L2**2*m2 + L1**2*m1*m2 + 2*L1*L2*m2**2*np.cos(q2) + 2*L1*L2**3*m2*np.cos(q2) - L1**2*L2**2*m2**2*np.cos(q2)**2 - 2*L1*L2**3*m2**2*np.cos(q2))]])
	C=np.identity(4)

	return A,B,C


