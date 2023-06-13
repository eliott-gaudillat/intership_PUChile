syms q1 q2 w1 w2 m1 L1 m2 L2 g tau tau2

X=[q1;q2;w1;w2];
Tau=[tau;tau2];

M=[m1*L1^2+m2*(L1^2+2*L1*L2*cos(q2)+L2^2),m2*(L1*L2*cos(q2)+L2^2); ...
    m2*(L1*L2*cos(q2)+L2^2),m2+L2^2];

C=[-m2*L1*L2*sin(q2)*(2*w1*w2+w2^2);
    m2*L1*L2*w1^2*sin(q2)];

g=[(m1+m2)*L1*g*cos(q1)+m2*g*L2*cos(q1+q2);
    m2*g*L2*cos(q1+q2)];

h=C+g;


Q_second=M\(Tau-h);


Xpoint=[X(3);
    X(4);
    Q_second(1);
    Q_second(2)];


A=jacobian(Xpoint,[X(1),X(2),X(3),X(4)]);
B=jacobian(Xpoint,[tau,tau2]);