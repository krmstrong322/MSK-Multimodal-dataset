function angleout=jointangle2(m1,m2,m3,m4,Data);


m1=m1; pt1=Data(:,m1*3-2:m1*3);
m1=m2; pt2=Data(:,m1*3-2:m1*3);  
m1=m3; pt3=Data(:,m1*3-2:m1*3);  
m1=m4; pt4=Data(:,m1*3-2:m1*3);  

angle=my3Dangle([pt1,pt2,pt3,pt4]);
angleout=180-angle;