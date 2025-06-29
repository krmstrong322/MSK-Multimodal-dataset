function angle=my3Dangle(xyzdata)

% Calculate 3D joint angle
% function [alpha]=angle3d(data)
% Description:  Calculates the angle between 2 vectors (given by pairs of points) in 2 dimensions
% Input: data = [P1x y z P2x y z P3x y z P4x y z]
% Note that "data" can have several rows (e.g. different time points).
% Output: joint1: angle (in deg) between the vectors P1-P2 and P3-P4
% Author: Christoph Reinschmidt, HPL, The University of Calgary
% Date: October, 1994
% Last Changes: November 28, 1996
% Version: 1.0
% Modified by David Mullineaux, 5Oct07

r1=xyzdata(:,1:3);
r2=xyzdata(:,4:6);
r3=xyzdata(:,7:9);
r4=xyzdata(:,10:12);

v1=r2-r1; v2=r4-r3;
for j=1:size(v1,1);
    vect1=[v1(j,:)]'; vect2=[v2(j,:)]';
    x=cross(vect1,vect2);
    alphacos=(acos(sum(vect1.*vect2)/(norm(vect1)*norm(vect2))))*180/pi;
    y=x(3,1);
    % Determining if alpha b/w 0 and pi or b/w -pi and 0
%     if sign(y)==-1;   alphacos=-alphacos; end
    angle(j,:)=[alphacos];
end
