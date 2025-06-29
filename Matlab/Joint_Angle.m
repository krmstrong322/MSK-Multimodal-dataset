clc;
clear all;
close all;


%--------------------------------------------------------------------------
%                               CREATE MENU BOX
%--------------------------------------------------------------------------
prompt={'Enter data path:','Enter filename for key file:',... 
    'Enter first trial','Enter last trial, or 0 for maximum','Save data (Yes(Y) or No(N))',...
    'Enter kinematic cutoff (enter "0" to skip filter or "1" to apply filter)','Enter the Recording Frequency (Hz)'}; %'Print graphs (Yes or No; warning will crash if over 80trials)', 
Title='Start Menu';
lines=1;

% Add Default Values
def={'D:\MSK','MSK_keyfile','1','1','N','1','150'};

answer=inputdlg(prompt,Title,lines,def);

% Assign Inputs
data_path=char(answer{1});
keyfile=char(answer{2});
starttrial=str2num(char(answer{3}));
endtrial=str2num(char(answer{4}));
savefile=char(answer{5});
CFkin=str2num(char(answer{6}));
FREQ=str2num(char(answer{7}));

% set data path; evaluate; load keyfile
openKeyfile = strcat(data_path,'\',keyfile,'.txt') ;

% load keyfile, and assign each column to different variables
% fIDs = fopen('keyfile');
[fid, msg]=fopen(openKeyfile,'r');
C=textscan(fid,'%s %s %s', 'Delimiter', '\t');
filedir=(C{1});
subdir2=(C{2});
subject=(C{3});

[ntrials, ~]=size(filedir);

% Determine num of trials to analyse (e.g. trials 1 to 2)
if endtrial<1
endtrial=ntrials;
end

%--------------------------------------------------------------------------
%                              LOAD TRC FILES
%--------------------------------------------------------------------------
% start loop to load each file in turn. Combine keyfile info to get filenames etc
for i = starttrial:endtrial
    fileloc = char(strcat(filedir(i,:), subdir2(i,:)));
    fileloc = strrep(fileloc, 'none', ''); % Replaces text
    trcfile = char(strcat(subject(i,:)));
    trcfile = strrep(trcfile, 'none', ''); % Replaces text

% Check if File Exists

A = [fileloc trcfile];
B = exist(A,"file");

if B == 0
continue
end    

switch char(filedir(i,:))
    case 'none'
    rawkine = []; % Ignore rows with no file (needed to keep sequence of trials)
    otherwise
    rawkine = dlmread([fileloc trcfile],'\t',6,2);
    rawkine = rawkine(:,1:72);
%        FREQ = dlmread([fileloc trcfile],'\t',[2,3,2,3]);
end


%--------------------------------------------------------------------------
%                       GAP FILL SCRIPT (FROM DAVID)
%--------------------------------------------------------------------------
% FILL GAPS (needs to be a loop otherwise smooths across columns)

[nrkin,nckin]=size(rawkine); 

rawkinetemp=repmat(NaN,nrkin,nckin);
index=find(abs(rawkine));
rawkinetemp(index)=rawkine(index);
rawkine=rawkinetemp;
 
for j=1:nckin
index=find(isnan(rawkine(:,j)));
[nrtemp, nctemp]=size(index);
if nrtemp<nrkin %use this to jump filling if no data, otherwise will error 
X=rawkine(:,j);
X(isnan(X)) = interp1(find(~isnan(X)), X(~isnan(X)), find(isnan(X)),'spline'); %use nearest, spline, not spline, cubic, pchip
rawkine(:,j)=X;
end
end %jumps if nrtemp = nrkin (as no need to fill; also will error)


% Filter
TempKineMData=rawkine;
if CFkin>0
    KineMData=mybutter(2,6,FREQ,'low',TempKineMData,0.25); %(butterorder,CFkin,SF,type,data,damped);
else
    KineMData = rawkine;
end

%--------------------------------------------------------------------------
%                     Calculate joint angles                 
%--------------------------------------------------------------------------
% this code uses David's joint angle function to calculate 3D joint angle
% by taking cross product:
% Knee = GT, Lat Knee, Lat Ankle
% Ankle = Lat Knee, Lat Ankle, Met5

% Joint angles calculated for the entire length of recording


LeftKneeAngle=jointangle2(5,9,9,15,KineMData);
RightKneeAngle=jointangle2(6,11,11,17,KineMData);

LeftAnkleAngle=jointangle2(9,15,15,20,KineMData); 
RightAnkleAngle=jointangle2(11,17,17,23,KineMData);  

figure();
plot(LeftKneeAngle);
title('Left Knee Angle');

figure();
plot(RightKneeAngle);
title('Right Knee Angle');

figure();
plot(LeftAnkleAngle);
title('Left Ankle Angle');

figure();
plot(RightAnkleAngle);
title('Right Ankle Angle');

end
