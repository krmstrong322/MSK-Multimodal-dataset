function smoothdata=mybutter(butterorder,CFkin,SF,type,data,damped);

% see page 48 of Winter's book about damped and critically damped:

% best to use butterworth catering for filtfilt with damped=0.25
% "slight overshoot in response to step or impulse type inputs, but they have a much shorter rise time")
% or
% use critically damped butterworth catering for filtfilt with damped=0.5
% "no overshoot, but suffer from a slower rise time. Because impulsive type inputs 
% are rarely seen in human movement, the Butterworth filter is preferred"

% INPUT
% butter order integer (e.g. 2)
% CF cutoff frequency as integer (e.g. 6)
% SF sampling frequency as integer (e.g. 60; at least twice CF)
% type is text (options are: 'low' 'high' 'stop') - see butter function help
% data is in columns (e.g. xyz kinematics in columns 1,2,3)
% damped is 1, 0.25 or 0.5 (see below)

% OUTPUT
% smoothed data (see notes below for eventual filter type options)

switch damped
    case 1 %output still zero-lag (because of filtfilt), but butterorder doubled
        damppower=1/butterorder;
    case 0.25 %output still zero-lag (because of filtfilt), but butterorder NOT changed
        damppower=1;
    case 0.5 %output still zero-lag (because of filtfilt), but butterorder NOT changed.
        % Data is critically damped
        damppower=2;      
end

CFcorrection=(2^(1/(damppower*butterorder))-1)^damped;
[b,a]=butter(butterorder,(CFkin/CFcorrection)/(SF/2),type);
smoothdata = filtfilt(b,a,data);
