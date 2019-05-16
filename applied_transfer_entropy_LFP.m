%% Transfer Entropy Applied to for Local Field Potential Recordings

function [T,T_indx,X_filt,Y_filt]=applied_transfer_entropy_LFP(X,Y,global_tlag,low_band,high_band,Fs,order)

%%%%%
% This is the overhead function for calculating the transfer entropy
% between two multi-channel EEG or field potential signals from brain
% regions X and Y

% Inputs:
% X: 2D Matrix of EEG/LFP signals (signal,channel number) lagging Y, based upon global tlag estimate (V)
% Y: 2D Matrix of EEG/LFP signal leading X (V)
% global_tlag: Global time lag (ms) (must be a single positive value)
% low_band: High frequency cut-off for bandpass filter (default=0.5Hz)
% high_band: Low frequency cut-off for bandpass filter (default=100Hz)
% Fs: Sampling Frequency
% order: zero-phase Butterworth filter order (default=3)

% Outputs:
% T: Transfer Entropy Values
% T_indx: Index of pairwise combinations corresponding to Transfer Entropy
% Values (T)
% X_filt: Butterworth filtered 2D Matrix of X
% Y_filt: Butterworth filtered 2D Matrix of Y

% James Young 2019

if nargin<7 || isempty(order)
    order=3;
end

%% Pre-Processing - Bandpass Filter for Specified Frequency Band

if isempty(low_band) && length(high_band)==1
    
    %     HP=low_band*(2/Fs);
    LP=high_band*(2/Fs);
    [d,c]=butter(order,LP,'high');
    X_filt=double(filtfilt(d,c,X));
    Y_filt=double(filtfilt(d,c,Y));
    
elseif isempty(high_band) && length(low_band)==1
    
    HP=low_band*(2/Fs);
    [d,c]=butter(order,HP,'low');
    X_filt=double(filtfilt(d,c,X));
    Y_filt=double(filtfilt(d,c,Y));
    
elseif isempty(low_band) && isempty(high_band)
    
    X_filt=double(X);
    Y_filt=double(Y);
    
else
    
    HP=low_band*(2/Fs);
    LP=high_band*(2/Fs);
    [d,c]=butter(order,[HP LP],'bandpass');
    X_filt=double(filtfilt(d,c,X));
    Y_filt=double(filtfilt(d,c,Y));
    
end

t=0; %time lag in X from present is set 0
global_tlag_samp=fix(global_tlag*(Fs/1000)); % Convert to from Time(ms) to Number of Samples
%% Calculation of Transfer Entropy

signal_mat=size(X);
k=1;
for i=1:signal_mat(2)

    for j=1:signal_mat(2)
        
[T(k)]=transferEntropyPartition(X_filt(:,i),Y_filt(:,j),t,global_tlag_samp);
T_indx(k,1)=i; % Channel Number of Signal X
T_indx(k,2)=j; % Channel Number of Signal Y
fprintf('Transfer Entropy for Pair-Wise Combination %d Done.\n',k)
k=k+1;

    end
    
end
T=T';

end

function [T nPar dimPar]=transferEntropyPartition(X,Y,t,w)

%%%%%
% This function computes the transfer entropy between time series X and Y,
% with the flow of information directed from X to Y. Probability density
% estimation is based on the Darbellay-Vajda partitioning algorithm.
%
% For details, please see T Schreiber, "Measuring information transfer", Physical Review Letters, 85(2):461-464, 2000.
%
% Inputs:
% X: source time series in 1-D vector
% Y: target time series in 1-D vector
% t: time lag in X from present
% w: time lag in Y from present
%
% Outputs:
% T: transfer entropy (bits)
% nPar: number of partitions
% dimPar: 1-D vector containing the length of each partition (same along all three dimensions)
%
%
% Copyright 2011 Joon Lee
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
%%%%%


% fix block lengths at 1
l=1; k=1;

X=X(:)';
Y=Y(:)';

% go through the time series X and Y, and populate Xpat, Ypat, and Yt
Xpat=[]; Ypat=[]; Yt=[];
for i=max([l+t k+w]):1:min([length(X) length(Y)])
    Xpat=[Xpat; X(i-l-t+1:i-t)];
    Ypat=[Ypat; Y(i-k-w+1:i-w)];
    Yt=[Yt; Y(i)];
end

% ordinal sampling (ranking)
Nt=length(Xpat);
[B,IX]=sort(Xpat);
Xpat(IX)=1:Nt;
[B,IX]=sort(Ypat);
Ypat(IX)=1:Nt;
[B,IX]=sort(Yt);
Yt(IX)=1:Nt;

% compute transfer entropy
partitions=DVpartition3D(Xpat,Ypat,Yt,1,Nt,1,Nt,1,Nt);
nPar=length(partitions);
dimPar=zeros(nPar,1);
for i=1:nPar
    dimPar(i)=partitions(i).Xmax-partitions(i).Xmin+1;
end
T=0;
for i=1:length(partitions)
    a=partitions(i).N/Nt;
    b=sum(Xpat>=partitions(i).Xmin & Xpat<=partitions(i).Xmax & Ypat>=partitions(i).Ymin & Ypat<=partitions(i).Ymax)/Nt;
    c=sum(Yt>=partitions(i).Zmin & Yt<=partitions(i).Zmax & Ypat>=partitions(i).Ymin & Ypat<=partitions(i).Ymax)/Nt;
    d=(partitions(i).Ymax-partitions(i).Ymin+1)/Nt;
    T=T+a*log2((a*d)/(b*c));
end


end

function partitions=DVpartition3D(X,Y,Z,Xmin,Xmax,Ymin,Ymax,Zmin,Zmax)

%%%%%
% This function implements the Darbellay-Vajda partitioning algorithm for a
% 3D space in a recursive manner. In order to support recursion, Each
% function call should specify a sub-space where partitioning should be
% executed. It is assumed that X, Y, and Z are ordinal (ranked, starting from 1) samples.
%
% For details, please see G A Darbellay and I Vajda, "Estimation of the information by an adaptive partitioning of the observation space", IEEE Transactions on Information Theory, 45(4):1315-1321, 1999.
%
% Inputs:
% X: 1-D vector containing coordinates in the first dimension
% Y: 1-D vector containing coordinates in the second dimension
% Z: 1-D vector containing coordinates in the third dimension
% Xmin: lower limit in the first dimension
% Ymin: lower limit in the second dimension
% Zmin: lower limit in the third dimension
% Xmax: upper limit in the first dimension
% Ymax: upper limit in the second dimension
% Zmax: upper limit in the third dimension
%
% Outputs:
% partitions: 1-D structure array that contains the lower and upper limits of each partition in each dimension as well as the number of data points (N) in the partition
%
%
% Copyright 2011 Joon Lee
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
%%%%%


alpha=0.05;

idx= X>=Xmin & X<=Xmax & Y>=Ymin & Y<=Ymax & Z>=Zmin & Z<=Zmax;
Xsub=X(idx);
Ysub=Y(idx);
Zsub=Z(idx);

Xdiv=floor(mean([Xmin Xmax]));
Ydiv=floor(mean([Ymin Ymax]));
Zdiv=floor(mean([Zmin Zmax]));

N=[sum(Xsub<=Xdiv & Ysub<=Ydiv & Zsub<=Zdiv) sum(Xsub>Xdiv & Ysub<=Ydiv & Zsub<=Zdiv) sum(Xsub<=Xdiv & Ysub>Ydiv & Zsub<=Zdiv) sum(Xsub>Xdiv & Ysub>Ydiv & Zsub<=Zdiv) ...
    sum(Xsub<=Xdiv & Ysub<=Ydiv & Zsub>Zdiv) sum(Xsub>Xdiv & Ysub<=Ydiv & Zsub>Zdiv) sum(Xsub<=Xdiv & Ysub>Ydiv & Zsub>Zdiv) sum(Xsub>Xdiv & Ysub>Ydiv & Zsub>Zdiv)];
T=sum((mean(N)-N).^2)./mean(N);  % has been corrected to include a normalization factor in the denominator
partitions=struct('Xmin',{},'Xmax',{},'Ymin',{},'Ymax',{},'Zmin',{},'Zmax',{},'N',{});

if T>icdf('chi2',1-alpha,7) && Xmax~=Xmin && Ymax~=Ymin && Zmax~=Zmin
    if N(1)~=0
        partitions=[partitions DVpartition3D(X,Y,Z,Xmin,Xdiv,Ymin,Ydiv,Zmin,Zdiv)];
    end
    if N(3)~=0
        partitions=[partitions DVpartition3D(X,Y,Z,Xmin,Xdiv,Ydiv+1,Ymax,Zmin,Zdiv)];
    end
    if N(2)~=0
        partitions=[partitions DVpartition3D(X,Y,Z,Xdiv+1,Xmax,Ymin,Ydiv,Zmin,Zdiv)];
    end
    if N(4)~=0
        partitions=[partitions DVpartition3D(X,Y,Z,Xdiv+1,Xmax,Ydiv+1,Ymax,Zmin,Zdiv)];
    end
    if N(5)~=0
        partitions=[partitions DVpartition3D(X,Y,Z,Xmin,Xdiv,Ymin,Ydiv,Zdiv+1,Zmax)];
    end
    if N(7)~=0
        partitions=[partitions DVpartition3D(X,Y,Z,Xmin,Xdiv,Ydiv+1,Ymax,Zdiv+1,Zmax)];
    end
    if N(6)~=0
        partitions=[partitions DVpartition3D(X,Y,Z,Xdiv+1,Xmax,Ymin,Ydiv,Zdiv+1,Zmax)];
    end
    if N(8)~=0
        partitions=[partitions DVpartition3D(X,Y,Z,Xdiv+1,Xmax,Ydiv+1,Ymax,Zdiv+1,Zmax)];
    end
elseif sum(idx)~=0
    partitions(1).Xmin=Xmin;
    partitions(1).Xmax=Xmax;
    partitions(1).Ymin=Ymin;
    partitions(1).Ymax=Ymax;
    partitions(1).Zmin=Zmin;
    partitions(1).Zmax=Zmax;
    partitions(1).N=sum(idx);
end

end