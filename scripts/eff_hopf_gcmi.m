% Hopf model inversion based on SEEG empirical data during task conditions
% Estimates the effective connectivity computed using Gaussian-Copula Mutual Information 

% specify condition and reaction time 
% condition=1; % 1:2 easy, hard
% reaction=1;  % 1:3 fast, middle, slow


function eff_hopf_gcmi(simulation, reaction, condition)

rng shuffle; % necessary to generate random numbers in the cluster

addpath('../data');

% SEEG Gaussian-Copula Mutual Information, empirical data
load('dfc-rt-binned_estimator-gcmi_freq-f50f150_align-sample_stim_lowpass-1.mat');
FC=squeeze(data(:,:,condition,reaction));

% Structural Connectivity matrix
load('SC.mat');
C = data;
C = double(C);
C = C - diag(diag(C));       % values of diagonal to zero
C = C/max(max(C))*0.2;       % scale structural connectivity (DTI) to a maximum value of 0.2

FCemp = FC;
nmask=~isnan(FCemp);         % mask of NaNs
NSIM=1;                    % total simulations

% Parameters of the data
TR=1;                         % Repetition Time (seconds)
N=82;                         % total regions
Isubdiag = find(tril(ones(N),-1));
f_diff=0.05*ones(1,N);        % frequency

% Bandpass filter settings
fnq=1/(2*TR);                 % Nyquist frequency
flp = 0.008;                  % lowpass frequency of filter (Hz)
fhi = 0.08;                   % highpass
Wn=[flp/fnq fhi/fnq];         % butterworth bandpass non-dimensional frequency
k=2;                          % 2nd order butterworth filter
[bfilt,afilt]=butter(k,Wn);   % construct the filter

% Parameters HOPF
a=-0.02*ones(N,2);
omega = repmat(2*pi*f_diff',1,2); omega(:,1) = -omega(:,1);
dt=0.1*TR/2;
sig=0.01;
dsig = sqrt(dt)*sig;
Tmax = 10000;

%%%%%%%%%%%%
%% Optimize
%%
GCsim2=zeros(NSIM,N,N); % simulated Gaussian-Copula Mutual Information
Cnew=C;                 % effective connectivity matrix updated in each iteration
for iter=1:500
    iter
    wC = Cnew; % updated in each iteration
    sumC = repmat(sum(wC,2),1,2); % for sum Cij*xj
    for sub=1:NSIM
        sub
        xs=zeros(Tmax,N);
        z = 0.1*ones(N,2); % --> x = z(:,1), y = z(:,2)
        nn=0;
        % discard first 3000 time steps
        for t=0:dt:3000
            suma = wC*z - sumC.*z; % sum(Cij*xi) - sum(Cij)*xj
            zz = z(:,end:-1:1); % flipped z, because (x.*x + y.*y)
            z = z + dt*(a.*z + zz.*omega - z.*(z.*z+zz.*zz) + suma) + dsig*randn(N,2);
        end
        % actual modeling (x=BOLD signal (Interpretation), y some other oscillation)
        for t=0:dt:((Tmax-1)*TR)
            suma = wC*z - sumC.*z; % sum(Cij*xi) - sum(Cij)*xj
            zz = z(:,end:-1:1); % flipped z, because (x.*x + y.*y)
            z = z + dt*(a.*z + zz.*omega - z.*(z.*z+zz.*zz) + suma) + dsig*randn(N,2);
            if abs(mod(t,TR))<0.01
                nn=nn+1;
                xs(nn,:)=z(:,1)';
            end
        end
        
        %%%%
        simulated_signal=xs';
        signal_filt22=zeros(N,nn);
        for seed=1:N
            simulated_signal(seed,:)=demean(detrend(simulated_signal(seed,:)));
            signal_filt22(seed,:)=filtfilt(bfilt,afilt,simulated_signal(seed,:));
        end
        signal_filt=signal_filt22(:,20:end-20);
        
        for i=1:N
            for j=1:N
                if (nmask(i,j))
                    GCsim2(sub,i,j)=gcmi_cc(signal_filt22(i,:)',signal_filt22(j,:)');
                else
                    GCsim2(sub,i,j)=0;
                end
            end
        end
    end
    
    fcsimul=squeeze(mean(GCsim2,1));
    fcsimuli(iter,:,:)= fcsimul - diag(diag(fcsimul)); % diagonal values to zero
    fittFC(iter)=sqrt(nanmean((FCemp(Isubdiag)-fcsimul(Isubdiag)).^2)); %fitting
    
    
    corr2FCSC(iter)=corr2(fcsimul(Isubdiag),C(Isubdiag));
    fc=FCemp(Isubdiag);
    idx=find(isnan(fc));
    fc(idx)=[];
    fcsimdia=fcsimul(Isubdiag);
    fcsimdia(idx)=[];
    corr2FCemp_sim(iter)=corr2(fcsimdia,fc);

    
    % effective connectivity
    for i=1:N
        for j=1:N
            if(nmask(i,j))
                if (C(i,j)>0 || j==N/2+i)
                    Cnew(i,j)=Cnew(i,j)+0.005*(FCemp(i,j)-fcsimul(i,j));
                    if (Cnew(i,j)<0 || isnan(Cnew(i,j)))
                        Cnew(i,j)=0;
                    end
                end
            end
        end
    end
    
    Cnew= Cnew/max(max(Cnew))*0.2; % scale effective connectivity to a maximum value of 0.2    
end

save (sprintf('results_eff_hopf_GCMI_%s_%s_%s.mat', num2str(condition),num2str(reaction), num2str(simulation)),'fcsimul','fcsimuli','FCemp', ...
    'corr2FCSC','corr2FCemp_sim','fittFC','Cnew');
end