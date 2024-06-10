clear; close all;
load('processed_data_2.mat');
rng(42);

%Downsample data from 200Hz to 10Hz
p1=p1(:,1:20:end);
p2=p2(:,1:20:end);
p3=p3(:,1:20:end);
q1=q1(:,1:20:end);
q2=q2(:,1:20:end);
q3=q3(:,1:20:end);
y=y(1:20:end,:);
y=y(:);

%Set map area
p=[p1 p2 p3];
N=size(p,2);

marg=3;
xl=min(p(1,:))-marg;
xu=max(p(1,:))+marg;
yl=min(p(2,:))-marg;
yu=max(p(2,:))+marg;
zl=min(p(3,:))-marg;
zu=max(p(3,:))+marg;

%Define GP Prior
sigma_SE=0.074453;
l_SE=0.86365;
sigma_y=0.004159;
N_m=100;

%Prior covariance
[Indices, Lambda]=Lambda3D(N_m,xl,xu,yl,yu,zl,zu,sigma_SE,l_SE);

testpointsx=(xu-xl-2*marg)*rand(100,1)+xl+marg;
testpointsy=(yu-yl-2*marg)*rand(100,1)+yl+marg;
testpointsz=(zu-zl-2*marg)*rand(100,1)+zl+marg;
ps=[testpointsx, testpointsy, testpointsz]';

%Reduced rank estimate
phi=Phi3D(p,N_m,xl,xu,yl,yu,zl,zu,Indices);
phis=Phi3D(ps,N_m,xl,xu,yl,yu,zl,zu,Indices);
muR=phis*inv(phi'*phi+sigma_y^2.*inv(Lambda))*phi'*y;

N=size(p,2);
K=Kern(p,p,sigma_SE,l_SE);
Ks=Kern(ps,p,sigma_SE,l_SE);
Kss=Kern(ps,ps,sigma_SE,l_SE);
mu=Ks*((K+sigma_y^2.*eye(N))\y);


RMSE_to_noise_ratio=sqrt(mean((mu-muR).^2))./sigma_y;

