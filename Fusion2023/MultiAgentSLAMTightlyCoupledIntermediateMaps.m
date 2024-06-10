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

%Set map area
N=size(p1,2);
n=3;
p=[p1 p2 p3];
x=zeros(3,N,n);
q=zeros(4,N,n);
x(:,:,1)=p1;
x(:,:,2)=p2;
x(:,:,3)=p3;
q(:,:,1)=q1;
q(:,:,2)=q2;
q(:,:,3)=q3;

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

alpha=0.2; %Percentage probability communication failure

%Prior covariance
[Indices, Lambda]=Lambda3D(N_m,xl,xu,yl,yu,zl,zu,sigma_SE,l_SE);

%Allocate space for the information matrix and the information vector
iota=zeros(N_m,1);
I=zeros(N_m,N_m);
rtime=eye(n);

%Prepare plots
z=mean([zl zu]);
res=0.05;
[X,Y]=meshgrid(xl:res:xu,yl:res:yu);
Z=z*ones(size(X));
pointsVec=[X(:),Y(:),Z(:)]';
PhiVec=Phi3D(pointsVec,N_m,xl,xu,yl,yu,zl,zu,Indices);

%Simulate odometry based on the optitrack positions
delta_x=zeros(3,N,n);
sigma_p=0.0022;
bias=zeros(n,1,3);
bias(:,1,1)=[0; 0.001; 0];
bias(:,1,2)=[-0.001; -0.0005; 0];
bias(:,1,3)=[0.001;  -0.0005; 0];

%Deduce some orientation odometry
delta_q=zeros(3,N);
sigma_q=0.000001;
Q_q=sigma_q^2*eye(3);
for t=2:N
    delta_x(:,t,:)=x(:,t,:)-x(:,t-1,:)+bias;
    for i=1:n
        delta_x(:,t,i)=quat2Rot(q(:,t,i))'*delta_x(:,t,i);
        q_t_C=[-q(1,t-1,i); q(2:4,t-1,i)];
        delta_q(:,t,i)=quat2angleaxis(quatprod(q(:,t,i),q_t_C))+reshape(mvnrnd(0,sigma_q^2,3),3,1);
    end
end

%Initialize the position estimates
x_DR=zeros(3,N,n);
q_DR=zeros(4,N,n);
x_DR(:,1,:)=x(:,1,:);
q_DR(:,1,:)=q(:,1,:);
x_0=x(:,1,:);
q_0=q(:,1,:);

for t=2:N
    %Compare with DR
    for i=1:n
        q_DR(:,t,i)=exp_q_L(delta_q(:,t,i),q_DR(:,t-1,i));
        x_DR(:,t,i)=x_DR(:,t-1,i)+quat2Rot(q_DR(:,t,i))*delta_x(:,t,i);
    end
end

N_comm=10; %Number of communication steps for average consensus

[x_hat,q_hat,m,P]=Algorithm2PlotIntermediateMaps(delta_x,delta_q,y,Lambda,x_0,q_0,sigma_y,sigma_p,sigma_q,n,xl,xu,yl,yu,zl,zu,Indices,alpha,N_comm,PhiVec,X,Y,marg);

%Compare with single-agent SLAM
x_hat_single_agents=zeros(3,N,n);
for i=1:n
[x_hat_i]=Algorithm1(delta_x(:,:,i),delta_q(:,:,i),y(:,i),Lambda,x_0(:,:,i),q_0(:,:,i),sigma_y,sigma_p,sigma_q,1,xl,xu,yl,yu,zl,zu,Indices);
x_hat_single_agents(:,:,i)=x_hat_i;
end

%Retrieve the magnetic field estimate
P_m=P(6*n+1:6*n+N_m,6*n+1:6*n+N_m);
FieldVec=PhiVec*m;
N_plots=size(FieldVec,1);
VarVec=zeros(N_plots,1);
for i=1:N_plots
    VarVec(i)=PhiVec(i,:)*P_m*PhiVec(i,:)';
end
Field=reshape(FieldVec,size(X));
Var=reshape(VarVec,size(X));

%% Plot results


A=viridis();
mustard=[240, 176, 0]./255;
black=[0, 0, 0]./255;
green=[0.1263, 0.6441, 0.5253];
%Normalise Y data
mu_normalised=reshape(Field,size(X))-min(min(reshape(Field,size(X))));
mu_normalised=mu_normalised./max(max(mu_normalised));
indices=ceil((mu_normalised).*254+1);

%Normalise Opacity
variance_normalized=reshape(Var-min(min(Var)),size(X));
variance_normalized=variance_normalized./max(max(max(variance_normalized)),sigma_SE^2+sigma_y^2);

%Quick calculation
ColorPicture=reshape(A(indices,:),size(mu_normalised,1),size(mu_normalised,2),3);
ColorPicture=flip(ColorPicture,1);
variance_normalized=flip(variance_normalized,1);
imagename='SingleTilePicture';
imwrite(ColorPicture,[imagename,'.png'],'Alpha',(1-variance_normalized));

figure; clf;
Ts=0.1;
fontsize=12;
time=Ts:Ts:N*Ts;
errors_multi=zeros(N,3);
errors_single=zeros(N,3);
errors_DR=zeros(N,3);
for i=1:n
    set(gcf,'Position',[100 100 450 700]);
    subplot(n+1,1,i);
    errors_multi(:,i)=reshape(sqrt(sum((x(:,1:N,i)-x_hat(:,:,i)).^2,1)),N,1);
    plot(time,errors_multi(:,i),'Color',black,'linewidth',1.2);
    hold on;
    errors_single(:,i)=reshape(sqrt(sum((x(:,1:N,i)-x_hat_single_agents(:,:,i)).^2,1)),N,1);
    plot(time,errors_single(:,i),'Color',green,'linewidth',1.2);
    hold on;
    errors_DR(:,i)=reshape(sqrt(sum((x(:,1:N,i)-x_DR(:,:,i)).^2,1)),N,1);
    plot(time,errors_DR(:,i),'r','linewidth',1.2);
    box off;
    xlabel('time (s)');
    ylabel('$\|\hat{p}_t-p_t\|_2$','Interpreter','Latex');
    title(['Agent ',num2str(i)]);
set(gca, 'FontName', 'Times');
set(gca,'fontsize',fontsize);
set(gca,'TickLabelInterpreter','latex');
grid on;
exportgraphics(gca,['Illustrations/OptiTrackPositionErrors',num2str(i),'.png'],'Resolution',500);
end
subplot(n+1,1,n+1);
plot(1,NaN,'Color',black,'linewidth',1.2); hold on;
plot(1,NaN,'Color',green,'linewidth',1.2);
plot(1,NaN,'r','linewidth',1.2);
legend('Algorithm 2','Single agent SLAM','Odometry','Location','North');
set(gca, 'FontName', 'Times');
box off;
axis off;
set(gca,'fontsize',fontsize);
exportgraphics(gca,'Illustrations/OptiTrackPositionErrorsLegends.png','Resolution',500);





