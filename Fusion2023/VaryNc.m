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

%Prior covariance
[Indices, Lambda]=Lambda3D(N_m,xl,xu,yl,yu,zl,zu,sigma_SE,l_SE);

experiments=100;
alphas=0.0:0.1:1; %Percentage probability communication failire
N_cs=1:2:10;
params=length(alphas);
param2s=length(N_cs);
DeviationsDistributed=zeros(experiments,params,param2s);
DeviationsSingleAgent=zeros(experiments,params,param2s);
RMSEDistributed=zeros(experiments,params,param2s);
RMSESingleAgent=zeros(experiments,params,param2s);
RMSECentralized=zeros(experiments,params,param2s);

for experiment=1:experiments
    
    %Simulate odometry based on the optitrack positions
    delta_x=zeros(3,N,n);
    sigma_p=0.006;
    bias=zeros(n,1,3);
    bias(:,1,1)=[0.000; 0.001; 0];
    bias(:,1,2)=[-0.001; -0.0005; 0];
    bias(:,1,3)=[0.001;  -0.0005; 0];
    
    %Deduce some orientation odometry
    delta_q=zeros(3,N);
    sigma_q=0.00001;
    Q_q=sigma_q^2*eye(3);
    for t=2:N
        delta_x(:,t,:)=x(:,t,:)-x(:,t-1,:)+reshape(mvnrnd(0,0.1*sigma_p^2,3*n),3,1,n)+bias;
        for i=1:n
            delta_x(:,t,i)=quat2Rot(q(:,t,i))'*delta_x(:,t,i);
            q_t_C=[-q(1,t-1,i); q(2:4,t-1,i)];
            delta_q(:,t,i)=quat2angleaxis(quatprod(q(:,t,i),q_t_C))+reshape(mvnrnd(0,sigma_q^2,3),3,1);
        end
    end
    
    for param=1:params
        for param2=1:param2s
            
            tic;
            
            alpha=alphas(param);
            
            %Allocate space for the information matrix and the information vector
            iota=zeros(N_m,1);
            I=zeros(N_m,N_m);
            rtime=eye(n);
            
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
            
            N_comm=N_cs(param2); %Number of communication steps for average consensus
            
            [x_hat,q_hat,m,P]=Algorithm2(delta_x,delta_q,y,Lambda,x_0,q_0,sigma_y,sigma_p,sigma_q,n,xl,xu,yl,yu,zl,zu,Indices,alpha,N_comm);
            [x_c]=Algorithm1(delta_x,delta_q,y,Lambda,x_0,q_0,sigma_y,sigma_p,sigma_q,n,xl,xu,yl,yu,zl,zu,Indices);
            
            %Compare with single-agent SLAM
            x_hat_single_agents=zeros(3,N,n);
            for i=1:n
                [x_hat_i]=Algorithm1(delta_x(:,:,i),delta_q(:,:,i),y(:,i),Lambda,x_0(:,:,i),q_0(:,:,i),sigma_y,sigma_p,sigma_q,1,xl,xu,yl,yu,zl,zu,Indices);
                x_hat_single_agents(:,:,i)=x_hat_i;
            end
            
            deviationdist=sqrt(mean(mean(sum((x_c(:,1:N,:)-x_hat(:,:,:)).^2,1),3)));
            deviationSingle=sqrt(mean(mean(sum((x_c(:,1:N,:)-x_hat_single_agents(:,:,:)).^2,1),3)));
            rmsedist=sqrt(mean(mean(sum((x(:,1:N,:)-x_hat(:,:,:)).^2,1),3)));
            rmseSingle=sqrt(mean(mean(sum((x_c(:,1:N,:)-x_hat_single_agents(:,:,:)).^2,1),3)));
            rmseCentralized=sqrt(mean(mean(sum((x(:,1:N,:)-x_c(:,:,:)).^2,1),3)));
            
            DeviationsDistributed(experiment,param,param2)=deviationdist;
            DeviationsSingleAgent(experiment,param,param2)=deviationSingle;
            
            RMSEDistributed(experiment,param,param2)=rmsedist;
            RMSESingleAgent(experiment,param,param2)=rmseSingle;
            RMSECentralized(experiment,param,param2)=rmseCentralized;
            
            
            toc;
            
            disp(['Experiment: ',num2str(experiment),'/',num2str(experiments),', param:',num2str(param),'/',num2str(params),' param2:',num2str(param2),'/',num2str(param2s)']);
            disp(['Error distributed: ',num2str(deviationdist),', error single: ',num2str(deviationSingle)]);
        end
    end
    
end

currenttime=clock;
save(['Results-',num2str(currenttime(3)),'-',num2str(currenttime(2)),'-',num2str(currenttime(4)),num2str(currenttime(5)),'.mat']);

%% Plot results
%The results in the Result-folder was used in the paper!

figure; clf;
plum=[214, 50, 217]./255;
lightblue=[0.1263, 0.6441, 0.5253];
white=[1 1 1];
legends={};
for param2=1:param2s
    value=((param2-1)./param2s)*0.85;
    hue=166./360;
    color=hsv2rgb([hue, 1, value]);
    legends{param2}=['$N_c=',num2str(N_cs(param2)),'$'];
    errorbar(1-alphas,mean(DeviationsDistributed(:,:,param2)),std(DeviationsDistributed(:,:,param2)),'Color',color,'linewidth',1.1);
    hold on;
end
xlabel('$(1-\alpha)$','Interpreter','Latex','Fontsize',14);
ylabel('Difference to centralized estimate','Fontname','Times');
legend(legends,'Interpreter','Latex');
set(gca,'fontname','times');
set(gca,'FontSize',14);
xlim([0.1 1]);
box off;
%exportgraphics(gca,'Illustrations/MCRepetitionsVaryNc.png','Resolution',500);

figure; clf;
mean_single=mean(RMSESingleAgent(:,1,1))+0*alphas;
std_single=std(RMSESingleAgent(:,1,1))+0*alphas;
area(1-alphas,mean_single+std_single,'EdgeColor','None','FaceColor',lightblue,'FaceAlpha',0.2);
hold on;
area(1-alphas,mean_single-std_single,'EdgeColor','None','FaceColor',white);
p1=plot(1-alphas,mean_single,'Color',lightblue,'linewidth',1.3);
mean_single=mean(RMSECentralized(:,1,1))+0*alphas;
std_single=std(RMSECentralized(:,1,1))+0*alphas;
area(1-alphas,mean_single+std_single,'EdgeColor','None','FaceColor',plum,'FaceAlpha',0.2);
area(1-alphas,mean_single-std_single,'EdgeColor','None','FaceColor',white);
p2=plot(1-alphas,mean_single,'Color',plum,'linewidth',1.3);
p3=errorbar(1-alphas,mean(RMSEDistributed(:,:,1)),std(RMSEDistributed(:,:,1)),'k','linewidth',1.1);
xlabel('$(1-\alpha)$','Interpreter','Latex','Fontsize',14);
ylabel('Position RMSE','Fontname','Times');
xlim([0.1 1]);
legend([p1 p2 p3],'Single agent SLAM','Algorithm 1','Algorithm 2','Fontname','Times');
set(gca,'fontname','times');
set(gca,'FontSize',14)
box off;
%exportgraphics(gca,'Illustrations/MCRepetitionsVaryAlpha.png','Resolution',500);

