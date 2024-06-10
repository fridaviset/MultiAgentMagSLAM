function [x_hat,q_hat,m,P]=Algorithm2(delta_x,delta_q,y,Lambda,x_0,q_0,sigma_y,sigma_p,sigma_q,n,xl,xu,yl,yu,zl,zu,Indices,alpha,N_comm)

N=length(delta_x);
N_m=size(Lambda,1);

%Initialise the linearisation points
x_hat=zeros(3,N,n);
q_hat=zeros(4,N,n);
x_hat(:,1,:)=x_0;
q_hat(:,1,:)=q_0;
P=zeros(6*n+N_m,6*n+N_m,n);
for i=1:n
    P(:,:,i)=[zeros(6*n), zeros(6*n,N_m);
        zeros(N_m,6*n), Lambda];
    
end
m=zeros(N_m,n);
I=zeros(6*n+N_m,6*n+N_m,n);

for t=2:N
    
    %Integrate the odometry
    for i=1:n
        q_hat(:,t,i)=exp_q_L(delta_q(:,t,i),q_hat(:,t-1,i));
        x_hat(:,t,i)=x_hat(:,t-1,i)+quat2Rot(q_hat(:,t,i))*delta_x(:,t,i);
    end
    
    F_all=zeros(N_m+6*n,N_m+6*n,n);
    for i=1:n
        F_all(:,:,i)=eye(N_m+6*n,N_m+6*n);
    end
    Q_all=zeros(N_m+6*n,N_m+6*n,n);
    for i=1:n
        Q_p=sigma_p^2*eye(3);
        Q_q=sigma_q^2*eye(3);
        Qi=[Q_p, zeros(3);...
            zeros(3), Q_q];
        wi=x_hat(:,t,i);
        p_x=[0, -wi(3), wi(2);...
            wi(3), 0, -wi(1);...
            -wi(2), wi(1), 0];
        Fi=[eye(3) n*quat2Rot(q_hat(:,t,i))*p_x;...
            zeros(3), eye(3)];
        pi=3*(i-1)+1:3*i;
        qi=3*(i+n-1)+1:3*(i+n);
        F_all([pi qi],[pi qi],i)=Fi;
        Q_all([pi qi],[pi qi],i)=n*Qi;
    end
    
    %%Simulate a communication graph
    G=simulate_communication_graph(alpha,N_comm,n);
    
    %%Compute the average consensus weights
    W=maximum_degree_weights(G);
    
    %Execute the average consensus algorithm
    [F_all]=avg_consensus(F_all,zeros(N_m+6*n,n),W);
    [Q_all]=avg_consensus(Q_all,zeros(N_m+6*n,n),W);
    Q=mean(Q_all,3);
    
    for i=1:n
        F=F_all(:,:,i);
        P(:,:,i)=F*P(:,:,i)*F'+Q;
    end
    
    iota=zeros(6*n+N_m,n);
    for i=1:n
        I(:,:,i)=inv(P(:,:,i));
    end

    for j=1:n
        f=zeros(3*n,1);
        f(3*(j-1)+1:3*(j-1)+3)=Nabla_Phi3D(x_hat(:,t,j),N_m,xl,xu,yl,yu,zl,zu,Indices)*m(:,j);
        Phi_t=Phi3D(x_hat(:,t,j),N_m,xl,xu,yl,yu,zl,zu,Indices);
        H_t=[f', zeros(3*n,1)', Phi_t];
        I(:,:,j)=I(:,:,j)+n./sigma_y^2*(H_t'*H_t);
        y_hat=Phi_t*m(:,j);
        iota(:,j)=n*1./sigma_y^2*H_t'*(y(t,j)'-y_hat);
    end

    %Simulate a communication graph
    G=simulate_communication_graph(alpha,N_comm,n);
    
    %%Compute the average consensus weights
    W=maximum_degree_weights(G);
    
    %Execute the average consensus algorithm
    [I,iota]=avg_consensus(I,iota,W);
    
    eps=zeros(N_m+6*n,n);
    for i=1:n
        P(:,:,i)=inv(I(:,:,i));
        eps(:,i)=I(:,:,i)\iota(:,i);
        P(:,:,i)=1/2*(P(:,:,i)+P(:,:,i)');
    end
    %Let each agent update their own belief about their state
    for i=1:n
        x_hat(:,t,i)=x_hat(:,t,i)+reshape(eps(3*(i-1)+1:3*(i-1)+3,1),3,1,1);
        eta_q=reshape(eps(3*n+3*(i-1)+1:3*n+3*(i-1)+3,1),3,1,1);
        q_hat(:,t,i)=exp_q_L(eta_q,q_hat(:,t,i));
        m(:,i)=m(:,i)+eps(6*n+1:N_m+6*n,i);
    end
    
   
end

 m=m(:,1);
end