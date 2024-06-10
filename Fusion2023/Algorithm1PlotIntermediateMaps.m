function [x_hat,q_hat,m,P,Field,Var]=Algorithm1PlotIntermediateMaps(delta_x,delta_q,y,Lambda,x_0,q_0,sigma_y,sigma_p,sigma_q,x_DR,PhiVec,n,xl,xu,yl,yu,zl,zu,Indices,X,Y,x,marg)

N=length(delta_x);
N_m=size(Lambda,1);

%Initialise the linearisation points
x_hat=zeros(3,N,n);
q_hat=zeros(4,N,n);
x_hat(:,1,:)=x_0;
q_hat(:,1,:)=q_0;
P=[zeros(6*n), zeros(6*n,N_m);
    zeros(N_m,6*n), Lambda];
m=zeros(N_m,1);


for t=2:N
    
    %Integrate the odometry
    for i=1:n
        q_hat(:,t,i)=exp_q_L(delta_q(:,t,i),q_hat(:,t-1,i));
        x_hat(:,t,i)=x_hat(:,t-1,i)+quat2Rot(q_hat(:,t,i))*delta_x(:,t,i);
    end
    F=eye(3);
    for i=1:n
        Q_p=sigma_p^2*eye(3);
        Q_q=sigma_q^2*eye(3);
        Q=[Q_p, zeros(3);...
            zeros(3), Q_q];
        wi=x_hat(:,t,i);
        p_x=[0, -wi(3), wi(2);...
            wi(3), 0, -wi(1);...
            -wi(2), wi(1), 0];
        F=[eye(3) quat2Rot(q_hat(:,t,i))*p_x;
            zeros(3), eye(3)];
        pi=3*(i-1)+1:3*i;
        qi=3*(i+n-1)+1:3*(i+n);
        Pi=P([pi qi],[pi qi]);
        Pi=F*Pi*F'+Q;
        P([pi qi],[pi qi])=Pi;
    end
    
    %One measurement at a time
    for j=1:n
        f=zeros(3*n,1);
        f(3*(j-1)+1:3*(j-1)+3)=Nabla_Phi3D(x_hat(:,t,j),N_m,xl,xu,yl,yu,zl,zu,Indices)*m;
        Phi_t=Phi3D(x_hat(:,t,j),N_m,xl,xu,yl,yu,zl,zu,Indices);
        H_t=[f', zeros(3*n,1)', Phi_t];
        S_t=H_t*P*H_t'+sigma_y^2;
        K_t=P*H_t'*inv(S_t);
        y_hat=Phi_t*m;
        eps=K_t*(y(t,j)'-y_hat);
        m=m+eps(6*n+1:N_m+6*n);
        for i=1:n
            x_hat(:,t,i)=x_hat(:,t,i)+reshape(eps(3*(i-1)+1:3*(i-1)+3),3,1,1);
            eta_q=reshape(eps(3*n+3*(i-1)+1:3*n+3*(i-1)+3),3,1,1);
            q_hat(:,t)=exp_q_L(eta_q,q_hat(:,t));
        end
        P=P-K_t*S_t*K_t';
        P=1/2*(P+P');
    end
    
    
    %Plot the estimated map of agent 1 at time t
    if t==20 ||  t==75 || t==200 || t==N
        figure; clf;
        P_m=P(6*n+1:6*n+N_m,6*n+1:6*n+N_m);
        FieldVec=PhiVec*m;
        N_plots=size(FieldVec,1);
        VarVec=zeros(N_plots,1);
        for i=1:N_plots
            VarVec(i)=PhiVec(i,:)*P_m*PhiVec(i,:)';
        end
        Field=reshape(FieldVec,size(X));
        Var=reshape(VarVec,size(X));
        colormap(viridis());
        surf(X,Y,0.*Field,Field,'AlphaData',-Var,'FaceAlpha','flat','EdgeColor','none');
        view(2);
        caxis([-0.15 0.15]);
        hold on;
        for i=1:n
            plot3(x_hat(1,1:t,i),x_hat(2,1:t,i),x_hat(3,1:t,i),'k','linewidth',1);
            scatter3(x_hat(1,t,i),x_hat(2,t,i),x_hat(3,t,i),'k+','linewidth',1.2);
        end
        axis off;
        axis equal;
        xlim([xl+0.5*marg xu-0.5*marg]);
        ylim([yl+0.5*marg yu-0.5*marg]);
        exportgraphics(gca,['Illustrations/OptiTrackTime',num2str(t),'.png'],'Resolution',800);
    end
    
end

%Make a colorbar plot
figure; clf;
axis off;
caxis([-0.15 0.15]);
colormap(viridis());
c=colorbar;
c.LineWidth=2;
c.Position=[0.8 0.3 0.03 0.4];
set(c,'YTick',[-0.15 0 0.15],'TickLabelInterpreter','latex','Fontsize',14,'Color', 'k');
exportgraphics(gca,'Illustrations/ColorBar.png','Resolution',500);

end