[X1,X2] = meshgrid(0:0.05:1, 0:0.05:1);

load('processed_data_2.mat');

%Downsample data from 200Hz to 10Hz
p1=p1(:,1:20:end);
p2=p2(:,1:20:end);
p3=p3(:,1:20:end);
q1=q1(:,1:20:end);
q2=q2(:,1:20:end);
q3=q3(:,1:20:end);
y=y(1:20:end,:);

x=[p1 p2 p3]';
y=y(:);

covfunc = @covSEiso; 
likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);

hyp2.cov = [0 ; 0];    
hyp2.lik = log(0.1);
hyp2 = minimize(hyp2, @gp, -100, @infExact, [], covfunc, likfunc, x, y);
exp(hyp2.lik)
nlml2 = gp(hyp2, @infExact, [], covfunc, likfunc, x, y)

[m s2] = gp(hyp2, @infExact, [], covfunc, likfunc, x, y, x);


sigma_SEl=exp(hyp2.cov(2));
l_SEl=exp(hyp2.cov(1));
sigma_yl=exp(hyp2.lik);
disp(['sigma_SE=',num2str(sigma_SEl),';']);
disp(['l_SE=',num2str(l_SEl),';']);
disp(['sigma_y=',num2str(sigma_yl),';']);