function [I, iota]=avg_consensus(I,iota,W)
%This function simulates the execution of average consensus between n
%agents

%I - n times N times N information matrix, where n is the number of agents, and N
%is the dimension of the state-space to be estimated
%iota - n times N information vector
%W - n times n times N_comm matrix, which described for the communication
%perios consisting of N_comm interactions, which of the n agents can
%communicate with the n other agents

N_comm=size(W,3);
N=size(iota,1);
n=size(iota,2);
% for t=1:N_comm
%     W_t=kron(eye(N),W(:,:,t));
%     iota=reshape(W_t*reshape(iota',N*n,1),n,N)';
%     for k1=1:N
%     I(k1,:,:)=permute(reshape(W_t*reshape(permute(I(k1,:,:),[3 1 2]),N*n,1),n,N,1),[2 3 1]);
%     end
% end

for t=1:N_comm
    W_t=W(:,:,t);
    iota=(W_t*iota')';
    I=reshape((W_t*(reshape(I,N*N,n))')',N,N,n);
end

end