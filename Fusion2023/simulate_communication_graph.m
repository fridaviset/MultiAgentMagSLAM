function G=simulate_communication_graph(alpha,N,n)

%Simulate the network connectivity for all node combinations at all times
G=(rand(n,n,N)>alpha);

%Ensure that the network connectivity is consistent between different
%ordering of the node indices (i.e. make G symmetric)
for i=1:n
    for j=1:i
        G(i,j,:)=G(j,i,:);
        if i==j
            G(i,j,:)=ones(N,1);
        end
    end
end
end