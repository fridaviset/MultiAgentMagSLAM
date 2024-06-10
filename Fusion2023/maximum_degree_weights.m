function W=maximum_degree_weights(G)
%Compute maximum degree weigths W based on communication graph G

N=size(G,3);
n=size(G,1);

%Allocate storage for the weights
W=zeros(n,n,N);

for t=1:N
    for i=1:n
        %Calculate the weights according to the maximum-degree weights
        d_it=0;
        for j=1:n
            if i~=j
                if G(i,j,t)
                    %Node i and j are connected
                    W(i,j,t)=1./n;
                    
                    %Increment the count of nodes connected to i
                    d_it=d_it+1;
                else
                    %Node i and j are not connected
                    W(i,j,t)=0;
                end
            end
        end
        W(i,i,t)=1-d_it./n;
    end
end

end