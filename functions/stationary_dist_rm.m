function [stationary_dist] = ...
                stationary_dist_rm(transition, a_policy_mat, a_grid)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Stationary Distribution Calculator
% Inputs: transition (matrix): income transition matrix
%         a_policy_mat: asset policy function matrix 
%         a_grid: fixed asset grid
% Outputs: stationary distribution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get dimensions
m = size(a_policy_mat,1); %dim of asset grid
n = size(a_policy_mat,2); %dim of income shocks
N=m*n; 

% Vectorize
a_reshape = reshape(a_policy_mat, N, 1);   
fspaceerg = fundef({'spli', a_grid, 0, 1});
Q_x = funbas(fspaceerg, a_reshape);

% Calculate stationary distribution
Q_theta=kron(transition,ones(m,1));
col = 1;
Q = zeros(N,N);
for the = 1:n
    for a = 1:m       
        Q(:,col) = Q_x(:,a).*Q_theta(:,the);        
        col = col + 1;
    end
end
[~,D,W] = eig(Q); D = real(D); W = real(W);
index = find(diag(D) > 0.9999, 1, 'last');
density_st = max(W(:,index), 0);

%Normalize and reshape for output
stationary_dist=density_st./sum(density_st); 
stationary_dist=reshape(stationary_dist,[m,n]);

end