function [transition, theta, uncond_dist] = ...
                tauchen_rm(rho, sigma, m, n, pi_e_u, pi_u_e, ui)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Discretization of AR(1) income process w/ unemployment state
%Method due to Tauchen (1986)
% By: Robert A. McDowall
% Inputs:  rho (float): AR(1) parameter rho, 
%          sigma (float): variance of random innovation
%          m (float): multiple of unconditional stnd dev to discretize
%          n (int): size of discretization 
%          pi_e_u, pi_u_e (float): employment (-> / <-)unemployment 
%                                  probabilities, respectively
%          ui (float) = proportion of average income in unemployment state  
% Output: nxn transition matrix (w/unemployed state), state space,
%          unconditional distribution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set up inputs
n = n-1; %drop one state, unemployment state added below
temp_mat = zeros(n); %initialize matrix to fill in with transitions
tol = 1*10^-3; 

% Get grid points
max_y = m * (sigma/(1-rho^2))^(1/2);
min_y = -max_y;
y = linspace(min_y,max_y,n); 
% Calculate distace between points in state space
d = y(n)- y(n-1);

% Calculate Transition Probabilities and fill in matrix
for j = 1:length(y)     
    bound_1 = (y(1) + (d/2) - rho*y(j))/sqrt(sigma);
    bound_n = (y(n) - (d/2) - rho*y(j))/sqrt(sigma);    
    temp_mat(j,1) = normcdf(bound_1);
    temp_mat(j,n) = 1-normcdf(bound_n);
    
    for k = 2:(n-1)       
        x_1 = (y(k) + (d/2) - rho*y(j))/sqrt(sigma);
        x_2 = (y(k) - (d/2) - rho*y(j))/sqrt(sigma);       
        temp_mat(j,k) = normcdf(x_1) - normcdf(x_2);        
    end
end

%rescale
theta_temp = exp(y);
S = length(theta_temp+1);

% compute invariant distribution for theta  
pr = ones(1,S)/S; dis = 1;      
while dis>tol
    pr_temp = pr*temp_mat; 
    dis = max(abs(pr_temp - pr)); 
    pr = pr_temp;
end

%OUTPUTS%%%%%%%%%%%
%add unemployment to statespace 
theta = [ui*mean(theta_temp) theta_temp]; %OUTPUT 1

%add unemployment to transition matrix, now n states
transition_top = [(1-pi_u_e), pi_u_e*pr];
transition_rest = [pi_e_u*ones(n,1), (1-pi_e_u)*temp_mat];
transition = [transition_top; transition_rest]; %OUTPUT 2
 
uncond_dist_temp = transition^1000; %decent approximation of limiting dist'n
uncond_dist = uncond_dist_temp(1,:); %OUTPUT 3

end
