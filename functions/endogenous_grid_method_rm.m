function [c_policy_mat, n_policy_mat, a_policy_mat, a_grid] = ...
    endogenous_grid_method_rm(transition, theta, beta, gamma, r, phi, ...
                             psi, eta, v, tao, tol)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Endogenous grid method with elastic labor supply
% By Robert A. McDowall
% Input:  transition (matrix): income transition matrix 
%         theta (vector): state space for income realizations
%         r - rate of return on liquid assets
%         preference parameters: 
%                           beta - discount factor
%                           gamma - risk aversion
%                           phi - ad-hoc borrowing constraint
%                           psi - coefficient on liesure utility
%                           eta - curvature of liesure utility
%                           v - lump-sum unemployment transfers
%                           tao - lump-sum tax faced in employment states
%         tol: convergence tolerance
% Output: c_policy_mat: consumption policy function matrix
%         n_policy_mat: labor policy function matrix
%         a_policy_mat: asset policy function matrix 
%         a_grid: fixed asset grid
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
options = optimset('Display', 'off') ;
options.LargeScale = 'off';
options.Display = 'off';
% Set up inputs
min_cons = 0.001;
S = length(theta);
asset_low = floor(-phi*1.1); asset_up = ceil(mean(theta)*25);

a_grid = [asset_low:.25:asset_up]; % Set up asset grid
a_y_grid = repmat(a_grid', 1, S); % (a,y) grid, repeated a_grid, endogenous grid
convg_low = find(a_grid >= -phi, 1); %Bound on interior to check for convergence
mat_init = zeros(length(a_grid),length(theta));

%unemployment transfer matrix
v_mat = [repmat(v, length(a_grid),1), zeros(length(a_grid), length(theta)-1)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%First Order conditions, called functions, and initial guess
du_dc = @(c) c.^(-gamma); %FOC for consumption
n_func = @(du_dc, theta_t) max(0, 1 - ...
    ((theta_t.*du_dc)./(psi)).^(-1/eta)); % Closed form solution for labor

%Intial guess: c_0 = r*a_i + y_j
c = mat_init;    
for i = 1:length(a_grid)
    for j = 1:length(theta)           
            c(i,j) = max(min_cons, r*a_grid(i) + theta(j));   
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%LOOP SECTION
dist = 1; %initial distance, will converge on each loop
iteration = 1; %loop counter
 
while dist>tol
    %%%%%%%%%%%%%%%%%%
    % Construct RHS of Euler Equations for each y_j        
    B = mat_init;      
    for i = 1:length(a_grid) 
       for j = 1:length(theta)
        B(i,j) = beta*(1+r)*transition(j,:)*du_dc(c(i,:)');
       end
    end
    %%%%%%%%%%%%%%%%%%  
    % Solve analytically for consumption and labor
    c_prev = c; %store previous iteration for convergence test
    c_temp = max(min_cons,(B).^(-1/gamma)); %implied optimal consumption
    
    FOC_temp = du_dc(c_temp);
    n_temp = mat_init;
    for i = 1:length(a_grid) %implied  opitimal labor
       for j = 1:length(theta)
            n_temp(i,j) = n_func(FOC_temp(i,j), theta(j));
       end
    end
   
    % Use budget constraint to solve for implied assets today
    y_mat = repmat(theta, length(a_grid),1);
    a_y_grid_lag = (1/(1+r)).*(c_temp+tao+a_y_grid-y_mat.*n_temp - v_mat); 

    % Binding borrowing constraint condition
    a_primebinds = a_y_grid <= -phi;    
    %find levels of assets today that induce constraint to bind tomorrow 
    a_y_grid_lag_constr = zeros(length(theta)); 
    for j=1:length(theta)
        a_y_grid_lag_col = a_y_grid_lag(:,j);
        a_y_grid_lag_constr(j) = max(a_y_grid_lag_col(a_primebinds(:,j))');  
    end 
    
    %%%%%%%%%%%%%%%%%% 
    % Interpolatation step - check if constraint binds, then interpolate
    for i = 1:length(a_grid)
    for j =1:length(theta)
        
            if a_grid(i) > a_y_grid_lag_constr(j) 
               for k =1:length(a_grid)-1
                    
                    if a_y_grid_lag(k,j) <= a_grid(i) && a_grid(i) <= a_y_grid_lag(k+1,j) 
                        c(i,j) = ...
                            interp1([a_y_grid_lag(k,j), a_y_grid_lag(k+1,j)]',...
                            [c_temp(k,j), c_temp(k+1,j)]', a_grid(i), 'linear', 'extrap'); 
                    elseif max(a_y_grid_lag(:,j)) < a_grid(i)       
                        c(i,j) = ...
                            interp1([a_y_grid_lag(length(a_grid)-1,j), a_y_grid_lag(length(a_grid),j)]',...
                            [c_temp(length(a_grid)-1,j), c_temp(length(a_grid),j)]', a_grid(i), 'linear', 'extrap');
                    end               
               end
                
            elseif a_grid(i) <= a_y_grid_lag_constr(j)
                lowercons = @(x) x - ((1+r)*a_grid(i) ...
                   + theta(j)*n_func(du_dc(x), theta(j)) ...
                   - tao + v_mat(i,j) - (-phi));
                c(i,j) = max(min_cons, fsolve(lowercons, .1, options));
                
            end            
    end
    end
    %%%%%%%%%%%%%%%%%% 
    % Check for convergence
    dist = norm(c(convg_low:end-1,1:end-1)-c_prev(convg_low:end-1,1:end-1));
    iteration = iteration + 1;

    ['This is iteration: ' num2str(iteration)]
    [ 'Distance is: ' num2str(dist)]
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OUTPUTS 
n_policy_mat = mat_init;
a_policy_mat = mat_init;
c_policy_mat = c(:,:,end);

fonc = du_dc(c_policy_mat);
for i = 1:length(a_grid) %calculate implied labor policy matrix
    for j = 1:length(theta)
        n_policy_mat(i,j) = n_func(fonc(i,j), theta(j));        
        a_policy_mat(i,j) = (1+r)*a_grid(i) + theta(j)*n_policy_mat(i,j)...
                            - c_policy_mat(i,j)  - tao + v_mat(i,j);
    end
end

end
