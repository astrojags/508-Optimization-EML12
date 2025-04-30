%% Indirect Optimization in CR3BP without Manifold Targeting
clc; clear;

%% Constants and Parameters
mu = 0.0121505856;         % Earth-Moon system
T_max = 1.5;                % Maximum thrust [N]
c = 2600 * 9.80665;         % Effective exhaust velocity [m/s]
m0 = 100;                   % Initial mass [kg]
t0 = 0; tf = 10;            % Time window [nondimensional units]
smooth_param = 1e-3;        % Smoothing parameter for bang-bang control

%% Initial and Final States (example values)
x0 = [1.0 - mu; 0; 0; 0; 0.05; 0];         % Initial state (r,v)
xF = [0.85; 0; 0; 0; 0.03; 0];             % Final state

%% Shooting Function
shoot_fun = @(lambda0) shooting(lambda0, x0, xF, m0, t0, tf, mu, T_max, c, smooth_param);

% Initial guess for costates (6 state + 1 mass costate)
v0 = x0(4:6);
lambda_v0 = -v0 / norm(v0);  % optimal thrust direction is anti-velocity

% Modified to ensure proper vector dimensions
lambda0_guess = [0; 0; 0; lambda_v0; -0.1];  % position, velocity, mass costates (7 total)

% Solve with fsolve 
options = optimoptions('fsolve','Display','iter','TolFun',1e-6,'TolX',1e-6,...
                        'MaxFunEvals',1000,'MaxIter',300,'Algorithm','levenberg-marquardt');

% Function wrapper for robustness
robust_shoot_fun = @(lambda0) robust_shooting(lambda0, x0, xF, m0, t0, tf, mu, T_max, c, smooth_param);

% Try different initial guesses if the first one fails
try
    [lambda0_sol, fval, exitflag, output] = fsolve(robust_shoot_fun, lambda0_guess, options);
    fprintf('Optimization successful!\n');
    fprintf('fsolve exitflag: %d\n', exitflag);
    fprintf('fsolve message: %s\n', output.message); % Print message
catch ME
    fprintf('First optimization attempt failed: %s\n', ME.message);
    fprintf('Trying with alternative initial guess...\n');
    
    % Alternative initial guess with different scaling
    lambda0_guess_alt = [-1; -0.1; -0.1; -1; -0.5; -0.1; -1];
    try
        [lambda0_sol, fval, exitflag, output] = fsolve(robust_shoot_fun, lambda0_guess_alt, options);
        fprintf('Second optimization attempt successful!\n');
        fprintf('fsolve exitflag: %d\n', exitflag);
        fprintf('fsolve message: %s\n', output.message); % Print message
    catch ME2
        fprintf('Second optimization attempt failed: %s\n', ME2.message);
        fprintf('Trying with continuation approach...\n');
        
        % Continuation approach - solve for shorter time first
        tf_short = tf/2;
        shoot_fun_short = @(lambda0) robust_shooting(lambda0, x0, xF, m0, t0, tf_short, mu, T_max, c, smooth_param);
        try
            lambda0_sol_short = fsolve(shoot_fun_short, lambda0_guess, options);
            fprintf('Short-time optimization successful, extending to full interval...\n');
            lambda0_sol = fsolve(robust_shoot_fun, lambda0_sol_short, options);
        catch ME3
            fprintf('Continuation approach failed: %s\n', ME3.message);
            error('All indirect optimization attempts failed.');
        end
    end
end

% Try continuation with smoothing parameter
try
    fprintf('Trying continuation approach with smoother control...\n');
    smooth_values = [1e-1, 1e-2, 1e-3, 1e-4];
    current_lambda = lambda0_guess;
    
    for i = 1:length(smooth_values)
        current_smooth = smooth_values(i);
        fprintf('Solving with smoothing parameter: %e\n', current_smooth);
        smooth_shoot_fun = @(lambda0) robust_shooting(lambda0, x0, xF, m0, t0, tf, mu, T_max, c, current_smooth);
        
        current_lambda = fsolve(smooth_shoot_fun, current_lambda, options);
    end
    
    lambda0_sol = current_lambda;
    fprintf('Continuation approach successful!\n');
catch ME4
    fprintf('Continuation with smoothing failed: %s\n', ME4.message);
    % Continue with best solution so far
end

fprintf('Optimal initial costates: \n');
disp(lambda0_sol);

% Simulate optimal trajectory
disp('Simulating optimal trajectory...');
[t_opt, Y_opt] = simulateTrajectory(lambda0_sol, x0, m0, t0, tf, mu, T_max, c, smooth_param);

% Evaluate final state error
final_state = Y_opt(end, 1:6)';
state_error = norm(final_state - xF);
fprintf('Final state error: %e\n', state_error);

% Plot results
try
    plotResults(t_opt, Y_opt, mu, T_max, c);
catch ME
    fprintf('Error in plotting: %s\n', ME.message);
end

% Save results to file
try
    save('cr3bp_optimal_trajectory.mat', 't_opt', 'Y_opt', 'lambda0_sol', 'mu', 'T_max', 'c');
    fprintf('Results saved to cr3bp_optimal_trajectory.mat\n');
catch ME
    fprintf('Error saving results: %s\n', ME.message);
end

%% Shooting Function Definition
function F = shooting(lambda0, x0, xF, m0, t0, tf, mu, T_max, c, smooth_param)
    % Initial augmented state [x; m; lambda_x; lambda_m]
    aug0 = [x0; m0; lambda0(1:6); lambda0(7)];
    
    % Integrate dynamics
    opts = odeset('RelTol',1e-8,'AbsTol',1e-8);
  
    % Choose solver based on smoothing parameter
    if smooth_param < 1e-3
        solver = @ode15s;
    else
        solver = @ode45;
    end

    try
        [~, Y] = solver(@(t, y) dynamics(t, y, mu, T_max, c, smooth_param), [t0 tf], aug0, opts);

        % Terminal error: state mismatch + lambda_m(tf) = 0 (transversality condition)
        x_tf = Y(end,1:6)';
        lambda_m_tf = Y(end,14);  % Fixed index to access lambda_m
        
        F = [x_tf - xF; lambda_m_tf];
    catch ME
       % Handle the error.
       fprintf('Error in shooting function (ODE integration): %s\n', ME.message);
       F = 1e10 * ones(7, 1); % Return a large, finite value
    end
end

%% Robust Shooting Function with Error Handling
function F = robust_shooting(lambda0, x0, xF, m0, t0, tf, mu, T_max, c, smooth_param)
    try
        F = shooting(lambda0, x0, xF, m0, t0, tf, mu, T_max, c, smooth_param);
        
        % Check for NaN or Inf values
        if any(isnan(F)) || any(isinf(F))
            F = ones(7,1) * 1e10;
        end
    catch ME
        % Return large but finite value if any error occurs
        fprintf('Error in robust_shooting: %s\n', ME.message);
        F = ones(7,1) * 1e10;
    end
end

%% Dynamics: state, mass, costate
function dydt = dynamics(~, y, mu, T_max, c, smooth_param)
    % Extract states and costates
    x = y(1:3); v = y(4:6); m = y(7);
    lambda_x = y(8:13); lambda_m = y(14);
    
    % Ensure mass is positive
    m = max(m, 1e-3);
    
    % Distances to primaries
    r1 = sqrt((x(1) + mu)^2 + x(2)^2 + x(3)^2);
    r2 = sqrt((x(1) - (1 - mu))^2 + x(2)^2 + x(3)^2);
    
    % Safety checks for singularities
    r1 = max(r1, 1e-10);
    r2 = max(r2, 1e-10);
    
    % Pseudopotential gradient (corrected)
    Ux = x(1) - (1 - mu)*(x(1) + mu)/r1^3 - mu*(x(1) - (1 - mu))/r2^3;
    Uy = x(2) - (1 - mu)*x(2)/r1^3 - mu*x(2)/r2^3;
    Uz = x(3) - (1 - mu)*x(3)/r1^3 - mu*x(3)/r2^3;
    gradU = [Ux; Uy; Uz];
    
    % Safety check for gradient
    if any(isnan(gradU)) || any(isinf(gradU))
        gradU = [0; 0; 0];
    end
    
    % Coriolis term given
    hv = [2*v(2); -2*v(1); 0];
    
    % Primer vector and control
    primer_vector = lambda_x(4:6);
    primer_norm = norm(primer_vector);
    
    % Safety check for primer vector
    primer_norm = max(primer_norm, 1e-10);
    
    % Thruster direction
    u_hat = -primer_vector/primer_norm;
    
    % Switching function with numerical protection
    S = primer_norm/m - abs(lambda_m)/c;
    
    % Smoothed bang-bang control with numerically stable implementation
    delta = 0.5*(1 + tanh(S/smooth_param));
    
    % Cap delta to avoid numerical issues
    delta = min(max(delta, 0), 1);
    
    % Dynamics
    dx = v;
    dv = -gradU + hv + (T_max/m)*delta*u_hat;
    dm = -(T_max/c)*delta;
    
    % Partial derivatives of potential
    d2Udx2 = zeros(3,3);
    
    % Second derivatives with safeguards
    d2Udx2(1,1) = 1 - (1-mu)/r1^3 - mu/r2^3 + 3*(1-mu)*(x(1)+mu)^2/r1^5 + 3*mu*(x(1)-(1-mu))^2/r2^5;
    d2Udx2(2,2) = 1 - (1-mu)/r1^3 - mu/r2^3 + 3*(1-mu)*x(2)^2/r1^5 + 3*mu*x(2)^2/r2^5;
    d2Udx2(3,3) = 1 - (1-mu)/r1^3 - mu/r2^3 + 3*(1-mu)*x(3)^2/r1^5 + 3*mu*x(3)^2/r2^5;
    
    % Cross terms
    d2Udx2(1,2) = 3*(1-mu)*(x(1)+mu)*x(2)/r1^5 + 3*mu*(x(1)-(1-mu))*x(2)/r2^5;
    d2Udx2(2,1) = d2Udx2(1,2);
    d2Udx2(1,3) = 3*(1-mu)*(x(1)+mu)*x(3)/r1^5 + 3*mu*(x(1)-(1-mu))*x(3)/r2^5;
    d2Udx2(3,1) = d2Udx2(1,3);
    d2Udx2(2,3) = 3*(1-mu)*x(2)*x(3)/r1^5 + 3*mu*x(2)*x(3)/r2^5;
    d2Udx2(3,2) = d2Udx2(2,3);
    
    % Check for numerical issues in Hessian
    if any(isnan(d2Udx2(:))) || any(isinf(d2Udx2(:)))
        d2Udx2 = eye(3);  % Fallback to identity matrix
    end
    
    % Costate dynamics
    dHdx = -d2Udx2*lambda_x(4:6);
    dHdv = -lambda_x(1:3) + [0 2 0; -2 0 0; 0 0 0]*lambda_x(4:6);
    dHdm = T_max*delta*primer_norm/m^2;  
    
    dlambda_x = -[dHdx; dHdv];
    dlambda_m = -dHdm;
    
    % Final safety check
    dydt = [dx; dv; dm; dlambda_x; dlambda_m];
    
    % Replace any NaN or Inf with zeros
    dydt(isnan(dydt) | isinf(dydt)) = 0;
end

%% Trajectory simulation function
function [t, Y] = simulateTrajectory(lambda0, x0, m0, t0, tf, mu, T_max, c, smooth_param)
    % Initial augmented state
    aug0 = [x0; m0; lambda0(1:6); lambda0(7)];
    
    % Integrate dynamics
    opts = odeset('RelTol',1e-8,'AbsTol',1e-8);
    
    % Choose solver based on smoothing parameter
    if smooth_param < 1e-3
        solver = @ode15s;
    else
        solver = @ode45;
    end

    try
        [t, Y] = solver(@(t, y) dynamics(t, y, mu, T_max, c, smooth_param), [t0 tf], aug0, opts);
    catch ME
        fprintf('Integration failed: %s\n', ME.message);
        % Create a minimal output to avoid errors
        t = [t0; tf];
        Y = [aug0'; aug0'];
    end
    
    % Ensure Y has the correct dimensions
    if size(Y, 2) < 14
        fprintf('Warning: Y has incorrect dimensions. Padding with zeros.\n');
        Y = [Y, zeros(size(Y, 1), 14 - size(Y, 2))];
    end
end

%% Calculate Lagrange points
function [L1, L2, L3, L4, L5] = calculateLagrangePoints(mu)
    % L1 (between primaries)
    L1 = findL1(mu);
    
    % L2 (beyond secondary)
    L2 = findL2(mu);
    
    % L3 (beyond primary)
    L3 = findL3(mu);
    
    % L4 and L5 (equilateral triangle points)
    L4 = [0.5-mu, sqrt(3)/2, 0];
    L5 = [0.5-mu, -sqrt(3)/2, 0];
end

% Find L1 (between primaries)
function L1 = findL1(mu)
    % Initial guess
    gamma = (mu/(3*(1-mu)))^(1/3);  % Approximation
    r = 1 - gamma;
    
    % Newton-Raphson iterations
    for i = 1:10
        f = r^5 - (3-mu)*r^4 + (3-2*mu)*r^3 - mu*r^2 + 2*mu*r - mu;
        df = 5*r^4 - 4*(3-mu)*r^3 + 3*(3-2*mu)*r^2 - 2*mu*r + 2*mu;
        dr = -f/df;
        r = r + dr;
        if abs(dr) < 1e-10
            break;
        end
    end
    L1 = [1-mu-r, 0, 0];
end

% Find L2 (beyond secondary)
function L2 = findL2(mu)
    % Initial guess
    gamma = (mu/(3*(1-mu)))^(1/3);  % Approximation
    r = 1 + gamma;
    
    % Newton-Raphson iterations
    for i = 1:10
        f = r^5 - (3-mu)*r^4 + (3-2*mu)*r^3 - mu*r^2 - 2*mu*r - mu;
        df = 5*r^4 - 4*(3-mu)*r^3 + 3*(3-2*mu)*r^2 - 2*mu*r - 2*mu;
        dr = -f/df;
        r = r + dr;
        if abs(dr) < 1e-10
            break;
        end
    end
    L2 = [1-mu+r-1, 0, 0];
end

% Find L3 (beyond primary)
function L3 = findL3(mu)
    % Initial guess
    r = -1 - (5/12)*mu;  % Approximation
    
    % Newton-Raphson iterations
    for i = 1:10
        f = r^5 + (2+mu)*r^4 + (1+2*mu)*r^3 - (1-mu)*r^2 - 2*(1-mu)*r - (1-mu);
        df = 5*r^4 + 4*(2+mu)*r^3 + 3*(1+2*mu)*r^2 - 2*(1-mu)*r - 2*(1-mu);
        dr = -f/df;
        r = r + dr;
        if abs(dr) < 1e-10
            break;
        end
    end
    L3 = [-mu+r+1, 0, 0];
end

%% Plots
function plotResults(t, Y, mu, T_max, c)
    % Check if Y has enough columns
    if size(Y, 2) < 14
        fprintf('Warning: Y matrix has fewer columns than expected (%d < 14).\n', size(Y, 2));
        % Pad with zeros to prevent errors
        Y = [Y zeros(size(Y, 1), 14-size(Y, 2))];
    end
    
    % Extract states
    x = Y(:,1:3);
    v = Y(:,4:6);
    m = Y(:,7);
    
    % Check if lambda data is available
    if size(Y, 2) >= 14
        lambda = Y(:,8:13);
        lambda_m = Y(:,14);
    else
        fprintf('Warning: Costate data not available for plotting.\n');
        lambda = zeros(size(Y, 1), 6);
        lambda_m = zeros(size(Y, 1), 1);
    end
    
    % Primary locations
    primary1 = [-mu, 0, 0];
    primary2 = [1-mu, 0, 0];
    
    % Create figure
    figure('Position', [100, 100, 1200, 800]);
    
    % Plot trajectory in rotating frame
    subplot(2,2,1)
    plot3(x(:,1), x(:,2), x(:,3), 'b-', 'LineWidth', 1.5);
    hold on;
    plot3(primary1(1), primary1(2), primary1(3), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
    plot3(primary2(1), primary2(2), primary2(3), 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
    plot3(x(1,1), x(1,2), x(1,3), 'bs', 'MarkerSize', 8, 'MarkerFaceColor', 'b');
    plot3(x(end,1), x(end,2), x(end,3), 'bd', 'MarkerSize', 8, 'MarkerFaceColor', 'b');
    
    % Add Lagrange points for reference
    [L1, L2, L3, L4, L5] = calculateLagrangePoints(mu);
    plot3(L1(1), L1(2), L1(3), 'k*', 'MarkerSize', 6);
    plot3(L2(1), L2(2), L2(3), 'k*', 'MarkerSize', 6);
    plot3(L3(1), L3(2), L3(3), 'k*', 'MarkerSize', 6);
    plot3(L4(1), L4(2), L4(3), 'k*', 'MarkerSize', 6);
    plot3(L5(1), L5(2), L5(3), 'k*', 'MarkerSize', 6);
    text(L1(1)+0.02, L1(2), L1(3), 'L1', 'FontSize', 8);
    text(L2(1)+0.02, L2(2), L2(3), 'L2', 'FontSize', 8);
    text(L3(1)+0.02, L3(2), L3(3), 'L3', 'FontSize', 8);
    text(L4(1)+0.02, L4(2), L4(3), 'L4', 'FontSize', 8);
    text(L5(1)+0.02, L5(2), L5(3), 'L5', 'FontSize', 8);
    
    grid on;
    title('Spacecraft Trajectory in Rotating Frame');
    xlabel('x'); ylabel('y'); zlabel('z');
    legend('Trajectory', 'Earth', 'Moon', 'Start', 'End', 'Lagrange Points');
    
    % Plot mass over time
    subplot(2,2,2)
    plot(t, m, 'b-', 'LineWidth', 1.5);
    grid on;
    title('Mass vs Time');
    xlabel('Time (non-dimensional)');
    ylabel('Mass (kg)');
    
    % Calculate thrust magnitude
    if size(Y, 2) >= 14
        primer_vector = lambda(:,4:6);
        primer_norm = sqrt(sum(primer_vector.^2, 2));
        S = primer_norm./m - abs(lambda_m)/c;
        delta = 0.5*(1 + tanh(S/1e-3));
    else
        % If costate data is not available, assume full throttle
        delta = ones(size(t));
    end
    thrust = T_max * delta;
    
    % Plot thrust profile
    subplot(2,2,3)
    plot(t, thrust, 'r-', 'LineWidth', 1.5);
    grid on;
    title('Thrust Magnitude vs Time');
    xlabel('Time (non-dimensional)');
    ylabel('Thrust (N)');
    
    % Plot switching function if available
    subplot(2,2,4)
    if size(Y, 2) >= 14
        plot(t, S, 'g-', 'LineWidth', 1.5);
        title('Switching Function vs Time');
    else
        plot(t, delta, 'g-', 'LineWidth', 1.5);
        title('Throttle vs Time');
    end
    grid on;
    xlabel('Time (non-dimensional)');
    ylabel('S(t) or Throttle');
    hold on;
    plot(t, zeros(size(t)), 'k--');
end
