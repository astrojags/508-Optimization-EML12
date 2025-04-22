function traj_l1_indirect_nomanifold()
    % CR3BP trajectory optimization problem using indirect method with the manifold targeting
    
    % Problem parameters
    mu = 0; % 0.012155; % Mass ratio of Moon to Earth+Moon system
    T_max = 0.1;   % Maximum thrust [N]
    Isp = 3000; % Typical Specific Impulse [Isp]
    g0 = 9.80665; % Gravity [m/s^2]
    c = Isp * g0;  % Characteristic exhaust velocity (Isp*g0) [m/s]
    m_0 = 1500;    % Initial mass [kg]

    %---------------------------------------------------------------------
    
    % Initial and final times [normalized units]
    t0 = 0;
    tf = 20; % Will be adjusted if time is free
    
    % GEO initial conditions in rotating frame (normalized)
    a_GEO = 52164/384400; % 42164/384400; % GEO radius normalized by Earth-Moon distance
    x0 = [(1-mu-a_GEO), 0, 0, 0, a_GEO, 0]'; % [x, y, z, vx, vy, vz]
    
    % L1 location (approximate)
    gamma = (mu/3/(1-mu))^(1/3);
    L1_x = 1 - mu - gamma;
    
    % Defining rho
    rho = 1e-2; % Smoothing parameter for switching function 
    rhoMin = 1e-4;
    
    % Final conditions (targeting L1 point with zero velocity)
    xf_target = [L1_x, 0, 0, 0, 0, 0]';
    
    % Initial costate guess
    lambda0_guess = [-0.1; -0.1; -0.1; -1.0; -1.0; -1.0; 0.1]; % Position, velocity, mass costates
    
    % Continuation parameters
    T_max_values = linspace(T_max*2, T_max, 5); % Start with higher thrust
    
    % Solve TPBVP using continuation
    solution_found = false;
    for i = 1:length(T_max_values)
        curr_T_max = T_max_values(i);
        
        fprintf('Attempting solution with T_max = %.4f\n', curr_T_max);
        
        try
            % Define options for fsolve
            options = optimoptions('fsolve', 'Display', 'iter', 'MaxFunctionEvaluations', 5000, ...
                'MaxIterations', 300, 'FunctionTolerance', 1e-6);
            
            % Solve the TPBVP
            [lambda0_sol, fval, exitflag] = fsolve(@(lambda0) shooting_function(lambda0, x0, m_0, ...
                xf_target, t0, tf, mu, curr_T_max, c, rho), lambda0_guess, options);
            
            if exitflag > 0 && norm(fval) < 1e-4
                solution_found = true;
                fprintf('Solution found with T_max = %.4f\n', curr_T_max);
                
                % Update guess for next iteration if needed
                lambda0_guess = lambda0_sol;
                break;
            else
                % Update guess for next iteration
                lambda0_guess = lambda0_sol;
            end
        catch e
            fprintf('Error in solution attempt: %s\n', e.message);
            % Continue to next iteration
        end
    end
    
    if solution_found
        % Compute final trajectory with solution
        [t, x, u, delta, switching] = integrate_trajectory(x0, m_0, lambda0_sol, t0, tf, mu, T_max, c, rho);
        
        % Plot results
        plot_results(t, x, u, delta, switching, xf_target, mu);
        
        % Calculate fuel consumption
        fuel_consumed = m_0 - x(end,7);
        fprintf('Fuel consumed: %.2f kg (%.2f%%)\n', fuel_consumed, 100*fuel_consumed/m_0);
    else
        fprintf('No solution found after all continuation attempts.\n');
    end

    rho = 0.9 * rho;
    if rho < rhoMin
        return;
    end
end

function error = shooting_function(lambda0, x0, m0, xf_target, t0, tf, mu, T_max, c, rho)
    % Shooting function for solving the TPBVP
    
    % Initial state with mass
    X0 = [x0; m0];
    
    % Initial costate vector
    lambda_x0 = lambda0(1:6);
    lambda_m0 = lambda0(7);
    
    % Combine state and costate for integration
    Z0 = [X0; lambda_x0; lambda_m0];
    
    % Integrate the trajectory with dynamics and costates
    [~, Z] = ode45(@(t, z) cr3bp_state_costate_dynamics(t, z, mu, T_max, c, rho), [t0, tf], Z0);
    
    % Extract final state and costate
    Xf = Z(end, 1:7)';
    lambda_xf = Z(end, 8:13)';
    lambda_mf = Z(end, 14);
    
    % Compute final state error relative to target
    error_state = Xf(1:6) - xf_target;
    
    % Compute Hamiltonian at final time (for free time problems)
    % H_f = compute_hamiltonian(Xf, lambda_xf, lambda_mf, mu, T_max, c);
    
    % Define boundary condition errors
    % For fixed final time:
    error = error_state;
    
    % For free final time: Uncomment the line below and add H_f to the error
    % error = [error_state; H_f];
end

function dZdt = cr3bp_state_costate_dynamics(t, Z, mu, T_max, c, rho)
    % Combined state and costate dynamics based on Equations 20, 21, 26
    
    % Extract state and costate
    r = Z(1:3);        % Position
    v = Z(4:6);        % Velocity
    m = Z(7);          % Mass
    lambda_r = Z(8:10);  % Position costate
    lambda_v = Z(11:13); % Velocity costate
    lambda_m = Z(14);    % Mass costate
    
    % State dynamics components
    [dUdr, h_v] = cr3bp_forces(r, v, mu);
    
    % Thrust control law (Eq. 27 and 28)
    % Primer vector
    lambda_v_norm = norm(lambda_v);
    
    % Thrust direction (along negative primer vector)
    if lambda_v_norm > 0
        u_hat = -lambda_v / lambda_v_norm;
    else
        u_hat = [0; 0; 0];
    end
    
    % Switching function (Eq. 28)
    S = lambda_v_norm - (1 - lambda_m) * m / c;
    
    % Throttle (bang-bang control)
    delta = 0.5 * (1 + tanh(S / rho));
    
    % Acceleration due to thrust (Eq. 20)
    a_thrust = (T_max / m) * delta * u_hat;
    
    % State derivatives
    dr_dt = v;
    dv_dt = -dUdr + h_v + a_thrust;
    dm_dt = -T_max * delta / c;  % Mass flow rate (Eq. 21)
    
    % Costate dynamics (Eq. 26)
    % Compute necessary Jacobians
    [d2Udr2, dh_v_dr, dh_v_dv] = cr3bp_jacobians(r, v, mu);
    
    % Costate derivatives from Eq. 26
    dlambda_r_dt = -lambda_v' * (-d2Udr2 - dh_v_dr);
    dlambda_v_dt = -lambda_r' - lambda_v' * (-dh_v_dv);
    dlambda_m_dt = -(lambda_v' * (-T_max * delta * u_hat / m^2));
    
    % Combine all derivatives
    dZdt = [dr_dt; dv_dt; dm_dt; dlambda_r_dt'; dlambda_v_dt'; dlambda_m_dt];
end

function [dUdr, h_v] = cr3bp_forces(r, v, mu)
    % Compute the forces in the CR3BP
    
    % Extract position components
    x = r(1);
    y = r(2);
    z = r(3);
    
    % Distances to primaries
    d1 = sqrt((x + mu)^2 + y^2 + z^2);    % Distance to m1 (Earth)
    d2 = sqrt((x - (1-mu))^2 + y^2 + z^2); % Distance to m2 (Moon)
    
    % Gradient of the pseudo-potential
    dUdr = zeros(3,1);
    
    % x component
    dUdr(1) = x - ((1-mu)*(x+mu))/(d1^3) - (mu*(x-(1-mu)))/(d2^3);
    
    % y component
    dUdr(2) = y - ((1-mu)*y)/(d1^3) - (mu*y)/(d2^3);
    
    % z component
    dUdr(3) = -((1-mu)*z)/(d1^3) - (mu*z)/(d2^3);
    
    % Coriolis and centrifugal terms (h(v) from Eq. 20)
    h_v = [2*v(2); -2*v(1); 0];
end

function [d2Udr2, dh_v_dr, dh_v_dv] = cr3bp_jacobians(r, v, mu)
    % Compute the Jacobians needed for costate dynamics
    
    % Extract position components
    x = r(1);
    y = r(2);
    z = r(3);
    
    % Distances to primaries
    d1 = sqrt((x + mu)^2 + y^2 + z^2);    % Distance to m1 (Earth)
    d2 = sqrt((x - (1-mu))^2 + y^2 + z^2); % Distance to m2 (Moon)
    
    % Second derivatives of the pseudo-potential (Hessian)
    d2Udr2 = zeros(3,3);
    
    % Common terms
    d1_5 = d1^5;
    d2_5 = d2^5;
    
    % Fill Hessian matrix
    % xx component
    d2Udr2(1,1) = 1 - (1-mu)/d1^3 - mu/d2^3 + 3*(1-mu)*(x+mu)^2/d1_5 + 3*mu*(x-(1-mu))^2/d2_5;
    
    % xy component
    d2Udr2(1,2) = 3*(1-mu)*(x+mu)*y/d1_5 + 3*mu*(x-(1-mu))*y/d2_5;
    d2Udr2(2,1) = d2Udr2(1,2);
    
    % xz component
    d2Udr2(1,3) = 3*(1-mu)*(x+mu)*z/d1_5 + 3*mu*(x-(1-mu))*z/d2_5;
    d2Udr2(3,1) = d2Udr2(1,3);
    
    % yy component
    d2Udr2(2,2) = 1 - (1-mu)/d1^3 - mu/d2^3 + 3*(1-mu)*y^2/d1_5 + 3*mu*y^2/d2_5;
    
    % yz component
    d2Udr2(2,3) = 3*(1-mu)*y*z/d1_5 + 3*mu*y*z/d2_5;
    d2Udr2(3,2) = d2Udr2(2,3);
    
    % zz component
    d2Udr2(3,3) = -(1-mu)/d1^3 - mu/d2^3 + 3*(1-mu)*z^2/d1_5 + 3*mu*z^2/d2_5;
    
    % Jacobian of h(v) with respect to r (zeros as h doesn't depend on r)
    dh_v_dr = zeros(3,3);
    
    % Jacobian of h(v) with respect to v
    dh_v_dv = zeros(3,3);
    dh_v_dv(1,2) = 2;  % derivative of 2*vy with respect to vy
    dh_v_dv(2,1) = -2; % derivative of -2*vx with respect to vx
end

function H = compute_hamiltonian(X, lambda_x, lambda_m, mu, T_max, c)
    % Compute the Hamiltonian value according to Eq. 25
    
    r = X(1:3);
    v = X(4:6);
    m = X(7);
    lambda_r = lambda_x(1:3);
    lambda_v = lambda_x(4:6);
    
    % Forces
    [dUdr, h_v] = cr3bp_forces(r, v, mu);
    
    % Primer vector
    lambda_v_norm = norm(lambda_v);
    
    % Thrust direction
    if lambda_v_norm > 0
        u_hat = -lambda_v / lambda_v_norm;
    else
        u_hat = [0; 0; 0];
    end
    
    % Switching function
    S = lambda_v_norm - (1 - lambda_m) * m / c;
    
    % Throttle
    delta = 0.5 * (1 + tanh(S / rho));
    
    % Hamiltonian from Eq. 25
    H = lambda_r' * v + lambda_v' * (-dUdr + h_v) + ...
        (T_max / c) * (1 - lambda_m) * delta + (T_max / m) * lambda_v' * u_hat * delta;
end

function [t, X, u, delta, switching] = integrate_trajectory(x0, m0, lambda0, t0, tf, mu, T_max, c, rho)
    % Integrate the full trajectory with control
    
    % Initial state with mass
    X0 = [x0; m0];
    
    % Initial costate vector
    lambda_x0 = lambda0(1:6);
    lambda_m0 = lambda0(7);
    
    % Combine state and costate for integration
    Z0 = [X0; lambda_x0; lambda_m0];
    
    % Define ODE options with event detection for switching
    options = odeset('RelTol', 1e-10, 'AbsTol', 1e-10, 'Events', @(t,z) switching_event(t, z, mu, T_max, c));
    
    % Initialize output arrays
    t = [];
    Z = [];
    u = [];
    delta = [];
    switching = [];
    
    % Integrate trajectory in segments (due to switching)
    tspan = [t0, tf];
    while tspan(1) < tf
        [t_seg, Z_seg, te, ze, ie] = ode45(@(t, z) cr3bp_state_costate_dynamics(t, z, mu, T_max, c, rho), tspan, Z0, options);
        
        % Extract control at each time point
        u_seg = zeros(length(t_seg), 3);
        delta_seg = zeros(length(t_seg), 1);
        for i = 1:length(t_seg)
            lambda_v = Z_seg(i, 11:13)';
            lambda_m = Z_seg(i, 14);
            m = Z_seg(i, 7);
            
            lambda_v_norm = norm(lambda_v);
            
            % Thrust direction
            if lambda_v_norm > 0
                u_hat = -lambda_v / lambda_v_norm;
            else
                u_hat = [0; 0; 0];
            end
            u_seg(i,:) = u_hat';
            
            % Switching function and throttle
            S = lambda_v_norm - (1 - lambda_m) * m / c;
            delta_seg(i) = double(S > 0);
        end
        
        % Append segment results
        t = [t; t_seg];
        Z = [Z; Z_seg];
        u = [u; u_seg];
        delta = [delta; delta_seg];
        
        % Check if we've reached the end or have a switching event
        if isempty(te) || te >= tf
            break;
        else
            % Record switching point
            switching = [switching; te];
            
            % Update initial conditions for next segment
            Z0 = ze;
            tspan = [te, tf];
        end
    end
    
    % Extract state variables
    X = Z(:, 1:7);
end

function [value, isterminal, direction] = switching_event(t, z, mu, T_max, c, rho)
    % Event function to detect control switching based on Eq. 28
    
    lambda_v = z(11:13);
    lambda_m = z(14);
    m = z(7);
    
    lambda_v_norm = norm(lambda_v);
    S = lambda_v_norm - (1 - lambda_m) * m / c;
    
    % Event occurs when switching function crosses zero
    value = tanh(S / rho); % smoother zero crossing
    isterminal = 0;  % Don't terminate
    direction = 0;   % Detect both rising and falling crossings
end

function plot_results(t, X, u, delta, switching, xf_target, mu)
    % Plot trajectory and control results
    
    % Figure for trajectory
    figure(1);
    clf;
    
    % 3D trajectory plot
    subplot(2,2,1);
    plot3(X(:,1), X(:,2), X(:,3), 'b-', 'LineWidth', 1.5);
    hold on;
    plot3(X(1,1), X(1,2), X(1,3), 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
    plot3(X(end,1), X(end,2), X(end,3), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
    plot3(xf_target(1), xf_target(2), xf_target(3), 'mo', 'MarkerSize', 8, 'MarkerFaceColor', 'm');
    plot3(-mu, 0, 0, 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'k'); % Earth
    plot3(1-mu, 0, 0, 'ko', 'MarkerSize', 6, 'MarkerFaceColor', 'k'); % Moon
    grid on;
    xlabel('x');
    ylabel('y');
    zlabel('z');
    title('Trajectory in Rotating Frame');
    
    % XY projection
    subplot(2,2,2);
    plot(X(:,1), X(:,2), 'b-', 'LineWidth', 1.5);
    hold on;
    plot(X(1,1), X(1,2), 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
    plot(X(end,1), X(end,2), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
    plot(xf_target(1), xf_target(2), 'mo', 'MarkerSize', 8, 'MarkerFaceColor', 'm');
    plot(-mu, 0, 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'k'); % Earth
    plot(1-mu, 0, 'ko', 'MarkerSize', 6, 'MarkerFaceColor', 'k'); % Moon
    grid on;
    xlabel('x');
    ylabel('y');
    title('XY Projection');
    axis equal;
    
    % Control profile (throttle)
    subplot(2,2,3);
    plot(t, delta, 'k-', 'LineWidth', 1.5);
    hold on;
    for i = 1:length(switching)
        plot([switching(i) switching(i)], [0 1], 'r--');
    end
    grid on;
    xlabel('Time');
    ylabel('Throttle (\delta)');
    title('Throttle Profile');
    ylim([-0.1, 1.1]);
    
    % Mass profile
    subplot(2,2,4);
    plot(t, X(:,7), 'b-', 'LineWidth', 1.5);
    grid on;
    xlabel('Time');
    ylabel('Mass (kg)');
    title('Mass Profile');
    
    % Figure for states and controls
    figure(2);
    clf;
    
    % Position
    subplot(3,2,1);
    plot(t, X(:,1), 'r-', t, X(:,2), 'g-', t, X(:,3), 'b-', 'LineWidth', 1.5);
    grid on;
    xlabel('Time');
    ylabel('Position');
    title('Position Components');
    legend('x', 'y', 'z');
    
    % Velocity
    subplot(3,2,2);
    plot(t, X(:,4), 'r-', t, X(:,5), 'g-', t, X(:,6), 'b-', 'LineWidth', 1.5);
    grid on;
    xlabel('Time');
    ylabel('Velocity');
    title('Velocity Components');
    legend('vx', 'vy', 'vz');
    
    % Control direction
    subplot(3,2,3);
    plot(t, u(:,1), 'r-', t, u(:,2), 'g-', t, u(:,3), 'b-', 'LineWidth', 1.5);
    grid on;
    xlabel('Time');
    ylabel('Control Direction');
    title('Thrust Direction Components');
    legend('u_x', 'u_y', 'u_z');
    
    % Thrust magnitude
    subplot(3,2,4);
    on_indices = delta > 0.5;
    thrust_mag = zeros(size(t));
    thrust_mag(on_indices) = 1;
    plot(t, thrust_mag, 'k-', 'LineWidth', 1.5);
    grid on;
    xlabel('Time');
    ylabel('Thrust Status');
    title('Thrust On/Off Status');
    ylim([-0.1, 1.1]);
    
    % Hamiltonian
    subplot(3,2,5);
    H = zeros(size(t));
    for i = 1:length(t)
        H(i) = compute_hamiltonian(X(i,:)', X(i,8:13)', X(i,14), mu, 0.1, 3000*9.80665);
    end
    plot(t, H, 'k-', 'LineWidth', 1.5);
    grid on;
    xlabel('Time');
    ylabel('Hamiltonian');
    title('Hamiltonian Value');
    
    % Performance index
    subplot(3,2,6);
    J = cumtrapz(t, delta);
    plot(t, J, 'b-', 'LineWidth', 1.5);
    grid on;
    xlabel('Time');
    ylabel('J');
    title('Performance Index (Fuel Consumption)');
end


% dfjbhviuofwejdheufhrofewjj