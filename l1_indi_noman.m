function cr3bp_indirect_optimization()
    % Earth-Moon system parameters
    mu = 0.01215;  % Mass ratio for Earth-Moon system
    % For Earth-Moon: 
    % mu = mass_moon / (mass_earth + mass_moon) ≈ 0.01215
    
    % Earth-Moon system
    % L1 point approximate coordinates
    L1_x = 0.8369;    % Approximate L1 x-coordinate for Earth-Moon
    L1_y = 0;
    L1_z = 0;
    L1_vx = 0;
    L1_vy = 0;
    L1_vz = 0;
    
    % GEO orbit parameters (in the rotating frame)
    % GEO is approximately at 42164 km from Earth's center
    % Earth-Moon distance is 384,400 km
    % So GEO is at about 0.11 in normalized units from Earth
    geo_radius = 0.11;
    geo_x = -mu + geo_radius;  % GEO around Earth
    geo_y = 0;
    geo_z = 0;
    
    % Initial velocity (circular orbit around Earth in rotating frame)
    geo_vx = 0;
    geo_vy = geo_radius * (1 - sqrt((1-mu) / geo_radius^3));  % Corrected for rotating frame
    geo_vz = 0;
    
    % Initial and target states
    x0 = [geo_x, geo_y, geo_z, geo_vx, geo_vy, geo_vz];
    xf = [L1_x, L1_y, L1_z, L1_vx, L1_vy, L1_vz];
    
    % Time of flight (normalized units)
    tf = 3.0;  % Adjust as needed
    
    % Initial costate guess
    lam_init_guess = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1];
    
    % Optimize
    disp('Starting optimization...');
    options = optimoptions('fminunc', 'Display', 'iter', 'MaxIterations', 100);
    [lam_opt, fval] = fminunc(@(lam) objective(lam, x0, xf, tf, mu), lam_init_guess, options);
    
    disp('Optimization complete.');
    disp(['Final costates: ', num2str(lam_opt)]);
    
    % Get final trajectory
    [sol, final_error, final_state] = shoot(lam_opt, x0, xf, tf, mu);
    
    % Fix: Convert final_error to a string representation for display
    error_str = sprintf('[%s]', sprintf(' %.6f', final_error));
    disp(['Terminal state error: ', error_str]);
    
    disp(['Initial Jacobi constant: ', num2str(compute_jacobi_constant(x0, mu))]);
    disp(['Final Jacobi constant: ', num2str(compute_jacobi_constant(final_state(1:6), mu))]);
    
    % Plot trajectory
    plot_trajectory(sol, mu, ['CR3BP Trajectory: GEO to L1 (Error: ', num2str(norm(final_error)), ')']);
    
    % Plot Hamiltonian (should be constant)
    plot_hamiltonian(sol, mu);
end

function [sol, error, final_state] = shoot(lam_init, x0, xf, tf, mu)
    % Shooting method for two-point BVP
    % Initial state with costates
    init_state = [x0, lam_init];
    
    % Integrate system
    opts = odeset('RelTol', 1e-8, 'AbsTol', 1e-8);
    [t, y] = ode45(@(t, y) cr3bp_dynamics(t, y, mu), [0, tf], init_state, opts);
    
    sol.t = t;
    sol.y = y';
    
    % Terminal state
    final_state = y(end, :)';
    
    % error in terminal constraints
    error = final_state(1:6) - xf';
end

function obj = objective(lam_init, x0, xf, tf, mu)
    % Objective function for the optimizer: sum of squared errors in terminal state
    [~, error, ~] = shoot(lam_init, x0, xf, tf, mu);
    obj = sum(error.^2);
end

function dydt = cr3bp_dynamics(t, state, mu)
    % CR3BP dynamics in the rotating frame
    % state = [x, y, z, vx, vy, vz, λ1, λ2, λ3, λ4, λ5, λ6]
    
    x = state(1);
    y = state(2);
    z = state(3);
    vx = state(4);
    vy = state(5);
    vz = state(6);
    lam = state(7:12);
    
    % Distances to primaries
    r1 = sqrt((x + mu)^2 + y^2 + z^2);      % Distance to larger primary (Earth)
    r2 = sqrt((x - 1 + mu)^2 + y^2 + z^2);  % Distance to smaller primary (Moon)
    
    % Position derivatives
    dx_dt = vx;
    dy_dt = vy;
    dz_dt = vz;
    
    % Velocity derivatives (CR3BP equations of motion)
    dvx_dt = 2*vy + x - (1-mu)*(x+mu)/r1^3 - mu*(x-1+mu)/r2^3;
    dvy_dt = -2*vx + y - (1-mu)*y/r1^3 - mu*y/r2^3;
    dvz_dt = -(1-mu)*z/r1^3 - mu*z/r2^3;
    
    % partial derivatives for costate equations
    % ∂U/∂x
    dUdx = 1 - (1-mu)/r1^3 - mu/r2^3 + 3*(1-mu)*(x+mu)^2/r1^5 + 3*mu*(x-1+mu)^2/r2^5;
    % ∂U/∂y
    dUdy = 1 - (1-mu)/r1^3 - mu/r2^3 + 3*(1-mu)*y^2/r1^5 + 3*mu*y^2/r2^5;
    % ∂U/∂z
    dUdz = -(1-mu)/r1^3 - mu/r2^3 + 3*(1-mu)*z^2/r1^5 + 3*mu*z^2/r2^5;
    % Mixed partials
    dUdxy = 3*(1-mu)*(x+mu)*y/r1^5 + 3*mu*(x-1+mu)*y/r2^5;
    dUdxz = 3*(1-mu)*(x+mu)*z/r1^5 + 3*mu*(x-1+mu)*z/r2^5;
    dUdyz = 3*(1-mu)*y*z/r1^5 + 3*mu*y*z/r2^5;
    
    % Costate derivatives 
    dlam1_dt = -lam(4);
    dlam2_dt = -lam(5);
    dlam3_dt = -lam(6);
    dlam4_dt = -lam(1) + 2*lam(5) - lam(4)*dUdx - lam(5)*dUdxy - lam(6)*dUdxz;
    dlam5_dt = -lam(2) - 2*lam(4) - lam(4)*dUdxy - lam(5)*dUdy - lam(6)*dUdyz;
    dlam6_dt = -lam(3) - lam(4)*dUdxz - lam(5)*dUdyz - lam(6)*dUdz;
    
    % Derivatives of the state vector
    dydt = [dx_dt; dy_dt; dz_dt; dvx_dt; dvy_dt; dvz_dt;
            dlam1_dt; dlam2_dt; dlam3_dt; dlam4_dt; dlam5_dt; dlam6_dt];
end

function C = compute_jacobi_constant(state, mu)
    % Compute the Jacobi constant (energy integral) for a given state
    % C = 2U - v²
    
    x = state(1);
    y = state(2);
    z = state(3);
    vx = state(4);
    vy = state(5);
    vz = state(6);
    
    r1 = sqrt((x + mu)^2 + y^2 + z^2);
    r2 = sqrt((x - 1 + mu)^2 + y^2 + z^2);
    
    % Potential energy
    U = (x^2 + y^2)/2 + (1-mu)/r1 + mu/r2;
    
    % Kinetic energy
    v_squared = vx^2 + vy^2 + vz^2;
    
    % Jacobi constant
    C = 2*U - v_squared;
end

function plot_trajectory(sol, mu, title_str)
    % Create 3D plot of the trajectory
    
    figure('Position', [100, 100, 800, 600]);
    ax = axes;
    
    % Extract trajectory
    x = sol.y(1, :);
    y = sol.y(2, :);
    z = sol.y(3, :);
    
    % Plot trajectory
    plot3(ax, x, y, z, 'b-', 'LineWidth', 2);
    hold on;
    
    % Plot primaries
    scatter3(ax, -mu, 0, 0, 100, 'blue', 'filled');
    scatter3(ax, 1-mu, 0, 0, 50, [0.5 0.5 0.5], 'filled');
    
    % Plot libration points (approximate locations)
    L1_x = 1 - mu - (mu/3)^(1/3);  % Approximate L1 x-coordinate
    scatter3(ax, L1_x, 0, 0, 50, 'red', 'x', 'LineWidth', 2);
    
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    title(title_str);
    legend('Trajectory', 'Earth', 'Moon', 'L1');
    grid on;
    
    % Set equal aspect ratio
    axis equal;
    
    % Adjust view
    view(45, 45);
    hold off;
end

function plot_hamiltonian(sol, mu)
    % Plot Hamiltonian (should be constant)
    t = sol.t;
    states = sol.y;
    H = zeros(size(t));
    
    for i = 1:length(t)
        x = states(1:6, i);
        lam = states(7:12, i);
        
        % Hamiltonian
        r1 = sqrt((x(1) + mu)^2 + x(2)^2 + x(3)^2);
        r2 = sqrt((x(1) - 1 + mu)^2 + x(2)^2 + x(3)^2);
        U = (x(1)^2 + x(2)^2)/2 + (1-mu)/r1 + mu/r2;
        
        % Hamiltonian = λ·f(x)
        H(i) = lam(1)*x(4) + lam(2)*x(5) + lam(3)*x(6) + ...
               lam(4)*(2*x(5) + x(1) - (1-mu)*(x(1)+mu)/r1^3 - mu*(x(1)-1+mu)/r2^3) + ...
               lam(5)*(-2*x(4) + x(2) - (1-mu)*x(2)/r1^3 - mu*x(2)/r2^3) + ...
               lam(6)*(-(1-mu)*x(3)/r1^3 - mu*x(3)/r2^3);
    end
    
    figure('Position', [100, 100, 800, 400]);
    plot(t, H, 'LineWidth', 2);
    title('Hamiltonian Along Trajectory');
    xlabel('Time');
    ylabel('Hamiltonian');
    grid on;
end

%trying to push to github