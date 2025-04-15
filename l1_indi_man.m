function cr3bp_manifold_approximation()
    % Earth-Moon system parameters
    mu = 0.01215;  % Mass ratio for Earth-Moon system
    
    % Run indirect optimization to get initial trajectory
    [sol, x0, xf] = cr3bp_indirect_optimization();
    
    % Extract trajectory points
    traj_points = sol.y(1:6,:)';
    times = sol.t;
    
    % Apply Chebyshev approximation to the manifold
    [cheb_coeffs, approx_traj] = approximate_manifold_chebyshev(traj_points, times);
    
    % Plot results
    plot_results(sol, approx_traj, times, mu);
    
    % Compute error between original and approximated trajectory
    error = compute_approximation_error(traj_points, approx_traj);
    disp(['RMS Approximation Error: ', num2str(error)]);
end

function [sol, x0, xf] = cr3bp_indirect_optimization()
    % Earth-Moon system parameters
    mu = 0.01215;  % Mass ratio for Earth-Moon system
    
    % L1 point approximate coordinates
    L1_x = 0.8369;    % Approximate L1 x-coordinate for Earth-Moon
    L1_y = 0;
    L1_z = 0;
    L1_vx = 0;
    L1_vy = 0;
    L1_vz = 0;
    
    % GEO orbit parameters (in the rotating frame)
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
end

function [cheb_coeffs, approx_traj] = approximate_manifold_chebyshev(traj_points, times)
    % Normalize time to [-1, 1] interval for Chebyshev polynomials
    t_normalized = 2*(times - times(1))/(times(end) - times(1)) - 1;
    
    % Number of Chebyshev points to use
    M = 20;  % Can be adjusted for accuracy
    
    % Generate Chebyshev-Gauss-Lobatto (CGL) points
    cgl_points = -cos(pi*(0:M)/M);
    
    % For each state variable, compute Chebyshev approximation
    num_states = size(traj_points, 2);
    cheb_coeffs = zeros(M+1, num_states);
    approx_traj = zeros(size(traj_points));
    
    for state_idx = 1:num_states
        % Interpolate trajectory points to CGL points
        state_at_cgl = interp1(t_normalized, traj_points(:,state_idx), cgl_points, 'spline');
        
        % Compute Chebyshev approximation using least squares
        T = zeros(M+1, M+1);
        for i = 1:M+1
            for j = 1:M+1
                T(i,j) = chebyshev_polynomial(j-1, cgl_points(i));
            end
        end
        
        % Weight matrix for discrete orthogonality
        W = diag([1/2, ones(1,M-1), 1/2]);
        
        % Solve for Chebyshev coefficients
        cheb_coeffs(:,state_idx) = (T'*W*T)\(T'*W*state_at_cgl');
        
        % Evaluate approximation at original time points
        for i = 1:length(times)
            approx_value = 0;
            for j = 1:M+1
                approx_value = approx_value + cheb_coeffs(j,state_idx) * chebyshev_polynomial(j-1, t_normalized(i));
            end
            approx_traj(i,state_idx) = approx_value;
        end
    end
end

function T_n = chebyshev_polynomial(n, x)
    % Evaluate Chebyshev polynomial of the first kind T_n(x) using recurrence relation
    if n == 0
        T_n = 1;
    elseif n == 1
        T_n = x;
    else
        T_nm1 = 1;    % T_{n-1}
        T_n = x;      % T_n
        for i = 2:n
            T_np1 = 2*x*T_n - T_nm1;  % T_{n+1} = 2xT_n - T_{n-1}
            T_nm1 = T_n;
            T_n = T_np1;
        end
    end
end

function error = compute_approximation_error(original, approximated)
    % Compute RMS error between original and approximated trajectories
    squared_error = sum((original - approximated).^2, 'all');
    error = sqrt(squared_error / numel(original));
end

function plot_results(sol, approx_traj, times, mu)
    % Plot original and approximated trajectories
    figure('Position', [100, 100, 1000, 800]);
    
    % 3D Plot
    subplot(2, 2, [1, 3]);
    
    % Extract trajectory
    x_orig = sol.y(1, :);
    y_orig = sol.y(2, :);
    z_orig = sol.y(3, :);
    
    x_approx = approx_traj(:, 1)';
    y_approx = approx_traj(:, 2)';
    z_approx = approx_traj(:, 3)';
    
    % Plot trajectories
    plot3(x_orig, y_orig, z_orig, 'b-', 'LineWidth', 2);
    hold on;
    plot3(x_approx, y_approx, z_approx, 'r--', 'LineWidth', 2);
    
    % Plot primaries
    scatter3(-mu, 0, 0, 100, 'blue', 'filled');
    scatter3(1-mu, 0, 0, 50, [0.5 0.5 0.5], 'filled');
    
    % Plot libration points (approximate location)
    L1_x = 0.8369;  % Approximate L1 x-coordinate
    scatter3(L1_x, 0, 0, 50, 'red', 'x', 'LineWidth', 2);
    
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    title('CR3BP Trajectory: GEO to L1');
    legend('Original Trajectory', 'Chebyshev Approximation', 'Earth', 'Moon', 'L1');
    grid on;
    
    % Set equal aspect ratio
    axis equal;
    
    % Adjust view
    view(45, 45);
    
    % XY Plane Projection
    subplot(2, 2, 2);
    plot(x_orig, y_orig, 'b-', 'LineWidth', 2);
    hold on;
    plot(x_approx, y_approx, 'r--', 'LineWidth', 2);
    scatter(-mu, 0, 100, 'blue', 'filled');
    scatter(1-mu, 0, 50, [0.5 0.5 0.5], 'filled');
    scatter(L1_x, 0, 50, 'red', 'x', 'LineWidth', 2);
    xlabel('X');
    ylabel('Y');
    title('XY Plane Projection');
    grid on;
    axis equal;
    
    % Error Plot
    subplot(2, 2, 4);
    error_x = x_orig - x_approx;
    error_y = y_orig - y_approx;
    error_z = z_orig - z_approx;
    total_error = sqrt(error_x.^2 + error_y.^2 + error_z.^2);
    
    plot(times, total_error, 'k-', 'LineWidth', 2);
    xlabel('Time');
    ylabel('Error Magnitude');
    title('Approximation Error');
    grid on;
end

function [sol, error, final_state] = shoot(lam_init, x0, xf, tf, mu)
    % Shooting method for two-point boundary value problem
    
    % Initial state augmented with costates
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
    
    % Costate derivatives (adjoint equations)
    dlam1_dt = -lam(4);
    dlam2_dt = -lam(5);
    dlam3_dt = -lam(6);
    dlam4_dt = -lam(1) + 2*lam(5) - lam(4)*dUdx - lam(5)*dUdxy - lam(6)*dUdxz;
    dlam5_dt = -lam(2) - 2*lam(4) - lam(4)*dUdxy - lam(5)*dUdy - lam(6)*dUdyz;
    dlam6_dt = -lam(3) - lam(4)*dUdxz - lam(5)*dUdyz - lam(6)*dUdz;
    
    % Return the derivative of the state vector
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

function perform_bivariate_approximation(sol, mu)
    % Example of bivariate approximation using Kronecker product
    
    % Extract trajectory
    x = sol.y(1, :);
    y = sol.y(2, :);
    times = sol.t;
    
    % Normalize time to [-1, 1]
    t_norm = 2*(times - times(1))/(times(end) - times(1)) - 1;
    
    % Create 2D grid points for u,v coordinates on manifold
    [U, V] = meshgrid(linspace(-1, 1, 10), linspace(-1, 1, 10));
    
    % Generate synthetic manifold data (for demonstration)
    G = zeros(size(U));
    for i = 1:size(U,1)
        for j = 1:size(U,2)
            u = U(i,j);
            v = V(i,j);
            % Example function mapping (u,v) to manifold
            G(i,j) = sin(pi*u) * cos(pi*v);
        end
    end
    
    % Number of Chebyshev points to use in each dimension
    M1 = 5;
    M2 = 5;
    
    % Generate Chebyshev-Gauss-Lobatto (CGL) points
    xi1 = -cos(pi*(0:M1)/M1);
    xi2 = -cos(pi*(0:M2)/M2);
    
    % Create matrices T1 and T2
    T1 = zeros(M1+1, M1+1);
    T2 = zeros(M2+1, M2+1);
    
    for i = 1:M1+1
        for j = 1:M1+1
            T1(i,j) = chebyshev_polynomial(j-1, xi1(i));
        end
    end
    
    for i = 1:M2+1
        for j = 1:M2+1
            T2(i,j) = chebyshev_polynomial(j-1, xi2(i));
        end
    end
    
    % Weight matrices
    W1 = diag([1/2, ones(1,M1-1), 1/2]);
    W2 = diag([1/2, ones(1,M2-1), 1/2]);
    
    % Construct g vector (sample function values at CGL nodes)
    g = zeros((M1+1)*(M2+1), 1);
    idx = 1;
    for j = 1:M2+1
        for i = 1:M1+1
            % Interpolate function value at CGL node
            u_val = xi1(i);
            v_val = xi2(j);
            g(idx) = interp2(U, V, G, u_val, v_val, 'spline');
            idx = idx + 1;
        end
    end
    
    % Compute Chebyshev coefficients using Kronecker product formula
    C1 = (T1'*W1*T1)\(T1'*W1);
    C2 = (T2'*W2*T2)\(T2'*W2);
    a_hat = kron(C1, C2) * g;
    
    % Reshape coefficients into matrix form
    a_matrix = reshape(a_hat, M1+1, M2+1);
    
    % Display some coefficients
    disp('Bivariate Chebyshev approximation coefficients:');
    disp(a_matrix(1:min(5,M1+1), 1:min(5,M2+1)));
end