%% DESCRIPTION
% 0. sample fixed points along the periodic orbit
% 1. Calculate eigenvectors and eigenvalues of monodromy matrix at these
% points
% 2. Plot trajectories along these emanating from the periodic orbit
clear all; close all; clc;
%% The Periodic Orbit
mu = 7.3476E22/(5.9724E24+7.3476E22);
s0 = [1.18090567981666	0	0.00219963894219877	0	-0.155896034718107	0]';
tf = 1.70775749693768;
%s0 = [0.841800000000000	0	0	0	-0.0396031330131240	0]';
%tf = 1.34805838258873;

%% Sample Fixed Points
times = linspace(0,2*tf,50);
intoptions = odeset('RelTol',3e-14,'AbsTol',3e-15);
[t_list,s_list] = ode89(@cr3bp_dyn,times,s0,intoptions);
figure(1)
scatter(s_list(:,1),s_list(:,2))
axis equal

figure(2)
axis equal
scatter(1-mu,0,20,"black",'filled')
hold on
scatter(-mu,0,20,"black",'filled')
%% multiple loop version 
for i = 1:length(s_list)
Mon = getMonodromyMat(s_list(i,:)',tf,mu);
[V,D] = eigs(Mon);
unstable_vec = V(:,1)
stable_vec = V(:,6);
alpha = 1e-13; %1e-11
tspan = [0 12*tf]; %12
[t_ulist,s_ulist] = ode89(@cr3bp_dyn,tspan,s_list(i,:)'+alpha*unstable_vec,intoptions);
[t_slist,s_slist] = ode89(@cr3bp_dyn,-1*tspan,s_list(i,:)'+alpha*stable_vec,intoptions);
plot3(s_ulist(:,1),s_ulist(:,2),s_ulist(:,3),color=[0,0,1])
plot3(s_slist(:,1),s_slist(:,2),s_slist(:,3),color=[1,0,0])
[t_ulist,s_ulist] = ode89(@cr3bp_dyn,tspan,s_list(i,:)'-alpha*unstable_vec,intoptions);
[t_slist,s_slist] = ode89(@cr3bp_dyn,-1*tspan,s_list(i,:)'-alpha*stable_vec,intoptions);
plot3(s_ulist(:,1),s_ulist(:,2),s_ulist(:,3),color=[0,0,1])
plot3(s_slist(:,1),s_slist(:,2),s_slist(:,3),color=[1,0,0])
%[t_list,s_list] = ode89(@cr3bp_dyn,tspan,s0,intoptions);
end

%% just the one for now
%{
Mon = getMonodromyMat(s0,tf,mu);
[V,D] = eigs(Mon);
%[norm(D(:,1)),norm(D(:,2)),norm(D(:,4)),norm(D(:,6))]
unstable_vec = V(:,1);
stable_vec = V(:,6);
alpha = 1e-9;
tspan = [0 7*tf]; 
[t_ulist,s_ulist] = ode89(@cr3bp_dyn,tspan,s0+alpha*unstable_vec,intoptions);
[t_list,s_list] = ode89(@cr3bp_dyn,tspan,s0,intoptions);
plot(s_list(:,1),s_list(:,2),"Color","r")
hold on
plot(s_ulist(:,1),s_ulist(:,2),"Color","b")
scatter(1-mu,0,20,"black",'filled','o')
%scatter(-mu,0,50,"black",'filled','o')
%}

function dy = cr3bp_dyn(~,y)
    mu = 7.3476E22/(5.9724E24+7.3476E22);
    r = y(1:3); v = y(4:6);
    h = [2*v(2);-2*v(1);0];
    dy(1:3) = v; dy(4:6) = grad_psU_cr3bp(r,mu) + h;
    dy = dy';
end

function dpsudr = grad_psU_cr3bp(r,mu)
    x = r(1); y = r(2); z = r(3);
    d1_3 = norm([x+mu;y;z])^3; d2_3 = norm([x-1+mu;y;z])^3;
    dpsudr = [x - (1-mu)*(x+mu)/d1_3 - mu*(x-1+mu)/d2_3; y - (1-mu)*y/d1_3 - mu*y/d2_3; -(1-mu)*z/d1_3 - mu*z/d2_3];
end

function d2y = d_cr3bp_dyn(y,mu)
    r = y(1:3);
    ddrdr = zeros(3);
    ddrdv = eye(3);
    ddvdr = hess_psU_cr3bp(r,mu);
    ddvdv = [0,2,0;-2,0,0;0,0,0];
    d2y = [ddrdr,ddrdv;ddvdr,ddvdv];
end

function d2psudr2 = hess_psU_cr3bp(r,mu)
    x = r(1); y = r(2); z = r(3);
    d1_3 = norm([x+mu;y;z])^3; d2_3 = norm([x-1+mu;y;z])^3;
    d1_5 = norm([x+mu;y;z])^5; d2_5 = norm([x-1+mu;y;z])^5;
    xx = 1 - (1-mu)/d1_3  - mu/d2_3 + (3/d1_5)*(1-mu)*(x+mu)^2 + (3/d2_5)*mu*(x-1+mu)^2;
    yy = 1 - (1-mu)/d1_3 - mu/d2_3 + (3/d1_5)*(1-mu)*y^2 + (3/d2_5)*mu*y^2;
    zz = -(1-mu)/d1_3 - mu/d2_3 + (3/d1_5)*(1-mu)*z^2 + (3/d2_5)*mu*z^2;
    xy = 3*(1-mu)*(x+mu)*y/d1_5 + 3*mu*(x-1+mu)*y/d2_5;
    xz = 3*(1-mu)*(x+mu)*z/d1_5 + 3*mu*(x-1+mu)*z/d2_5;
    yz = 3*(1-mu)*y*z/d1_5 + 3*mu*y*z/d2_5;
    d2psudr2 = [xx,xy,xz;xy,yy,yz;xz,yz,zz];
end

function dy = cr3bp_aug(~,y)
    mu = 7.3476E22/(5.9724E24+7.3476E22);
    s = y(1:6); Phi = reshape(y(7:42),6,6);
    dy(1:6) = cr3bp_dyn(0,y); dy(7:42) = reshape(d_cr3bp_dyn(s,mu)*Phi,[36, 1]);
    dy = dy';
end

function Mon = getMonodromyMat(s0,tf,mu)
    tspan = [0 2*tf]; 
    stmvec = reshape(eye(6),[36, 1]);
    s0 = [s0;stmvec]; 
    intoptions = odeset('RelTol',1e-13,'AbsTol',1e-13);
    [~,s] = ode89(@cr3bp_aug,tspan,s0,intoptions);
    Mon = reshape(s(end,7:42),[6,6]);
end




