% Clear variables
clear;clc;

% Load your data and measurements from the provided files
ds_1 = load('my_input.mat'); % Contains 'velocity', 'angular velocity', 'time interval'
ds_2 = load('my_measurements.mat'); % Contains 'range', 'bearing', 'landmarks'

range = ds_2.r;
bearing = ds_2.b;
time = ds_1.t;
velocity = ds_1.v;
omega = ds_1.om;
lm = ds_2.l;
dt = 0.1; % time step

% assumed variances for measurements and inputs
sd_vel = 0.01;
sd_omega = 0.25;
sd_range1 = 0.01;
sd_range2 = 0.09;
sd_bearing = 0.25;

% initial guess value for uncertaininty of state estimate 
prediction = diag([1 1 1])*10^-4;

% dynamic process noise having zero mean and covariance Q
process_noise = diag([sd_vel, sd_omega]);

% lidar's measurement noise having zero mean and covariance R
measurement_noise = diag([0.01,0.25,0.01,0.25,0.01,0.25,0.09,0.25,0.09,0.25,0.09,0.25]);

% initialization
x_state = zeros(3, 441);
jacobian_measurement = zeros(12,3);
x_corrected = zeros(3,441);
true_measurement = zeros(12,1);
predicted_measurement = zeros(12,1);
ChiStat = zeros(1,length(velocity));

%  intial state of robot
x_state(:, 1) = [1; 1; 0];
x_corrected(:, 1) = [1; 1; 0];

% KALMAN FILTER
for i = 2:length(omega)
    % bounding values between -pi and +pi
    x_state(3) = wrapToPi(x_state(3,i-1));
    
    %% state space model
    %% prediction step
    x_state(:, i) = x_state(:, i-1) + dt* [cos(x_state(3)) 0; 
                                           sin(x_state(3)) 0; 
                                                        0 1]  * ([velocity(i-1)+sd_vel; omega(i-1)+sd_omega]);
    % state transition matrix from 't-1' to 't'
    F = [1 0 -dt*sin(x_state(3))*velocity(i-1);
         0 1 dt*cos(x_state(3)*velocity(i-1));
         0 0 1;];

    % process noise
    tou = [dt*cos(x_state(3)) 0 ; dt*sin(x_state(3)) 0; 0 dt];
    
    % approximation for prediction step
    prediction = F*prediction*F' + tou*process_noise*tou';
    
    %% update / correction step
    % actual measurements
    true_measurement = [range(i, 1); bearing(i, 1); range(i, 2); bearing(i, 2); range(i, 3); bearing(i, 3); range(i, 4); bearing(i, 4); range(i, 5); bearing(i, 5); range(i, 6); bearing(i, 6)];
   
    for j = 1:6
        count = 2 * j - 1;
        % ground truth / actual location of landmarks
        landmark_x = lm(j, 1);
        landmark_y = lm(j, 2);
        
        % predicted states
        xm = x_state(1, i);
        ym = x_state(2, i);
        thetam = wrapToPi(x_state(3, i));
        
        % measurement model
        dx = xm - landmark_x;
        dy = ym - landmark_y;
        range_model = sqrt(dx^2 + dy^2);
        range_model_sq = range_model^2;

        % jacobian of measurement model
        jacobian_measurement(count, :) = [dx / range_model, dy / range_model, 0];
        jacobian_measurement(count + 1, :) = [-dy / range_model_sq, dx / range_model_sq, -1];

        % predicted measurement matrix
        dz1 = sqrt(dx^2 + dy^2);
        dz2 = wrapToPi(atan2(dy, dx) - thetam);
        predicted_measurement(count:count+1, :) = [dz1; dz2];


    end
    % innovation covariance
    inn_cov = (jacobian_measurement*prediction*jacobian_measurement' + measurement_noise)^(-1);

    % kalman gain
    Kalman_gain = prediction*jacobian_measurement'*inn_cov *12;

    % measurement residual
    measurement_difference = true_measurement-predicted_measurement;

    % updating state estimate
    x_state(:,i) = x_state(: ,i) + Kalman_gain*measurement_difference;
    x_corrected(:,i) = x_state(:, i);

    % updating covariance of state estimate
    prediction = prediction - Kalman_gain *jacobian_measurement*prediction;

    % chistat statistics
    ChiStat(i) = measurement_difference'*inn_cov*measurement_difference;
end

% single_landmark(1);
% single_landmark(2);
% single_landmark(3);
% single_landmark(4);
% single_landmark(5);
% single_landmark(6);

figure(1)
hold on;
title('sensor_fusion');
plot(x_corrected(1,:), x_corrected(2,:), 'r', 'LineWidth', 3);
hold on;
scatter(lm(:,1), lm(:,2), 'k', 'filled');
hold on;
axis auto;

figure(2)
hold on
plot(time(1,2:442),ChiStat)
xlabel('time')
ylabel('chistat-squared')
title('chistat-squared statistics for EKF')
