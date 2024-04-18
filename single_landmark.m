% clear variables
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

% initial guess value for uncertainty of state estimate 
prediction(:,:,1) = diag([0 0 0])*10^-3;

% dynamic process noise having zero mean and covariance Q
process_noise = diag([sd_vel, sd_omega]);

% lidar's measurement noise having zero mean and covariance R
measurement_noise = diag([sd_range1, sd_bearing]);

% initialization
xp(:, 1) = [1; 1; 0];
xc(:,1) = [1;1;0];
ChiStat = zeros(1,length(velocity));

%% prediction
for i = 2:length(velocity)
    tp = wrapToPi(xp(3,i-1));
    
    % state space model
    xp(:,i) = xp(:,i-1) + dt *[cos(tp) 0;
                               sin(tp) 0;
                                   0   1] * ([velocity(i-1) + sd_vel ; omega(i-1) + sd_omega]);
    % state transition matrix
    F = [1 0 -dt*sin(tp)*velocity(i-1); 0 1 dt*cos(tp)*velocity(i-1); 0 0 1];
    tou = [dt*cos(tp) 0;
           dt*sin(tp) 0;
                0    dt];
    % prediction step based on approximation
    prediction(:,:,i) = F*prediction(:,:,i-1)*F' + tou*process_noise*tou';
    xc(:,i) = xp(:,i);
end

x_state = zeros(3, 441);
x_state(:,1) = [1; 1; 0];
true_measurement(:,1) = [range(1,1); wrapToPi(bearing(1,1))];
F = eye(3);
prediction = diag([0 0 0]);
%% corection
for i = 2:length(omega)
    % bounding values between -pi and +pi
    tk = wrapToPi(x_state(3,i-1));

    %% state space model
    % prediction step
    x_state(:, i) = x_state(:, i-1) + dt* [cos(tk) 0; 
                                           sin(tk) 0; 
                                           0 1]  * ([velocity(i-1)+sd_vel; omega(i-1)+sd_omega]);
    % state transition matrix from 't-1' to 't'
    F = [1 0 -dt*sin(tk)*velocity(i-1);
         0 1 dt*cos(tk)*velocity(i-1);
         0 0 1];

    % process noise
    tou = [dt*cos(tk) 0 ; dt*sin(tk) 0; 0 dt];
    
    % approximation for prediction step
    prediction = F*prediction*F' + tou*process_noise*tou';

    landmark_x = lm(1, 1);
    landmark_y = lm(1, 2);

    % Predicted values 
    x_k = x_state(1, i); % Predicted x coordinate
    y_k = x_state(2, i); % Predicted y coordinate
    theta_k = wrapToPi(x_state(3, i)); % Predicted theta
    
    delta_x = (x_k - landmark_x);
    delta_y = (y_k - landmark_y);

    % Jacobian matrix: linearization of the measurement model Dimensions: 2 x 3
    H = zeros(2, 3);
    H1 = delta_x/ sqrt(delta_x^2 + delta_y^2);
    H2 = delta_y / sqrt(delta_x^2 + delta_y^2);
    H3 = 0;
    H4 = delta_y / ((-delta_x)^2 + (-delta_y)^2);
    H5 = delta_x / ((-delta_x)^2 + (-delta_y)^2);
    H6 = -1;
    H = [H1 H2 H3; H4 H5 H6];
    
    % measurements from LIDAR sensor
    true_measurement(:,i) = [range(i, 1); wrapToPi(bearing(i, 1))];

    % observed model
    predicted_measurement(:,i) = [sqrt(delta_x^2 + delta_y^2); wrapToPi(atan2(-delta_y, -delta_x)) - wrapToPi(theta_k)];
    
    % innovation covariance
    inn_cov = inv(H*prediction*H' + measurement_noise);
    
    % Kalman gain
    Kalman_gain = prediction*H'*inn_cov;
    
    % measurement residual
    measurement_difference = true_measurement(:,i) - predicted_measurement(:,i);
    
    % update state estimate
    x_state(:,i) = x_state(:,i) + Kalman_gain*measurement_difference;
    
    % updated covariance of state estimate
    prediction = prediction - Kalman_gain*H*prediction;

    % computing chi squared statistics
    ChiStat(i) = measurement_difference'*inn_cov*measurement_difference;
end

% Robot(bicycle model) tracking 
figure(1); 
hold on;
title('Robot tracking for single landmark');
plot(xp(1,:), xp(2,:), 'g', 'LineWidth', 3); % predicted state
hold on;
plot(x_state(1,:), x_state(2,:), 'blue', 'LineWidth', 2); % updated state
hold on;
plot(lm(1,1), lm(1,2), 'ko', 'MarkerFaceColor', 'black'); % point for landmark 1
xlabel("x");
ylabel("y");
legend('Predicted trajectory', 'Updated trajectory - landmark 1', 'Landmark 1')
hold off;

% chi-squared statistics
figure(2)
plot(time(1,2:442),ChiStat)
xlabel('Time')
ylabel('chi-squared')
title('chi-squared statistics')


% Bearing for landmark_1
figure(3);
hold on ; 
title("Bearing for landmark-1")
plot(time(1,2:442),true_measurement(2,:));
plot(time(1,2:442),predicted_measurement(2,:));
legend("bearing measurement" ,"filtered values" )
xlabel("time");
ylabel("z(bearing)");
hold off ;

% Range for landmark_1
figure(4);
hold on ; 
title("range for landmark-1 ")
plot(time(1,2:442),true_measurement(1,:));
plot(time(1,2:442),predicted_measurement(1,:));
legend("range measurement" ,"filtered values" )
xlabel("time");
ylabel("z(range)");
hold off ;
