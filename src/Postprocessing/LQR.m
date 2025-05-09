clear all
close all
clc

% Load Data
load('Koopman_ShamHC_all.mat')
r = Koopman_ShamHC_All(1:10000,:)';
load('Koopman_ShamPD2_all.mat')


% Define system matrices
A_PD2 = [0.999 0.00003 0; -0.00003 0.999 0; 0 0 0.951];
B_PD2 = eye(3); % Given Bpd2 is identity matrix

% Check Controllability
C = ctrb(A_PD2, B_PD2);
rank_C = rank(C);

if rank_C == size(A_PD2, 1)
    disp('The system is controllable.');
else
    disp('The system is not controllable.');
end

% Define weighting matrices
Q = eye(3); % Weighting matrix for state
R = eye(3); % Weighting matrix for control inputs

% Solve the discrete-time algebraic Riccati equation (DARE) to get the LQR gain
[K,~,~] = dlqr(A_PD2,B_PD2,Q,R);

% Simulate the system with LQR control
x_LQR = zeros(3,10000);
x_LQR(:,1) = Koopman_ShamPD2_All(1,:)';

% Initialize arrays to store control input and error
u_LQR_array = zeros(3,10000-1);
e_array = zeros(3,10000-1);

for i = 1:10000-1
    e = r(:,i) - x_LQR(:,i); % Calculate error
    u_LQR = K * e; % Calculate control input using LQR gain
    x_LQR(:,i+1) = A_PD2 * x_LQR(:,i) + B_PD2 * u_LQR; % Update state

    % Store control input and error
    u_LQR_array(:,i) = u_LQR;
    e_array(:,i) = e;
end

%% 
% Calculate the closed-loop system matrix
A_CL = A_PD2 - B_PD2 * K;

% Calculate the eigenvalues of the closed-loop system
eig_values_CL = eig(A_CL);
disp('Eigenvalues of the Controlled (Closed-Loop) System:');
disp(eig_values_CL);


% Calculate error metrics for each state variable separately
mse_x1 = mean(e_array(1, :).^2); % Mean Squared Error for x1
mae_x1 = mean(abs(e_array(1, :))); % Mean Absolute Error for x1
max_error_x1 = max(abs(e_array(1, :))); % Maximum Absolute Error for x1

mse_x2 = mean(e_array(2, :).^2); % Mean Squared Error for x2
mae_x2 = mean(abs(e_array(2, :))); % Mean Absolute Error for x2
max_error_x2 = max(abs(e_array(2, :))); % Maximum Absolute Error for x2

mse_x3 = mean(e_array(3, :).^2); % Mean Squared Error for x3
mae_x3 = mean(abs(e_array(3, :))); % Mean Absolute Error for x3
max_error_x3 = max(abs(e_array(3, :))); % Maximum Absolute Error for x3

% Generate sentences describing controller performance for each state variable
error_sentence_x1 = sprintf('For state x1, the controller achieved a Mean Squared Error (MSE) of %.4f, a Mean Absolute Error (MAE) of %.4f, and a Maximum Absolute Error of %.4f.', ...
    mse_x1, mae_x1, max_error_x1);
error_sentence_x2 = sprintf('For state x2, the controller achieved a Mean Squared Error (MSE) of %.4f, a Mean Absolute Error (MAE) of %.4f, and a Maximum Absolute Error of %.4f.', ...
    mse_x2, mae_x2, max_error_x2);
error_sentence_x3 = sprintf('For state x3, the controller achieved a Mean Squared Error (MSE) of %.4f, a Mean Absolute Error (MAE) of %.4f, and a Maximum Absolute Error of %.4f.', ...
    mse_x3, mae_x3, max_error_x3);

% Display the sentences
disp(error_sentence_x1)
disp(error_sentence_x2)
disp(error_sentence_x3)


% Plotting

% figure('Name', 'LQR Controller')
subplot(3,1,1)
plot(r(1,1:1000),'Color',[0.1 0.7 0.2],'LineWidth',1.5)
hold on
plot(x_LQR(1,1:1000),'--','Color',[0.9 0.2 0.1],'LineWidth',1.5)
% title('x1')
xlabel('Time(ms)')
ylabel('x1')
legend('x_{HC}','x_{PD2}','Location','best')

subplot(3,1,2)
plot(r(2,1:1000),'Color',[0.1 0.7 0.2],'LineWidth',1.5)
hold on
plot(x_LQR(2,1:1000),'--','Color',[0.9 0.2 0.1],'LineWidth',1.5)
% title('x2')
xlabel('Time(ms)')
ylabel('x2')

subplot(3,1,3)
plot(r(3,1:1000),'Color',[0.1 0.7 0.2],'LineWidth',1.5)
hold on
plot(x_LQR(3,1:1000),'--','Color',[0.9 0.2 0.1],'LineWidth',1.5)
% title('x3')
xlabel('Time(ms)')
ylabel('x3')
% Set title for the entire figure
% sgtitle('LQR Controller')
% Set legends for all subplots



% Calculate mean squared error (MSE)
mse = mean(error.^2, 2);

% Calculate mean absolute error (MAE)
mae = mean(abs(error), 2);

% Calculate maximum absolute error
max_error = max(abs(error), [], 2);

% Display error metrics
disp('Error Metrics:')
disp(['Mean Squared Error (MSE): ', num2str(mse')])
disp(['Mean Absolute Error (MAE): ', num2str(mae')])
disp(['Maximum Absolute Error: ', num2str(max_error')])



save('u_LQR.mat',"u_LQR_array")
h5create('u_LQR.h5', '/dataset', size(u_LQR_array));
h5write('u_LQR.h5', '/dataset', u_LQR_array);





