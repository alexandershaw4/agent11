% DCM_PongAgent11_Predictive.m
% -----------------------------------------
% Improved 11-module DCM agent for 2D Pong with:
% - full 2D sensory input
% - EFE-based planning over future ball trajectory
% - temporal abstraction node
% - predictive forward simulation of bounce

clear;

% Parameters
T = 2000;                    % time steps
W = 100; H = 100;           % environment size
ball = [W/2, H/2];          % initial ball position
ball_vel = [1, 1];          % initial ball velocity
paddle_y = H/2;             % paddle y-position
paddle_x = W - 2;           % paddle x-position
paddle_h = 10;              % paddle height

% Agent Setup
nM = 11;
M.nM = nM;
M.x = zeros(nM, 1);
M.precision = ones(nM,1);
M.y = zeros(nM,1);
M.u = zeros(T,1);
M.output = zeros(nM,T);

M.name = {
    'Meta-Learning'; 
    'Structure Learning'; 
    'Compositionality'; 
    'Self vs World';
    'Modularity'; 
    'Temporal Abstraction'; 
    'Precision Weighting'; 
    'Object Representation';
    'Relational Inference'; 
    'Invariant Mapping'; 
    'Theory of Mind';
};

Gf = zeros(nM);
Gf(1,[2 3]) = 1;
Gf(2,[4 5]) = 1;
Gf(3,[5 6]) = 1;
Gf(4,[7 8]) = 1;
Gf(5,[6 9]) = 1;
Gf(6,10) = 1;
Gf(11,[4 6 9]) = 1;
M.G_fwd = Gf;
M.G_bwd = Gf';

M.pE.F = randn(nM,nM) .* Gf;
M.pE.B = randn(nM,nM) .* Gf';

% Denerative functions per unit - TOYS
%------------------------------------------------------
M.g = cell(nM,1);
M.g{1}  = @(x,u) tanh(x + u);
M.g{2}  = @(x,u) tanh(x + u);
M.g{3}  = @(x,u) tanh(x .* u);
M.g{4}  = @(x,u) 1./(1 + exp(-x - u));
M.g{5}  = @(x,u) x + u;
M.g{6}  = @(x,u) x + 0.5 * sin(u);
M.g{7}  = @(x,u) exp(-x.^2) .* u;
M.g{8}  = @(x,u) tanh(x + u);
M.g{9}  = @(x,u) sqrt(x.^2 + u.^2);
M.g{10} = @(x,u) exp(-abs(x - u));
M.g{11} = @(x,u) tanh(x + 0.5 * sin(u));
g = M.g;

ball_path = zeros(T, 2);
paddle_path = zeros(T,1);

just_bounced = false;  % add above loop

for t = 1:T-1
    % -- Predictive Update (Generative Model) --
    sensory_input = [ball(1), ball(2), ball_vel(1), ball_vel(2)];
    x = M.x; Gf = M.G_fwd; pF = M.pE.F;

    for i = 1:nM
        if i == 4     % Self vs World
            y_obs = sensory_input(2);
        elseif i == 6 % Temporal abstraction
            y_obs = sensory_input(4);
        elseif i == 11
            y_obs = M.u(t);
        else
            y_obs = 0;
        end
        top_input = sum(pF(i,:) .* x');
        y_pred = g{i}(x(i), top_input);
        err = y_obs - y_pred;
        dx = 1e-6;
        dgdx = (g{i}(x(i)+dx,top_input) - g{i}(x(i)-dx,top_input)) / (2*dx);
        x(i) = x(i) + M.precision(i) * err * dgdx;
        M.y(i) = g{i}(x(i), top_input);
    end

    % -- EFE-Based Action Planning --
    tau = 8;
    pred_ball_y = ball(2) + tau * ball_vel(2);
    pred_ball_y = max(min(pred_ball_y, H), 1);
    
    EFE = zeros(2,1);
    for a = [-1 1]
        y_new = min(max(paddle_y + a * 2, 1), H);
        E_p = (y_new - pred_ball_y)^2;
        EFE((a+3)/2) = E_p;
    end
    [~, best_a] = min(EFE);
    action_choices = [-1, 1];
    action = action_choices(best_a);

    paddle_y = min(max(paddle_y + action * 2, 1), H);
    paddle_y = 0.8 * paddle_y + 0.2 * y_new;  % EMA smoothing
    M.u(t+1) = action;


    % Paddle Collision Detection --
    ball_r = 1; paddle_w = 2; collision_margin = 2;
    if abs(ball(1) - paddle_x) < (ball_r + paddle_w + collision_margin) && ...
       abs(ball(2) - paddle_y) < (ball_r + paddle_h/2)
    
        % Reflect X-velocity
        ball_vel(1) = -abs(ball_vel(1));  % ensure it always goes left after paddle
    
        % Adjust Y-velocity based on impact point
        %offset = (ball(2) - paddle_y) / (paddle_h / 2);  % [-1, 1]
        %ball_vel(2) = offset * norm(ball_vel);  % stronger, full deflection
   

        % Adjust Y-velocity based on impact point, preserving incoming angle
        offset = (ball(2) - paddle_y) / (paddle_h / 2);  % [-1, 1]
        ball_speed = norm(ball_vel);
        bounce_angle = atan2(ball_vel(2), -ball_vel(1));  % flip x-direction

        % Modulate angle based on offset (adds angle in vertical plane)
        angle_adjust = offset * (pi/4);  % up to ±45 degrees
        new_angle = bounce_angle + angle_adjust;

        % Convert back to velocity vector
        ball_vel(1) = ball_speed * cos(new_angle);
        ball_vel(2) = ball_speed * sin(new_angle);

        % Optional: Normalize speed to keep ball from accelerating too much
        speed = norm(ball_vel);
        desired_speed = 1.4;  % tweak this value
        ball_vel = ball_vel / speed * desired_speed;
    end

    % --- Paddle Collision Detection ---
    ball_r = 1;
    paddle_w = 2;
    collision_margin = 1;

    if abs(ball(1) - paddle_x) < (ball_r + paddle_w + collision_margin) && ...
            abs(ball(2) - paddle_y) < (ball_r + paddle_h/2)

        % Reflect horizontal velocity (send left)
        ball_vel(1) = -abs(ball_vel(1));

        % Add vertical deflection based on impact offset
        offset = (ball(2) - paddle_y) / (paddle_h / 2);   % range [-1, 1]
        ball_vel(2) = ball_vel(2) + offset * 1.2;         % tune 1.2 as needed

        % Optional: Clamp vertical speed to avoid too steep
        max_vy = 2.5;
        ball_vel(2) = max(min(ball_vel(2), max_vy), -max_vy);

        % Renormalize total speed if needed
        desired_speed = 1.5;
        ball_vel = (ball_vel / norm(ball_vel)) * desired_speed;

        just_bounced = true;
    end
    % % -- Predict Next Ball Position and Bounce Logic --
    % next_ball = ball + ball_vel;
    % 
    % if (next_ball(2) <= 0 && ball_vel(2) < 0) || (next_ball(2) >= H && ball_vel(2) > 0)
    %     ball_vel(2) = -ball_vel(2);
    % end
    % if (next_ball(1) <= 0 && ball_vel(1) < 0) || (next_ball(1) >= W && ball_vel(1) > 0)
    %     ball_vel(1) = -ball_vel(1);
    % end

    % -- Predict Next Ball Position and Bounce Logic --
    next_ball = ball + ball_vel;

    % Ball-wall bounce: disable X-bounce for one frame after paddle bounce
    if ~just_bounced
        if (next_ball(1) <= 0 && ball_vel(1) < 0) || (next_ball(1) >= W && ball_vel(1) > 0)
            ball_vel(1) = -ball_vel(1);
        end
    end

    % Always apply Y bounce
    if (next_ball(2) <= 0 && ball_vel(2) < 0) || (next_ball(2) >= H && ball_vel(2) > 0)
        ball_vel(2) = -ball_vel(2);
    end

    % % -- Avoid ball sticking to edge --
    % if ball(1) == 0 || ball(1) == W
    %     ball(1) = ball(1) + randn() * 0.1;
    % end
    % if ball(2) == 0 || ball(2) == H
    %     ball(2) = ball(2) + randn() * 0.1;
    % end

    % -- Update Position --
    ball = ball + ball_vel;
    ball(1) = max(min(ball(1), W), 0);
    ball(2) = max(min(ball(2), H), 0);

    % Reset just_bounced if ball has moved away
    if abs(ball(1) - paddle_x) > (ball_r + paddle_w + collision_margin + 1)
        just_bounced = false;
    end

    % -- Optional: Prevent near-zero velocity stalling --
    if abs(ball_vel(1)) < 0.2, ball_vel(1) = sign(ball_vel(1)) * 0.2; end
    if abs(ball_vel(2)) < 0.2, ball_vel(2) = sign(ball_vel(2)) * 0.2; end

    % Store
    ball_path(t,:) = ball;
    paddle_path(t) = paddle_y;
    
    M.x = x;
    M.output(:,t) = M.y;
end

% === Visualisation and Video ===
%video_filename = 'DCM_PongAgent11_predictive.mp4';
%v = VideoWriter(video_filename, 'MPEG-4');
%v.FrameRate = 20;
%open(v);

fig = figure('Color','w', 'Position', [100, 100, 600, 600]);
ax = axes('Parent', fig);
axis(ax, [0 W 0 H]);
axis(ax, 'manual');
daspect(ax, [1 1 1]);
grid(ax, 'on');
box(ax, 'on');
xlabel(ax, 'X'); ylabel(ax, 'Y');
set(ax, 'FontSize', 12);

for t = 1:T
    cla(ax);
    hold(ax, 'on');

    trail_len = 20;
    trail_start = max(1, t - trail_len);
    plot(ax, ball_path(trail_start:t,1), ball_path(trail_start:t,2), ...
         'Color', [1 0 0 0.3], 'LineWidth', 2);
    plot(ax, ball_path(t,1), ball_path(t,2), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
    rectangle(ax, 'Position', [W-2, paddle_path(t)-paddle_h/2, 2, paddle_h], ...
              'FaceColor', [0.2 0.4 1], 'EdgeColor','k', 'LineWidth', 1.5);
    title(ax, ['Time Step: ', num2str(t)], 'FontSize', 14);

    drawnow;

    % Capture full figure as an image (with labels, axis, no distortion)
    %frame_img = print(fig, '-RGBImage');
    %writeVideo(v, frame_img);
end

%close(v);
%disp(['Video saved: ', video_filename]);


% % === Visualisation and Video ===
% video_filename = 'DCM_PongAgent11_predictive.mp4';
% v = VideoWriter(video_filename, 'MPEG-4');
% v.FrameRate = 20;
% open(v);
% 
% figure('Color','w', 'Position', [100, 100, 600, 600]);
% 
% for t = 1:T
%     clf;
%     hold on;
%     trail_len = 20;
%     trail_start = max(1, t - trail_len);
%     plot(ball_path(trail_start:t,1), ball_path(trail_start:t,2), ...
%          'Color', [1 0 0 0.3], 'LineWidth', 2);
%     plot(ball_path(t,1), ball_path(t,2), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
%     rectangle('Position', [W-2, paddle_path(t)-paddle_h/2, 2, paddle_h], ...
%               'FaceColor', [0.2 0.4 1], 'EdgeColor','k', 'LineWidth', 1.5);
%     xlim([0 W]); ylim([0 H]);
%     daspect([1 1 1]);
%     grid on; box on;
%     title(['Time Step: ', num2str(t)], 'FontSize', 14);
%     xlabel('X'); ylabel('Y');
%     set(gca, 'FontSize', 12);
%     drawnow;
%     writeVideo(v, getframe(gca));
% end
% 
% close(v);
% disp(['Video saved: ', video_filename]);
