% Modular Active Inference Agent with EFE-based Action Selection
% Architecture: 11 modules (as per prior implementation)
% Environment: 1D drone altitude control (target = 0.5)

clear; 

%% Settings
T = 50;                           % time horizon
nM = 11;                          % number of modules
goal_altitude = 0.5;             % target for drone position

%% Module Names (11-node architecture)
M.name = {
    'Meta-Learning & Meta-Structuring';
    'Structure Learning';
    'Compositionality & Contextuality';
    'Self vs World Model';
    'Modularity & Factorisation';
    'Hierarchical Temporal Abstraction';
    'Precision Weighting & Attention';
    'Object-Centred Representation';
    'Relational Inference';
    'Invariant & Equivariant Mapping';
    'Theory of Mind (ToM)';
};

%% States, Inputs, Outputs
M.x = zeros(nM, 1);
M.precision = ones(nM, 1);
M.u = zeros(T, 1);                 % action over time
M.y = zeros(nM, 1);                % predicted output
M.output = zeros(nM, T);

%% Adjacency matrix (DCM-style: M(to,from))
Gf = zeros(nM);
Gf(2,1) = 1; Gf(3,1) = 1;              % Meta-learning → SL, Context
Gf(4,2) = 1; Gf(5,2) = 1;              % SL → SelfWorld, Modular
Gf(5,3) = 1; Gf(6,3) = 1;              % Context → Modular, HTA
Gf(7,4) = 1; Gf(8,4) = 1;              % SelfWorld → Precision, Object
Gf(6,5) = 1; Gf(9,5) = 1;              % Modular → HTA, Relational
Gf(10,6) = 1;                          % HTA → Invariant
Gf(4,11) = 1; Gf(9,11) = 1;            % ToM → SelfWorld, Relational

M.G_fwd = Gf;
M.G_bwd = Gf';
M.pE.F = randn(nM) .* Gf;
M.pE.B = randn(nM) .* Gf';

%% Module Dynamics (arbitrary nonlinearities)
g = cell(nM,1);
g{1} = @(x,u) tanh(x + u);                       % Meta-learning
g{2} = @(x,u) tanh(x + u);                       % Structure learning
g{3} = @(x,u) tanh(x .* u);                      % Contextuality
g{4} = @(x,u) 1./(1 + exp(-x - u));              % Self vs World
g{5} = @(x,u) x + u;                             % Modular relay
g{6} = @(x,u) x + 0.5 * sin(u);                  % Temporal abstraction
g{7} = @(x,u) exp(-x.^2) .* u;                   % Precision weighting
g{8} = @(x,u) tanh(x + u);                       % Object representation
g{9} = @(x,u) sqrt(x.^2 + u.^2);                 % Relational inference
g{10}= @(x,u) exp(-abs(x - u));                  % Invariant mapping
g{11}= @(x,u) tanh(x + 0.5 * sin(u));            % Theory of Mind
M.g = g;

%% Simulate sensory environment: agent senses drone altitude error
altitude = 0.2;                 % initial altitude
for t = 1:T-1

    % Observation: altitude error (goal - current)
    sensory_input = goal_altitude - altitude;

    % EFE-based action: agent chooses action to reduce expected error
    E = sensory_input^2;               % simplified EFE (expected cost)
    M.u(t) = -0.1 * sensory_input;     % gradient step

    % Belief propagation
    for i = 1:nM
        % Collect inputs from parents
        input_sum = sum(M.pE.F(i,:) .* M.y(:)');

        % Add sensory input to Self vs World (node 4)
        if i == 4
            y_obs = sensory_input;
        else
            y_obs = 0;
        end

        % Module dynamics
        M.x(i) = M.x(i) + 0.1 * (-M.x(i) + g{i}(input_sum, y_obs));
        M.y(i) = M.x(i);
        M.output(i,t) = M.y(i);
    end

    % Environment update: drone responds to agent action
    altitude = altitude + M.u(t);
end

%% Plot
figure;
for i = 1:nM
    subplot(nM+1,1,i);
    plot(1:T, M.output(i,:), 'LineWidth', 1.2);
    title(M.name{i}); ylabel('Output');
end
subplot(nM+1,1,nM+1);
plot(1:T, M.u, 'r', 'LineWidth', 1.2);
title('Action (u)'); xlabel('Time'); ylabel('Action');
