% === DCM Initialisation: 11-Node Agent (M(to, from)) ===

T = 50;                          % time horizon
nM = 11;                         % number of modules
DCM.M.nM = nM;

% --- Module Names ---
DCM.M.name = {
    'Meta-Learning & Meta-Structuring';        % 1
    'Structure Learning';                      % 2
    'Compositionality & Contextuality';        % 3
    'Self vs World Model';                     % 4
    'Modularity & Factorisation';              % 5
    'Hierarchical Temporal Abstraction';       % 6
    'Precision Weighting & Attention';         % 7
    'Object-Centred Representation';           % 8
    'Relational Inference';                    % 9
    'Invariant & Equivariant Mapping';         %10
    'Theory of Mind (ToM)';                    %11
};

% --- States, Precision, IO ---
DCM.M.x = zeros(nM, 1);                % latent states
DCM.M.precision = ones(nM, 1);         % precision per module
DCM.M.precision([1,7,11]) = 2;

DCM.M.u = zeros(T, 1);                 % action time series
DCM.M.y = zeros(nM, 1);                % predicted output per module
DCM.M.output = zeros(nM, T);           % full output trace

% --- Forward connections (M(to, from)) ---
Gf = zeros(nM);

% Meta-learning sends to Structure + Context
Gf(2,1) = 1;  % Structure ← Meta
Gf(3,1) = 1;  % Context ← Meta
Gf(1,1) = 1;

% Structure Learning sends to SelfWorld, Modularity, and ToM
Gf(4,2) = 1;  % SelfWorld ← Structure
Gf(5,2) = 1;  % Modularity ← Structure
Gf(11,2)= 1;  % ToM ← Structure

% Contextuality sends to Modularity + Abstraction
Gf(5,3) = 1;  % Modularity ← Context
Gf(6,3) = 1;  % Abstraction ← Context

% Self vs World sends to Precision + Object
Gf(7,4) = 1;  % Precision ← SelfWorld
Gf(8,4) = 1;  % Object ← SelfWorld

% Modularity sends to Abstraction + Relational
Gf(6,5) = 1;  % Abstraction ← Modularity
Gf(9,5) = 1;  % Relational ← Modularity

% Abstraction sends to Invariant
Gf(10,6) = 1; % Invariant ← Abstraction

% ToM receives from Invariant and Object
Gf(11,10) = 1; % ToM ← Invariant
Gf(11,8)  = 1; % ToM ← Object

% from TOM
Gf(4,11) = 1;  % ToM → Self-vs-World
Gf(9,11) = 1;  % ToM → Relational

DCM.M.G_fwd = Gf;
DCM.M.G_bwd = Gf';  % transpose for backward (child to parent)

% --- Parameters (structured priors) ---
DCM.M.pE.F = randn(nM, nM) .* Gf * 2;      % forward weights
DCM.M.pE.B = randn(nM, nM) .* Gf'* 2;     % backward weights

% --- Generative functions per module ---
g = cell(nM,1);
g{1}  = @(x,u) tanh(x + u);                        % Meta-learning
g{1} = @(x,u) tanh(x + u + 0.01 * randn);
g{2}  = @(x,u) tanh(x + u);                        % Structure learning
g{3}  = @(x,u) tanh(x .* u);                       % Context modulation
g{4}  = @(x,u) 1./(1 + exp(-x - u));               % Self/World classifier
g{5}  = @(x,u) x + u;                              % Modular relay
g{6}  = @(x,u) x + 0.5 * sin(u);                   % Temporal abstraction
g{7}  = @(x,u) exp(-x.^2) .* u;                    % Precision weighting
g{8}  = @(x,u) tanh(x + u);                        % Object-centred encoding
g{9}  = @(x,u) sqrt(x.^2 + u.^2);                  % Relational inference
g{10} = @(x,u) exp(-abs(x - u));                   % Invariant mapping
g{11} = @(x,u) tanh(x + u);                        % Theory of Mind
g{11} = @(x,u) tanh(x + 0.5 * sin(u));
DCM.M.g = g;

% --- Input stream (binary pulse) ---
sensory_input = zeros(T,1);
sensory_input(10:25) = 1;

% === Run Inference Loop ===
for t = 1:T-1
    DCM = update_dcm_modules_11(DCM, sensory_input(t), t);
end

% === Plot Outputs ===
figure;
for i = 1:nM
    subplot(nM+1,1,i);
    plot(1:T, DCM.M.output(i,:), 'LineWidth', 1.2);
    title(DCM.M.name{i}); ylabel('Output');
end
subplot(nM+1,1,nM+1);
plot(1:T, DCM.M.u, 'r', 'LineWidth', 1.2);
title('Action (u)'); xlabel('Time'); ylabel('Action');
