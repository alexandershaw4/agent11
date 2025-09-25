function DCM = update_dcm_modules_11(DCM, sensory_input, t)
% Update function for 11-module DCM-style Active Inference agent

nM = DCM.M.nM;
Gf = DCM.M.G_fwd;
Gb = DCM.M.G_bwd;
pF = DCM.M.pE.F;
pB = DCM.M.pE.B;
g  = DCM.M.g;
x  = DCM.M.x;

% === Top-down generative pass ===
for i = 1:nM
    % Determine sensory or prior observation
    if ismember(i, [7 8 9 10])  % Precision, Object, Relation, Invariance
        y_obs = sensory_input;

    elseif i == 11 && t < length(DCM.M.u)
        % Optional: ToM watches own action history (as proxy for other's actions)
        y_obs = DCM.M.u(t);
    elseif i == 11 && t > 1
        y_obs = DCM.M.u(t - 1);  % ToM predicts own past actions (introspection)

    else
        y_obs = 0;
    end

    % Predictive inputs from parent modules (top-down)
    topdown_input = sum(pF(i, Gf(i,:) ~= 0) .* x(Gf(i,:) ~= 0)');

    % Predicted output
    y_pred = g{i}(x(i), topdown_input);
    err = y_obs - y_pred;

    % Numerical derivative of g{i} wrt x(i)
    dx = 1e-6;
    g1 = g{i}(x(i) + dx, topdown_input);
    g0 = g{i}(x(i) - dx, topdown_input);
    dgdx = (g1 - g0) / (2 * dx);

    % Variational update of latent state
    x(i) = x(i) + DCM.M.precision(i) * err * dgdx;
    DCM.M.y(i) = g{i}(x(i), topdown_input);

    % Action selection (ToM module predicts next motor output)
    if i == 11 && t < length(DCM.M.u)
        DCM.M.u(t+1) = double(DCM.M.y(11) > 0.5);  % binary thresholded
    end
end

% === Bottom-up feedback (PE or salience signals) ===
eta = 0.1;
for i = 1:nM
    for j = find(Gb(i,:))  % j → i connections (child to parent)
        % Parent i predicts what j will output
        topdown_to_j = pF(j,i) * x(i);
        expected_j = g{j}(x(j), topdown_to_j);

        % Error between j's actual output and expected
        bu_err = DCM.M.y(j) - expected_j;

        % Apply to parent i
        x(i) = x(i) + eta * DCM.M.precision(i) * pB(i,j) * bu_err;
    end
end

% Store updated latent states and outputs
DCM.M.x = x;
DCM.M.output(:,t+1) = DCM.M.y;
end
