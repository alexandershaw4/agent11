function y_pred = predict_bounce_y(ball, vel, H, tau)
    pos = ball(2);
    vy = vel(2);
    for step = 1:tau
        pos = pos + vy;
        if pos <= 0
            pos = -pos;          % bounce off bottom
            vy = -vy;
        elseif pos >= H
            pos = 2*H - pos;     % bounce off top
            vy = -vy;
        end
    end
    y_pred = pos;
end