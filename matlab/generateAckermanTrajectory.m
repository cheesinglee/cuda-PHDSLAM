function [ traj,control ] = generateAckermanTrajectory( initial_state, motion_model, map, dt )
%GENERATEACKERMANTRAJECTORY Summary of this function goes here
%   Detailed explanation goes here
    close all
    figure(111)
    callstr = 'set(gcbf,''Userdata'',double(get(gcbf,''Currentcharacter''))) ; uiresume ' ;
    set(gcf,'keypressfcn',callstr) ;
    hold on
    grid on
    axis equal
    traj = initial_state(1:3) ;
    u = [0;0] ;
    control = [] ;
    
    alpha_step = deg2rad(10) ;
    v_step = 2 ;
    
    % set up the timer
    tt = timer ;
    tt.timerfcn = 'uiresume' ;
    tt.startdelay = 0.5 ;  
    while 1
        set(gcf,'UserData',-1) ;  
        start(tt) ;
        uiwait(gcf) ;
        ch = get(gcf,'UserData') ;
        stop(tt) ;
        if ch == 27 % esc
            break
        elseif ch == 30 % up arrow
            u(1) = u(1) + v_step ;
        elseif ch == 31 % down arrow
            u(1) = u(1) - v_step ;
        elseif ch == 28 % left arrow
            if wrapAngle(u(2) + alpha_step) < deg2rad(90) ;
                u(2) = wrapAngle(u(2) + alpha_step) ;
            end
        elseif ch == 29 % right arrow
            if wrapAngle(u(2) - alpha_step) > -deg2rad(90)
                u(2) = wrapAngle(u(2) - alpha_step) ;
            end
        end
        tmp.u = u ;
        tmp.dt = dt ;
        new_state = motion_model.computeMotion(traj(:,end), tmp) ;
        traj = [traj,new_state] ;
                plot(map(1,:),map(2,:),'k*') ;
        plot(traj(1,:),traj(2,:),'k--')     
        title(['ve=',num2str(u(1)),' alpha= ', num2str(u(2))])
        drawnow
        control = [control,tmp] ;
    end
end

