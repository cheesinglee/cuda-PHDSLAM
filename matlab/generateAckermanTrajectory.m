function [ traj,control ] = generateAckermanTrajectory( initial_state, motion_model, map )
%GENERATEACKERMANTRAJECTORY Summary of this function goes here
%   Detailed explanation goes here
    close all
    figure(111)
    callstr = 'set(gcbf,''Userdata'',double(get(gcbf,''Currentcharacter''))) ; uiresume ' ;
    set(gcf,'keypressfcn',callstr) ;
    hold on
    grid on
    axis equal
    traj = initial_state ;
    u = [0;0] ;
    control = [] ;
    
    % set up the timer
    tt = timer ;
    tt.timerfcn = 'uiresume' ;
    tt.startdelay = 1 ;  
    while 1
        set(gcf,'UserData',-1) ;  
        start(tt) ;
        uiwait(gcf) ;
        ch = get(gcf,'UserData') ;
        if ch == 27 % esc
            break
        elseif ch == 30 % up arrow
            u(1) = u(1) + 0.2 ;
        elseif ch == 31 % down arrow
            u(1) = u(1) - 0.2 ;
        elseif ch == 28 % left arrow
            u(2) = wrapAngle(u(2) + deg2rad(10)) ;
        elseif ch == 29 % right arrow
            u(2) = wrapAngle(u(2) - deg2rad(10)) ;
        end
        tmp.u = u ;
        tmp.dt = 1 ;
        new_state = motion_model.computeMotion(traj(:,end), tmp) ;
        traj = [traj,new_state] ;
                plot(map(1,:),map(2,:),'k*') ;
        plot(traj(1,:),traj(2,:),'k--')     
        title(['ve=',num2str(u(1)),' alpha= ', num2str(u(2))])
        drawnow
        control = [control,tmp] ;
    end
end

