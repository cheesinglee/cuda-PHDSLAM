function examineWeights
    global i nSteps
    load 'weightUpdates.log'
    nSteps = size(weightUpdates,1) ;
    figure('KeyPressFcn',@processKey) 
    i = 1 ;
    while true
        w = weightUpdates(i,:) ;
        semilogy(w,'.') ;
        ylim([0,1e-13])
        title(num2str(i)) ;
        uiwait ;
    end
end

function processKey(src,event)
    global i nSteps
    if (strcmp(event.Key,'backspace'))
       if ( i > 1)
           i = i - 1 ;
       end
    elseif ( i < nSteps)
        i = i+1 ;
    end
    uiresume;
end