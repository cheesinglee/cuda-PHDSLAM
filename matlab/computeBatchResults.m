close all
pause on
parentpath = uigetdir('../','Choose parent directory') ;
if ( path == 0 )
    return 
end

listing = dir([parentpath,'/2*']) ;
nRuns = size(listing) ;

for n = 1:nRuns
    subfolder = listing(n).name ;
    
end