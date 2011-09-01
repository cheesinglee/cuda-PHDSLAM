function particles = parseParticleFile(filename)
    if ( nargin==0 )
        [file,path] = uigetfile('../*.log') ;
        filename = [path,file] ;
    end
    fid = fopen(filename) ;
    if ( fid == -1 )
        error('Could not open file')
    end
    j = 1 ;
    while( ~feof(fid) )
        line = fgetl(fid) ;
        [C,pos] = textscan(line,'%f %f %f %f %f %f %f',1) ;
        particles.weights(j) = C{1} ;
        particles.poses(:,j) = [C{2} ; C{3} ; C{4} ; C{5} ; C{6} ; C{7}] ;
        particles.maps(j).weights = [] ;
        particles.maps(j).means = [] ;
        particles.maps(j).covs = [] ;
        if ( pos < length(line) )
            C = textscan(line(pos+1:end),'%f %f %f %f %f %f %f') ;
            particles.maps(j).weights = C{1} ;
            particles.maps(j).means = [C{2} C{3}]' ;
            particles.maps(j).covs = [C{4} C{5} C{6} C{7}]' ;
            particles.maps(j).covs = reshape(particles.maps(j).covs,2,2,[]) ;
        end
        j = j+1 ;
    end
    fclose(fid) ;
end