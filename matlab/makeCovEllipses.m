function pp = makeCovEllipses( means, covs, N )
%MAKECOVELLIPSES Summary of this function goes here
%   Detailed explanation goes here
    phi = 0:2*pi/N:2*pi ;
    circ = 3*[cos(phi); sin(phi)] ;
    count = 1 ;
    nFeatures = size(means,2) ;
    pp = zeros(2, nFeatures*(N+2)) ;
    for j = 1:nFeatures
        m = means(:,j) ;
        P = covs(:,:,j) ;
        r = sqrtm(P) ;
        a = r*circ ;
        pp(1,(count:(count+N+1))) = [a(1,:) + m(1) NaN] ;
        pp(2,(count:(count+N+1))) = [a(2,:) + m(2) NaN] ;
        count = count + N+2 ;
    end
end

