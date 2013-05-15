# -*- coding: utf-8 -*-

from numpy import *

import pyximport ; pyximport.install()
from munkres_step4 import step1_cython, step2_cython, step3_cython 
from munkres_step4 import step4_cython, step5_cython, step6_cython
from munkres_step4 import compute_cost


class Munkres2:
    starred = array([])
    primed = array([])
    col_covered = array([])
    row_covered = array([])
    zero_idx = array([])
    next_step = 1
    C = None
    path = None
    Z0_r = None
    Z0_c = None    
    def __step1(self):
        """ 
        for each row or col in the matrix (whichever is larger), find the 
        minimum value and subtract it from the row
        """
#        for i in xrange(self.C.shape[0]):
#            self.C[i,:] -= min(self.C[i,:])
#        self.next_step = 2
#        return
        step1_cython(self.C)
        self.next_step = 2
    
    def __step2(self):
        """
        for each zero in the matrix, star it if there are no other starred zeros
        in its row or column
        """
#        C = self.C
#        zero_mask = (C==0)
#        zero_mask[:,self.col_covered] = False
#        zero_mask[self.row_covered,:] = False
#        done = False
#        while not done:
#            idx = zero_mask.nonzero()
#            idx = vstack(idx).transpose()    
#            if idx.size == 0:
#                done = True
#            else:
#                [i,j] = idx[0,:]
#                self.starred[i,j] = True
#                zero_mask[i,:] = False
#                zero_mask[:,j] = False
        step2_cython(self.C,self.starred,self.row_covered,self.col_covered)
        self.next_step = 3
        return
    
    def __step3(self):
        """
        Cover each column containing a starred zero. If K columns are
        covered, the starred zeros describe a complete set of unique
        assignments. In this case, Go to DONE, otherwise, Go to Step 4.
        """
#        for j in xrange(self.cols):
#            if any(self.starred[:,j]):
#                self.col_covered[j] = True
#        if count_nonzero(self.col_covered) ==  self.K:
#            self.next_step = 7
#        else:
#            self.next_step = 4
        self.next_step = step3_cython(self.starred,self.col_covered)
    
    def __step4(self):
        """
        Find a noncovered zero and prime it. If there is no starred zero
        in the row containing this primed zero, Go to Step 5. Otherwise,
        cover this row and uncover the column containing the starred
        zero. Continue in this manner until there are no uncovered zeros
        left. Save the smallest uncovered value and Go to Step 6.
        """
#        zero_mask = (self.C == 0)
        results = step4_cython(self.C,self.row_covered,self.col_covered,
                               self.starred,self.primed)
        self.next_step = results[0]
        self.Z0_r = results[1]
        self.Z0_c= results[2]
                    
    def __step5(self):
        """
        Construct a series of alternating primed and starred zeros as
        follows. Let Z0 represent the uncovered primed zero found in Step 4.
        Let Z1 denote the starred zero in the column of Z0 (if any).
        Let Z2 denote the primed zero in the row of Z1 (there will always
        be one). Continue until the series terminates at a primed zero
        that has no starred zero in its column. Unstar each starred zero
        of the series, star each primed zero of the series, erase all
        primes and uncover every line in the matrix. Return to Step 3
        """
        
#        path = self.path
#        path.append([self.Z0_r,self.Z0_c])
#        done = False
#        while not done:
#            row = self.__find_star_in_col(path[-1][1])
#            if row >= 0:
#                path.append([row,path[-1][1]])
#            else:
#                done = True
#        
#            if not done:
#                col = self.__find_prime_in_row(path[-1][0])
#                path.append([path[-1][0],col])
#                
#        # convert primes to stars and unstar stars
#        for n in xrange(len(path)):
#            [i,j] = path[n]
#            if mod(n,2) == 0:   # prime -> star
#                self.starred[i,j] = True
#            else:               # star -> unstar
#                self.starred[i,j] = False
#
#        self.row_covered[:] = False
#        self.col_covered[:] = False
#        self.primed[:] = False

        step5_cython(self.starred,self.primed,self.row_covered,
                     self.col_covered,self.Z0_r,self.Z0_c)
        self.next_step = 3 
        return
        
    def __step6(self):
        """
        Add the value found in Step 4 to every element of each covered
        row, and subtract it from every element of each uncovered column.
        Return to Step 4 without altering any stars, primes, or covered
        lines.
        """
#        uncovered_mask = ones_like(self.C,dtype=bool)
#        uncovered_mask[self.row_covered,:] = False
#        uncovered_mask[:,self.col_covered] = False
#        uncovered_vals = self.C[uncovered_mask]
#        minval = amin(uncovered_vals)
#        self.C[self.row_covered,:] += minval
#        self.C[:,logical_not(self.col_covered)] -= minval
        step6_cython(self.C,self.row_covered,self.col_covered)
        self.next_step = 4
        
#    def __find_prime_in_row(self, row):
#        """
#        Find the first prime element in the specified row. Returns
#        the column index, or -1 if no starred element was found.
#        """
#        col = -1
#        primes = argwhere( self.primed[row,:] )
#        if primes.size > 0:
#            col = primes[0][0]
#        return col
#    
#    def __find_star_in_row(self, row):
#        """
#        Find the first starred element in the specified row. Returns
#        the column index, or -1 if no starred element was found.
#        """
#        col = -1
#        stars = argwhere( self.starred[row,:] )
#        if stars.size > 0:
#            col = stars[0][0]
#        return col
#        
#    def __find_star_in_col(self, col):
#        """
#        Find the first starred element in the specified row. Returns
#        the column index, or -1 if no starred element was found.
#        """
#        row = -1
#        stars = argwhere(self.starred[:,col])
#        if stars.size > 0:
#            row = stars[0][0]
#        return row
    
    def compute(self,C):
        self.C = copy(C)
        self.rows = size(C,0)
        self.cols = size(C,1)
        self.K = min( self.rows, self.cols )
        self.row_covered = zeros(self.rows)
        self.col_covered = zeros(self.cols)

        self.starred = zeros_like(C)
        self.primed = zeros_like(C)
        self.path = []
        self.next_step = 1
        
#        # make rectangular matrices square
#        [m,n] = self.C.shape
#        if m < n:
#            self.C = vstack( (self.C, zeros([n-m,n]) ) )
#        elif m > n:
#            self.C = hstack( (self.C, zeros([m,m-n]) ) )
        
        while self.next_step < 7:
            if self.next_step == 1:
                self.__step1()
            elif self.next_step == 2: 
                self.__step2()
            elif self.next_step == 3: 
                self.__step3()
            elif self.next_step == 4: 
                self.__step4()
            elif self.next_step == 5: 
                self.__step5()
            elif self.next_step == 6: 
                self.__step6()
                    
        # get the solution from starred zeros
        results = nonzero(self.starred)
        results = vstack(results).transpose()
        return results
        

def ospa_distance(X,Y,p=1,c=10):
    """ Compute the OSPA metric between two sets of points. """
    
    # check for empty sets
    if size(X) == 0 and size(Y) == 0:
        return (0,0,0)
    elif size(X) == 0 or size(Y) == 0 :
        return (c,0,c)
        

    # we assume that Y is the larger set
    m = size(X,0) 
    n = size(Y,0)
    if m > n:
        X,Y = Y,X
        m,n = n,m        
    
    # compute the cost matrix using Euclidean distance
    dists = empty([m,n])
    compute_cost(X,Y,dists)
#    i = 0
#    for x in X:
#        diff = x - Y 
#        if diff.ndim > 1:
#            dist_row = sqrt(sum(diff**2,axis=1))
#        else:
#            dist_row = sqrt(diff**2)
#        dist_row[ dist_row > c ] = c    # apply the cutoff        
#        dists[i,:] = dist_row 
#        i += 1
        
#    # pad the matrix with dummy points
#    if n > m:
#        dists[m:n,:] = c
    
    # compute the optimal assignment using the Hungarian (Munkres) algorithm
    assignment = Munkres2()
    indices = assignment.compute(dists)
    
    # compute the OSPA metric
    total = 0 
    total_loc = 0
    for [i,j] in indices:
        total_loc += dists[i][j]**p
    err_cn = (float(c**p*(n-m))/n)**(1/p)
    err_loc = (float(total_loc)/n)**(1/p) 
    ospa_err = ( float(total_loc + (n-m)*c**p) / n)**(1/p)
    ospa_tuple = (ospa_err,err_loc,err_cn) 
    return ospa_tuple
    
if __name__ == '__main__':
    # test routine
    X = arange(6,dtype='float')
    Y = array([0,-3,-6],dtype='float')
    d = ospa_distance(X,Y)
    print(d)
    
    
    
            
        
        
