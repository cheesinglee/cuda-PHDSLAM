# Cython implementation of the Munkres/Hungarian assignment algorithm
#
# Author: Chee Sing Lee <cheesinglee@gmail.com>

import numpy
import sys

cdef extern from "math.h":
    double sqrt(double x)
    double pow(double x, double e)

cdef extern from "/usr/lib/pymodules/python2.7/numpy/core/include/numpy/arrayobject.h":

    ctypedef int intp 

    ctypedef extern class numpy.dtype [object PyArray_Descr]:
        cdef int type_num, elsize, alignment
        cdef char type, kind, byteorder, hasobject
        cdef object fields, typeobj

    ctypedef extern class numpy.ndarray [object PyArrayObject]:
        cdef char *data
        cdef int nd
        cdef intp *dimensions
        cdef intp *strides
        cdef object base
        cdef dtype descr
        cdef int flags

    void import_array()

import_array()

def step1_cython(ndarray C):
    """ 
    for each row or col in the matrix (whichever is larger), find the 
    minimum value and subtract it from the row
    """
    cdef double* c_ptr = <double*>C.data
    cdef double* row_ptr
    cdef double* col_ptr
    cdef int rows = C.dimensions[0]
    cdef int cols = C.dimensions[1]
    cdef int i
    cdef int j
    cdef minval
    if cols > rows: 
        for i from 0 <= i < rows:
            row_ptr = c_ptr + i*cols
            minval = 999999999
            for j from 0 <= j < cols:
                minval = min(minval,row_ptr[j])
            for j from 0 <= j < cols:
                row_ptr[j] -= minval
    else:
        for j from 0 <= j < cols:
            col_ptr = c_ptr + j
            minval = 99999999
            for i from 0 <= i < rows:
                minval = min(minval,col_ptr[i*cols])
            for i from 0 <= i < rows:
                col_ptr[i*cols] -= minval
        

def step2_cython(ndarray C, ndarray starred, ndarray row_covered, ndarray col_covered):
    """
    for each uncovered zero in C, star it if there are no other starred zeros
    in its row or column
    """
    cdef int i
    cdef int j  
    cdef int ii
    cdef int jj
    cdef int n
    cdef int K
    cdef char found_other_star
    cdef double* c_ptr
    cdef char* rc_ptr
    cdef char* cc_ptr
    cdef char* star_ptr
    cdef char* row_ptr
    cdef char* col_ptr
    cdef int rows = C.dimensions[0]
    cdef int cols = C.dimensions[1]
    c_ptr = <double*>C.data
    rc_ptr = <char*>row_covered.data    
    cc_ptr = <char*>col_covered.data    
    star_ptr = <char*>starred.data
    K = C.dimensions[0]
    
    i = 0
    j = 0
    for n from 0 <= n < rows*cols:
        if c_ptr[n] == 0 and not rc_ptr[i] and not cc_ptr[j]:
            found_other_star = 0
            # look for a star in this row
            row_ptr = star_ptr + i*cols
            for jj from 0 <= jj < cols:
                if row_ptr[jj]:
                    found_other_star = 1
                    break
            if not found_other_star:
                # look fora star in this column
                col_ptr = star_ptr + j
                for ii from 0 <= ii < rows:
                    if col_ptr[ii*cols]:
                        found_other_star = 1
                        break
            if not found_other_star:
                star_ptr[i*cols+j] = 1
        j += 1
        if j == cols:
            i += 1
            j = 0
            
def step3_cython(ndarray starred, ndarray col_covered):
    """
    Cover each column containing a starred zero. If K columns are
    covered, the starred zeros describe a complete set of unique
    assignments. In this case, Go to DONE, otherwise, Go to Step 4.
    """
    cdef char* cc_ptr = <char*>col_covered.data
    cdef char* star_ptr = <char*>starred.data
    cdef char* col_ptr
    cdef int i = 0
    cdef int j = 0
    cdef int rows = starred.dimensions[0]
    cdef int cols = starred.dimensions[1]
    
    # cover columns containing starred zeros
    for j from 0 <= j < cols:
        col_ptr = star_ptr + j
        for i from 0 <= i < rows:
            if col_ptr[i*cols]:
                cc_ptr[j] = 1
                break
    
    # check number of covered columns
    cdef int n_covered = 0
    for j from 0 <= j < cols:
        if cc_ptr[j]:
            n_covered += 1
            
    # figure out where to go next
    next_step = -1
    if n_covered >= min(rows,cols):
        next_step = 7
    else:
        next_step = 4
    return next_step

def step4_cython(ndarray C, ndarray row_covered, ndarray col_covered, 
                ndarray starred, ndarray primed):
#    if chr(zeros.descr.type) <> "b":
#        raise TypeError("char array required")
#    if zeros.nd <> 2:
#        raise ValueError("2 dimensional array required")
    cdef int rows
    cdef int cols
    cdef double* z_ptr
    cdef char* rc_ptr
    cdef char* cc_ptr
    cdef char* star_ptr
    cdef char* prime_ptr
    c_ptr = <double*>C.data
    rc_ptr = <char*>row_covered.data    
    cc_ptr = <char*>col_covered.data    
    star_ptr = <char*>starred.data    
    prime_ptr = <char*>primed.data    
    rows = C.dimensions[0]
    cols = C.dimensions[1]

    cdef int i
    cdef int j
    cdef int ii
    cdef int jj
    cdef int star_col
    cdef double* row_ptr
    cdef char* row_ptr2
    cdef char done
    cdef char found_zero
    done = 0
    Z0r = -1
    Z0c = -1
    next_step = -1
    while not done:
        # look for an uncovered zero
        found_zero = 0    
        ii = -1
        jj = -1
        i = -1
        j = -1
        while not found_zero:
            i += 1
            if i == rows:
                break
            if not rc_ptr[i]:
                row_ptr = c_ptr + i*cols
                for j from 0 <= j < cols:
                    if row_ptr[j]==0 and not cc_ptr[j]:
                        ii = i
                        jj = j
                        found_zero = 1
                        break
        if ii < 0:
            # did not find any uncovered zeros
            done = 1
            next_step = 6
        else:
            # prime the zero
            prime_ptr[ii*cols+jj] = 1
            # look for a starred zero in the same row
            star_col = -1
            row_ptr2 = star_ptr + ii*cols
            for j from 0 <= j < cols:
                if row_ptr2[j]:
                    star_col = j
                    break
            if star_col < 0:
                # did not find a starred zero
                done = 1
                Z0r = ii
                Z0c = jj
                next_step = 5
            else:
                # cover this row, and uncover the starred zero's column
                rc_ptr[ii] = 1
                cc_ptr[star_col] = 0
    return (next_step,Z0r,Z0c)
   
def step5_cython(ndarray starred, ndarray primed, 
                 ndarray row_covered, ndarray col_covered,
                 int Z0r, int Z0c):
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
        cdef int rows = starred.dimensions[0]
        cdef int cols = starred.dimensions[1]
        cdef char* star_ptr = <char*>starred.data
        cdef char* prime_ptr = <char*>primed.data
        cdef char* rc_ptr = <char*>row_covered.data
        cdef char* cc_ptr = <char*>col_covered.data
        cdef int star_row
        cdef int prime_col
        cdef char* row_ptr
        cdef char* col_ptr
        cdef int i
        cdef int j
        cdef int last_row
        cdef int last_col
        path = [[Z0r,Z0c]]
        cdef char done = 0
        while not done:
            # look for a star in the same column
            last_col = path[-1][1]
            col_ptr = star_ptr + last_col
            star_row = -1
            for i from 0 <= i < rows:
                if col_ptr[i*cols]:
                    star_row = i
                    break
            if star_row >= 0:
                path.append([star_row,last_col])
            else:
                done = 1
            
            # look for a prime in the same row
            if done == 0:
                last_row = path[-1][0]
                row_ptr = prime_ptr + last_row*cols
                for j from 0 <= j < cols:
                    if row_ptr[j]:
                        prime_col = j
                        break
                path.append([last_row,prime_col])
        
        # unstar starred zeros and star primed zeros
        cdef char prime_or_star = 0
        for [i,j] in path:
            if prime_or_star == 0:
                star_ptr[i*cols + j] = 1 
                prime_or_star = 1
            elif prime_or_star == 1:
                star_ptr[i*cols + j] = 0
                prime_or_star = 0
                
        # erase all primes
        for i from 0 <= i < rows*cols:
            prime_ptr[i] = 0
        
        # uncover all lines
        for i from 0 <= i < rows:
            rc_ptr[i] = 0        
        for j from 0 <= j < cols:
            cc_ptr[j] = 0        
            
    
def step6_cython(ndarray C, ndarray row_covered, ndarray col_covered):
    """
    Add the value found in Step 4 to every element of each covered
    row, and subtract it from every element of each uncovered column.
    Return to Step 4 without altering any stars, primes, or covered
    lines.
    """
    cdef int i
    cdef int j    
    cdef int n
    cdef int rows
    cdef int cols
    cdef double* c_ptr
    cdef char* rc_ptr
    cdef char* cc_ptr
    cdef double minval
    c_ptr = <double*>C.data
    rc_ptr = <char*>row_covered.data    
    cc_ptr = <char*>col_covered.data    
    rows = C.dimensions[0]
    cols = C.dimensions[1]
    
    # compute the minimum uncovered value
    minval = 99999999999999   
    i = 0
    j = 0
    for n from 0 <= n < rows*cols:
        if not (rc_ptr[i] or cc_ptr[j]):
            minval = min(minval,c_ptr[n])
        j = j + 1
        if j == cols:
            i = i+1
            j = 0
            
    # subtract the minimum value from each uncovered column and add
    # it to each covered row
    i = 0
    j = 0
    for n from 0 <= n < rows*cols:
        if rc_ptr[i] and cc_ptr[j]:
            c_ptr[n] += minval
        elif not (rc_ptr[i] or cc_ptr[j]):
            c_ptr[n] -= minval
        j += 1
        if j >= cols:
            i += 1
            j = 0

def compute_cost(ndarray X, ndarray Y, ndarray C):
    cdef int m = X.dimensions[0]
    cdef int n = Y.dimensions[0]
    cdef int dims = X.dimensions[1]
    cdef int i
    cdef int j
    cdef int k
    cdef double* x_ptr = <double*>X.data
    cdef double* y_ptr = <double*>Y.data
    cdef double* c_ptr = <double*>C.data
    cdef double dist
    
    # check for 1D data
    if X.nd == 1:
        dims = 1
        m = X.dimensions[0]
        n = Y.dimensions[0]
    
    for i from 0 <= i < m:
        for j from 0 <= j < n:
            dist = 0
            for k from 0 <= k < dims:
                dist += pow( (x_ptr[i*dims+k] - y_ptr[j*dims+k]), 2 )
            c_ptr[i*n+j] = sqrt(dist)