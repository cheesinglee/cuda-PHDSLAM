#include <float.h>

typedef struct{
    int r ;
    int c ;
    node* next ;
} node ;

/// Modified Munkres Assignment Algorithm
/**
  Given an mxn cost matrix, find the optimal assignment. The result is written
  to an integer array of m elements
  **/
__device__ void
munkres_assign(double* C, int rows, int cols, bool* starred,bool* primed,
               bool* row_covered, bool* col_covered, int* result)
{
    // initialize variables
    for ( int n = 0 ; n < cols ; n++ )
    {
        col_covered[n] = false ;
        for ( m = 0 ; m < rows ; m++ )
        {
            result[m] = -1 ;
            row_covered[m] = false ;
            starred[n*rows+m] = false ;
            primed[n*rows+m] = false ;
        }
    }
    int k = 0 ;
    int Z0r = -1 ;
    int Z0c = -1 ;

    // preliminaries - with each column or row (whichever is larger), subtract
    // the minimum entry from all entries
    if ( rows >= cols )
    {
        for ( int n = 0 ; n < cols ; n++ )
        {
            double min_val = DBL_MAX ;
            for ( int m = 0 ; m < rows ; m++ )
            {
                if ( C[n*rows+m] < min_val)
                    min_val = C[n*rows+m] ;
            }

            for ( int m = 0 ; m < rows ; m++ )
            {
                C[n*rows+m] -= min_val ;
            }
        }
        k = cols ;
    }
    else
    {
        for ( int m = 0 ; m < rows ; m++ )
        {
            double min_val = DBL_MAX ;
            for ( int n = 0 ; n < cols ; n++ )
            {
                if ( C[n*rows+m] < min_val)
                    min_val = C[n*rows+m] ;
            }

            for ( int m = 0 ; m < rows ; m++ )
            {
                C[n*rows+m] -= min_val ;
            }
        }
        k = rows ;
    }

    ////////////////////////////////////////////////////////////////////
    //
    // Main loop
    //
    /////////////////////////////////////////////////////////////////////
    int step = 1 ;
    while ( step < 6 )
    {

        if ( step == 1 )
            // find an uncovered zero in C, and star it if there are no other
            // starred zeros in its row or column
        {
            for ( int n = 0 ; n < cols ; n++ )
            {
                if ( col_covered[n] )
                    continue ;
                for ( int m = 0 ; m < rows ; m++ )
                {
                    if ( row_covered[n] )
                        continue ;
                    if( C[n*rows+m] == 0 )
                    {
                        bool found_star = false ;
                        // look for a starred zero in row m
                        for ( int j = 0 ; j < cols ; j++ )
                        {
                            if ( starred[j*rows+m] )
                            {
                                found_star = true ;
                                break ;
                            }
                        }
                        if ( found_star )
                            continue ;
                        else
                        {
                            // look for a starred zero in column n
                            for ( int i = 0 ; i < rows ; i++ )
                            {
                                if ( starred[n*rows+i] )
                                {
                                    found_star = true ;
                                    break ;
                                }
                            }
                        }
                        // star the zero and stop searching this column
                        if ( !found_star )
                        {
                            starred[n*rows+m] = true ;
                            break ;
                        }
                    }
                }
            }
        }
        else if( step == 2 )
            // cover every column containing a 0*. If k columns are covered,
            // then we are done. Otherwise, go to step 3.
        {
            int n_covered = 0 ;
            for ( int n = 0 ; n < cols ; n++ )
            {
                double* col_ptr = starred+n*rows ;
                for ( int m = 0 ; m < rows ; m++ )
                {
                    if (col_ptr[m] == 0)
                    {
                        col_covered[n] = true ;
                        n_covered++ ;
                        break ;
                    }
                }
            }
            if ( n_covered == k )
                step = 6 ;
            else
                step = 3 ;
        }
        else if( step == 3 )
            // Find an uncovered zero and prime it. If there is no 0* in its
            // row, go to step 4. Otherwise, cover this row, and uncover the
            // column containing the 0*. Continue until there are no uncovered
            // zeros and go to step 5.
        {
            bool done = false ;
            while ( !done )
            {
                // look for an uncovered zero
                int i = -1 ;
                int j = -1 ;
                for ( int n = 0 ; n < cols ; n++ )
                {
                    if ( col_covered[n] )
                        continue ;
                    for ( int m = 0 ; m < rows ; m++ )
                    {
                        if ( row_covered[m] )
                            continue ;
                        if ( C[n*rows+m] == 0 )
                        {
                            primed[n*rows+m] = true ;
                            i = m ;
                            j = n ;
                            break ;
                        }
                    }
                    if ( i >= 0 )
                        break ;
                }
                if ( i >= 0 )
                {
                    int starred_col = -1 ;
                    for ( int jj = 0 ; jj < cols ; jj++ )
                    {
                        if ( starred[jj*rows+i] )
                        {
                            starred_col = jj ;
                            break ;
                        }
                    }
                    if ( starred_col >= 0  )
                    {
                        row_covered[i] = true ;
                        col_covered[starred_col] = false;
                    }
                    else
                    {
                        Z0r = i ;
                        Z0c = j ;
                        step = 4 ;
                        done = true ;
                    }
                }
                else
                {
                    step = 5 ;
                    done = true ;
                }
            }
        }
        else if( step == 4 )
            // Let Z0 = the uncovered 0' from step 3, then Z1 is the 0* in the
            // same column of Z0 (if there is one), and Z2 is the 0' in the same
            // row as Z1 (there will always be one). Continue building this
            // sequence up to the 0' with no 0* in its column. Unstar each 0*
            // in the sequence, and star each 0'. Erase all primes and uncover
            // every line, then go to step 2.
        {
            node* head = (node*)malloc( sizeof(node) ) ;
            head->r = Z0r ;
            head->c = Z0r ;
            node* tail = head ;
            bool done = false;
            int i = -1 ;
            int j = -1 ;
            while (true)
            {
                j = tail->c ;
                // look for a star in the same column
                int star_row = -1 ;
                for ( i = 0 ; i < rows ; i++ )
                {
                    if ( starred[j*rows+i] )
                    {
                        star_row = i ;
                        break ;
                    }
                }
                if ( star_row == -1 )
                {
                    // if no star, stop building the list
                    break ;
                }
                else
                {
                    // add a node to the list
                    node* Z = (node*)malloc(sizeof(node)) ;
                    Z->r = star_row ;
                    Z->c = j ;
                    tail->next = Z ;
                    tail = Z ;

                    // find the prime in the same row as the star
                    i = star_row ;
                    int prime_col = -1 ;
                    for ( j = 0 ; j < cols ; j++ )
                    {
                        if ( primed[j*rows+i] )
                        {
                            prime_col = j ;
                            break ;
                        }
                    }

                    // add a node to the list
                    Z = (node*)malloc(sizeof(node)) ;
                    Z->r = i ;
                    Z->c = primed_col ;
                    tail = Z ;
                }
            }

            // traverse the list, star the primes, unstar the stars, and
            // destroy the nodes
            node* Z = head ;
            bool prime = true ;
            while (Z != tail )
            {
                i = Z->r ;
                j = Z->c ;
                if ( prime )
                {
                    starred[j*rows+i] = true ;
                    prime = false ;
                }
                else
                {
                    starred[j*rows+i] = false ;
                    prime = true ;
                }
                node* tmp = Z ;
                Z = Z->next ;
                free(tmp) ;
            }
            i = Z->r ;
            j = Z->c ;
            starred[j*rows+i] = true ;

            // remove all primes and uncover all lines
            i = 0 ;
            j = 0 ;
            for ( int n = 0 ; n < rows*cols ; n++ )
            {
                primed[n] = false ;
                row_covered[i] = false ;
                i++ ;
                if ( i == rows )
                {
                    i = 0 ;
                    j++ ;
                    col_covered[j] = false ;
                }
            }

            // go to step 2
            step = 2 ;
        }
        else if( step == 5 )
            // Let h be the smallest uncovered value in C. Add h to every
            // covered row, then subract h from every uncovered column.
            // Go to step 3.
        {
            // find the minimum uncovered entry of C
            double h = DBL_MAX ;
            int i = 0 ;
            int j = 0 ;
            for ( int n = 0 ; n < rows*cols ; n++ )
            {
                if ( !row_covered[i] && !col_covered[j] )
                    h = fmin(h,C[n]) ;
                i++ ;
                if ( i == rows )
                {
                    i = 0 ;
                    j++ ;
                }
            }

            // add h to every covered row, and subtract from every uncovered col
            i = 0 ;
            j = 0 ;
            for ( int n = 0 ; n < rows*cols ; n++ )
            {
                if ( row_covered[i] && col_covered[j] )
                    C[n] += h ;
                else if ( !row_covered[i] && !col_covered[j] )
                    C[n] -= h ;
                if ( i == rows )
                {
                    i = 0 ;
                    j++ ;
                }
            }
            step = 3 ;
        }
    }

    // construct the optimum assignment from the starred zeros

    for ( int i = 0 ; i < rows ; i++ )
    {
        int star_col = -1 ;
        for ( int j = 0 ; j < cols ; j++ )
        {
            if ( starred[j*rows+i] )
            {
                star_col = j ;
            }
        }
        result[i] = star_col ;
    }
}


