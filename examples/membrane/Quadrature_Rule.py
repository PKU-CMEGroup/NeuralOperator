import numpy as np

def gaussian_quad(nPoints):
    '''
    % Description:
    %   This function contains a library of 1D Gaussian Quadrature
    %   rules to be used in approximating the local stiffness matrix
    %   and load vectors.
    %
    % Outputs:
    %   q    <nPoints x 1 double>   List of sample points
    %   w    <nPoints x 1 double>   List of sample weights
    '''

    # Set quadrature rule
    if nPoints == 1:
        q = np.array([0])
        w = np.array([2])

    elif  nPoints == 2:
        q = np.sqrt(1./3.)*np.array([ -1., 1. ])
        w = np.array([1, 1])

    elif nPoints == 3:
        q = np.sqrt(3./5.)*np.array([ 0., -1., 1.])
        w = 1./9.*np.array([ 8., 5., 5.])

    else:
        print('Unsupported Gaussian Quadrature rule requested')

    return q, w

