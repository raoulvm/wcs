from IPython.core.display import Markdown, display
import numpy as np


def printmd(string:str):
    '''
    Markdown printout in Jupyter
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    prints a ``string`` that contains markdown (or more precisely, can contain markdown) in Jupyter cell output
    '''
    display(Markdown(string))

def print_matrix(matrix:np.ndarray, decimals:int=None, name:str=None, maxSize:int=20):
    '''
    Matrix Markdown in Jupyter
    ==========================

    prints a ``matrix:numpy.ndarray`` as Matrix in Jupyter cell output  \
        
    ``decimals`` decimal places for each element. None= use unformatted output.  \
    ``name`` can be specified to print name= before the matrix  
    ``maxSize`` can be specified to limit the columns/rows printed to maxSize elements. More rows/columns will be skipped indicated with ...  
      
    List use: Both ``matrix`` and ``name`` can be a python list, preferably wit hthe same numbers of elements, to print multiple matrices at once.
    
    Raises:
    ~~~~~~~
    *  Exception if the matrix to print has more than two dimensions!  
    *  TypeError if the matrix is not an numpy.ndarray  
    '''
    def oneMatrix(matr, name):
        if isinstance(matr, np.ndarray):
            if name!=None:
                mdheader = f'$ {name} = \\begin{{bmatrix}}'
            else:    
                mdheader = '$ \\begin{bmatrix}'
            mstr = ''
            if len(matr.shape)==1:
                matr = matr.reshape(matr.shape[0],1)
            if len(matr.shape)>2:
                raise Exception('cannot print more than two dimensions on a flat screen')
            cskipat = matr.shape[1]+1
            cskipto = matr.shape[1]+1
            rskipat = matr.shape[0]+1
            rskipto = matr.shape[0]+1
            if matr.shape[1]>maxSize:
                cskipat = maxSize-2
                cskipto = matr.shape[1]-1
            if matr.shape[0]>maxSize:
                rskipat = maxSize-2
                rskipto = matr.shape[0]-1
            rskip=False
            for row in range(matr.shape[0]):

                if row>=rskipat and row<rskipto:
                    if not rskip:
                        # row to skip
                        if cskipat!=cskipto: # there are columns to skip, too: Use diagonal dots
                           mstr += "\\vdots & " * (maxSize-2) + ' \\ddots & \\vdots \\\\ '
                        else:
                            mstr += "\\vdots & " * (min(maxSize, matr.shape[1])-1) + ' \\vdots \\\\ '
                        rskip=True
                else:
                    cskip=False
                    for col in range(matr.shape[1]):
                        # debug {
                        #mstr += '[' + str(col) +'<=>' + str(cskipat) + ']'
                        #debug }
                        if col>=cskipat and col<cskipto:
                            # column to skip
                            if not cskip:
                                mstr += "\cdots & "
                                cskip = True
                        else:
                            if decimals!=None:
                                mstr += "{{:.{}f}}".format(decimals).format(matr[row][col])
                            else:
                                mstr += str(round(matr[row][col],15))
                            if col<matr.shape[1]-1:
                                mstr += ' & '
                    if row<matr.shape[0]-1:
                        mstr += ' \\\\ '
            mdfooter = f' \end{{bmatrix}}_{{Shape{matr.shape}}} $'
            return mdheader+mstr+mdfooter
        else:
            # return (type(matr) + ' is not supported')
            raise TypeError('Wrong type of matrix: only numpy.ndarray is supported.')
            
    if isinstance(matrix, list):
        coll = ''
        if isinstance(name, list):
            if len(name)<len(matrix):
                name = name + [None]*(len(matrix)-len(name))
        else:
            name = [name] + [None]*(len(matrix)-1)
        for m,n in zip(matrix,name):
            coll += oneMatrix(m,n)
    else:
        coll = oneMatrix(matrix,name)
    printmd(coll)
    #print(coll)    
        
def matrixInfo(matrix:np.ndarray, name:str='A', verbose:bool=False, decimals:int=None, maxSize:int=20, surfaceGraph=False):
    '''
    Matrix quick analysis in Jupyter
    ================================
    Prints some short analysis of the matrix passed, such as determinant, eingenvectors and -values, inverse

    ``decimals`` decimal places for each element. None= use unformatted output.  
    ``name`` can be specified to print name= before the matrix  
    ``maxSize`` can be specified to limit the columns/rows printed to maxSize elements. More rows/columns will be skipped indicated with ...  
    ``verbose`` True will print more hints to the analyses, e.g. Wikipedia links.
    '''
    if len(matrix.shape) in [1,2]:
        printmd(f'## Overview for the {len(matrix.shape)}-dimensional matrix {name}')
        printMatrix(matrix, name=name, decimals=decimals)
    else:
        printmd(f'## Overview for the {len(matrix.shape)}-dimensional matrix {name}')
    eigval, eigvec = np.linalg.eig(matrix)
    printmd('### Eigenvalues and corresponding eigenvectors')
    if verbose:
        printmd('Eigenvectors are the vectors (different from the nullvector) that are only _scaled_ by a transformation matrix operation _but not rotated_.  ')
        printmd('The eigenvalues are the measure of scaling. Eigenvectors by numpy are normalized in length.  ')
        printmd('There might not be a solution in real space, so the eigenvectors and eigenvalues can be complex vectors and numbers respectively.  ')
        printmd('[Wikipedia link.](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors)  ')
    printMatrix([eigvec[:,x] for x in range(eigvec.shape[0])], name=['v_{{{}}}'.format(x) for x in list(eigval)], decimals=decimals, maxSize=maxSize)
    printmd('## Euclidian Norm (2nd)')
    if verbose:
        # https://en.wikipedia.org/wiki/Matrix_norm
        printmd('[Wikipedia link.](https://en.wikipedia.org/wiki/Matrix_norm)  ')
    printmd(f'$ {{\|{name}\|_2}} =' + str(np.linalg.norm(matrix))+'$')
    printmd('### Determinant')
    if verbose:
        # https://en.wikipedia.org/wiki/Matrix_norm
        printmd('[Wikipedia link.](https://en.wikipedia.org/wiki/Determinant)  ')
    printmd(f'${{det}}_{{{name}}} = $' + str(np.linalg.det(matrix)))
    printmd('### Rank')
    if verbose: #
        printmd('[Wikipedia link](https://en.wikipedia.org/wiki/Rank_(linear_algebra))')
    r = np.linalg.matrix_rank(matrix)
    printmd(f'$rank({name}) = $' + str(r))
    if r==min(matrix.shape):
        printmd('Matrix is FULL RANK')
    printmd('### Inverse')
    try:
        i = np.linalg.inv(matrix)
        printMatrix(i, name= f'{{{name}}}^{{-1}}', decimals=decimals, maxSize=maxSize)
    except Exception as exc:
        printmd('_there is no inverse to that matrix, or at least it could not be computed._')
        print(exc)
    if surfaceGraph and len(matrix.shape)==2:
        import plotly.graph_objects as go
        fig = go.Figure(go.Surface(
            contours = {
                "z": {"show": True, "start": np.mean(matrix.flatten())-np.std(matrix.flatten()), "end": np.mean(matrix.flatten())+np.std(matrix.flatten())*1.01, "size": np.std(matrix.flatten())}
                },
            x = list(range(matrix.shape[0])),
            y = list(range(matrix.shape[1])),
            z = matrix))
        fig.layout.title.text = "Surface approximation (with +/- one std deviation markers"
        fig.update_layout(xaxis_title = 'column',
            yaxis_title='row')
        fig.show()

    
  