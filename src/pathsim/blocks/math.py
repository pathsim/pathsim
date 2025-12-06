#########################################################################################
##
##                                      MATH BLOCKS 
##                                    (blocks/math.py)
##
##                  definitions of elementary math and function blocks
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._block import Block

from ..utils.register import Register
from ..optim.operator import Operator


# BASE MATH BLOCK =======================================================================

class Math(Block):
    """Base math block.

    Note
    ----
    This block doesnt implement any functionality itself. 
    Its intended to be used as a base for the elementary math blocks. 
    Its **not** intended to be used directly!

    """

    def __len__(self):
        """Purely algebraic block"""
        return 1


    def update(self, t):
        """update algebraic component of system equation 

        Parameters
        ----------
        t : float
            evaluation time
        """
        u = self.inputs.to_array()
        y = self.op_alg(u)
        self.outputs.update_from_array(y)


# BLOCKS ================================================================================

class Sin(Math):
    """Sine operator block.

    This block supports vector inputs. This is the operation it does:
        
    .. math::
        
        \\vec{y} = \\sin(\\vec{u}) 


    Attributes
    ----------
    op_alg : Operator
        internal algebraic operator
    """

    def __init__(self):
        super().__init__()

        #create internal algebraic operator
        self.op_alg = Operator(
            func=np.sin, 
            jac=lambda x: np.diag(np.cos(x))
            )


class Cos(Math):
    """Cosine operator block.

    This block supports vector inputs. This is the operation it does:
        
    .. math::
        
        \\vec{y} = \\cos(\\vec{u}) 
        
    Attributes
    ----------
    op_alg : Operator
        internal algebraic operator
    """

    def __init__(self):
        super().__init__()

        #create internal algebraic operator
        self.op_alg = Operator(
            func=np.cos, 
            jac=lambda x: -np.diag(np.sin(x))
            )


class Sqrt(Math):
    """Square root operator block.

    This block supports vector inputs. This is the operation it does:
        
    .. math::
        
        \\vec{y} = \\sqrt{|\\vec{u}|} 
        
    Attributes
    ----------
    op_alg : Operator
        internal algebraic operator
    """

    def __init__(self):
        super().__init__()

        #create internal algebraic operator
        self.op_alg = Operator(
            func=lambda x: np.sqrt(abs(x)), 
            jac=lambda x: np.diag(1/np.sqrt(abs(x)))
            )


class Abs(Math):
    """Absolute value operator block.

    This block supports vector inputs. This is the operation it does:
        
    .. math::
        
        \\vec{y} = \\vert| \\vec{u} \\vert| 
        
    Attributes
    ----------
    op_alg : Operator
        internal algebraic operator
    """

    def __init__(self):
        super().__init__()

        #create internal algebraic operator
        self.op_alg = Operator(
            func=lambda x: abs(x), 
            jac=lambda x: np.diag(np.sign(x))
            )


class Pow(Math):
    """Raise to power operator block.

    This block supports vector inputs. This is the operation it does:
        
    .. math::
        
        \\vec{y} = \\vec{u}^{p} 

    Parameters
    ----------
    exponent : float, array_like
        exponent to raise the input to the power of
        
    Attributes
    ----------
    op_alg : Operator
        internal algebraic operator
    """

    def __init__(self, exponent=2):
        super().__init__()

        self.exponent = exponent

        #create internal algebraic operator
        self.op_alg = Operator(
            func=lambda x: np.power(x, self.exponent), 
            jac=lambda x: np.diag(self.exponent * np.power(x, self.exponent - 1))
            )


class PowProd(Math):
    """Power-Product operator block.

    This block raises each input to a power and then multiplies all results together:
        
    .. math::
        
        y = \\prod_i u_i^{p_i}

    Parameters
    ----------
    exponents : float, array_like
        exponent(s) to raise the inputs to the power of. If scalar, 
        applies same exponent to all inputs.
        
    Attributes
    ----------
    op_alg : Operator
        internal algebraic operator
    """

    def __init__(self, exponents=2):
        super().__init__()

        self.exponents = exponents    
        
        def _jac(x):
            if np.isscalar(self.exponents):
                exps = np.full_like(x, self.exponents)
            else:
                exps = np.array(self.exponents)
            
            product = np.prod(np.power(x, exps))
            
            # Jacobian is a row vector since output is scalar
            jac = np.zeros((1, len(x)))
            for j in range(len(x)):
                if x[j] != 0:
                    jac[0, j] = product * exps[j] / x[j]
                else:
                    jac[0, j] = 0
            
            return jac

        #create internal algebraic operator
        self.op_alg = Operator(
            func=lambda x: np.prod(np.power(x, self.exponents)), 
            jac=_jac
            )


class Exp(Math):
    """Exponential operator block.

    This block supports vector inputs. This is the operation it does:
        
    .. math::
        
        \\vec{y} = e^{\\vec{u}} 
        
    Attributes
    ----------
    op_alg : Operator
        internal algebraic operator
    """

    def __init__(self):
        super().__init__()

        #create internal algebraic operator
        self.op_alg = Operator(
            func=np.exp, 
            jac=lambda x: np.diag(np.exp(x))
            )


class Log(Math):
    """Natural logarithm operator block.

    This block supports vector inputs. This is the operation it does:
        
    .. math::
        
        \\vec{y} = \\ln(\\vec{u}) 
        
    Attributes
    ----------
    op_alg : Operator
        internal algebraic operator
    """

    def __init__(self):
        super().__init__()

        #create internal algebraic operator
        self.op_alg = Operator(
            func=np.log, 
            jac=lambda x: np.diag(1/x)
            )


class Log10(Math):
    """Base-10 logarithm operator block.

    This block supports vector inputs. This is the operation it does:
        
    .. math::
        
        \\vec{y} = \\log_{10}(\\vec{u}) 
        
    Attributes
    ----------
    op_alg : Operator
        internal algebraic operator
    """

    def __init__(self):
        super().__init__()

        #create internal algebraic operator
        self.op_alg = Operator(
            func=np.log10, 
            jac=lambda x: np.diag(1/(x * np.log(10)))
            )


class Tan(Math):
    """Tangent operator block.

    This block supports vector inputs. This is the operation it does:
        
    .. math::
        
        \\vec{y} = \\tan(\\vec{u}) 
        
    Attributes
    ----------
    op_alg : Operator
        internal algebraic operator
    """

    def __init__(self):
        super().__init__()

        #create internal algebraic operator
        self.op_alg = Operator(
            func=np.tan, 
            jac=lambda x: np.diag(1/np.cos(x)**2)
            )


class Sinh(Math):
    """Hyperbolic sine operator block.

    This block supports vector inputs. This is the operation it does:
        
    .. math::
        
        \\vec{y} = \\sinh(\\vec{u}) 
        
    Attributes
    ----------
    op_alg : Operator
        internal algebraic operator
    """

    def __init__(self):
        super().__init__()

        #create internal algebraic operator
        self.op_alg = Operator(
            func=np.sinh, 
            jac=lambda x: np.diag(np.cosh(x))
            )


class Cosh(Math):
    """Hyperbolic cosine operator block.

    This block supports vector inputs. This is the operation it does:
        
    .. math::
        
        \\vec{y} = \\cosh(\\vec{u}) 
        
    Attributes
    ----------
    op_alg : Operator
        internal algebraic operator
    """

    def __init__(self):
        super().__init__()

        #create internal algebraic operator
        self.op_alg = Operator(
            func=np.cosh, 
            jac=lambda x: np.diag(np.sinh(x))
            )


class Tanh(Math):
    """Hyperbolic tangent operator block.

    This block supports vector inputs. This is the operation it does:
        
    .. math::
        
        \\vec{y} = \\tanh(\\vec{u}) 
        
    Attributes
    ----------
    op_alg : Operator
        internal algebraic operator
    """

    def __init__(self):
        super().__init__()

        #create internal algebraic operator
        self.op_alg = Operator(
            func=np.tanh, 
            jac=lambda x: np.diag(1 - np.tanh(x)**2)
            )


class Atan(Math):
    """Arctangent operator block.

    This block supports vector inputs. This is the operation it does:
        
    .. math::
        
        \\vec{y} = \\arctan(\\vec{u}) 
        
    Attributes
    ----------
    op_alg : Operator
        internal algebraic operator
    """

    def __init__(self):
        super().__init__()

        #create internal algebraic operator
        self.op_alg = Operator(
            func=np.arctan, 
            jac=lambda x: np.diag(1/(1 + x**2))
            )


class Norm(Math):
    """Vector norm operator block.

    This block computes the Euclidean norm of the input vector:
        
    .. math::
        
        y = \\|\\vec{u}\\|_2 = \\sqrt{\\sum_i u_i^2}
        
    Attributes
    ----------
    op_alg : Operator
        internal algebraic operator
    """

    def __init__(self):
        super().__init__()

        #create internal algebraic operator
        self.op_alg = Operator(
            func=np.linalg.norm, 
            jac=lambda x: x/np.linalg.norm(x)
            )


class Mod(Math):
    """Modulo operator block.

    This block supports vector inputs. This is the operation it does:
        
    .. math::
        
        \\vec{y} = \\vec{u} \\bmod m


    Note
    ----
    modulo is not differentiable at discontinuities

    Parameters
    ----------
    modulus : float
        modulus value
        
    Attributes
    ----------
    op_alg : Operator
        internal algebraic operator
    """

    def __init__(self, modulus=1.0):
        super().__init__()

        self.modulus = modulus

        #create internal algebraic operator
        self.op_alg = Operator(
            func=lambda x: np.mod(x, self.modulus), 
            jac=lambda x: np.diag(np.ones_like(x)) 
            )


class Clip(Math):
    """Clipping/saturation operator block.

    This block supports vector inputs. This is the operation it does:
        
    .. math::
        
        \\vec{y} = \\text{clip}(\\vec{u}, u_{min}, u_{max}) 

    Parameters
    ----------
    min_val : float, array_like
        minimum clipping value
    max_val : float, array_like
        maximum clipping value
        
    Attributes
    ----------
    op_alg : Operator
        internal algebraic operator
    """

    def __init__(self, min_val=-1.0, max_val=1.0):
        super().__init__()

        self.min_val = min_val
        self.max_val = max_val

        #create internal algebraic operator
        def _clip_jac(x):
            """Jacobian is 1 where not clipped, 0 where clipped"""
            mask = (x >= self.min_val) & (x <= self.max_val)
            return np.diag(mask.astype(float))

        self.op_alg = Operator(
            func=lambda x: np.clip(x, self.min_val, self.max_val), 
            jac=_clip_jac
            )


class Matrix(Math):
    """Linear matrix operation (matrix-vector product).

    This block supports vector inputs. This is the operation it does:
        
    .. math::
        
        \\vec{y} = \\mathbf{A} \\vec{u} 

    Parameters
    ----------
    A : np.ndarray
        matrix, 2d array with dim=2
        
    Attributes
    ----------
    op_alg : Operator
        internal algebraic operator
    """

    def __init__(self, A=np.eye(1)):
        super().__init__()

        # check matrix dimension and shape
        if not A.ndim == 2:
            raise ValueError("'A' must be dim 2")
        n, m = A.shape

        # initialize io registers
        self.inputs = Register(size=m)
        self.outputs = Register(size=n)

        # block matrix
        self.A = A

        self.op_alg = Operator(
            func=lambda u: np.dot(self.A, u), 
            jac=lambda u: np.dot(self.A, u)
            )