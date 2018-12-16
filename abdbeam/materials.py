# -*- coding: utf-8 -*-
"""
Contains all material classes used by Section objects.
"""

import numpy as np
import math

class Material:
    """
    Parent class for all materials.

    May be instantiated directly but self.abd_c needs to be manually entered.
   
    Attributes
    ----------
    t : float
        The thickness of the material. Since for this parent class the 
        compliance matrix is provided directly, the thickness is used for 
        reference/plot purposes only.
    abd_c : numpy.ndarray
        The material 6x6 compliance matrix based on CLT (Classical Laminate
        Theory).
    description: str
        The description of the material.
        
    Methods
    -------
    calculate_abd_c()
        Method used by classes that inherit this base class to calculate the
        compliance matrix of the material based on the Classical Laminate 
        Theory.
    """


    def __init__(self, t, abd_c=np.zeros((6,6), float), description=''):
        """
        Creates a material instance.
        
        Parameters
        ----------
        t : float
            The thickness of the material. For reference/plot purposes only.
        abd_c : numpy.ndarray, default np.zeros((6,6), float)
            The material 6x6 compliance matrix based on CLT (Classical Laminate
            Theory).
        description: str, default ''
            The description of the material.     
        """
        self.t = t
        self.abd_c = abd_c
        self.description = description


    def __repr__(self):
        return ('{}({}, {}, {})'.format(self.__class__.__name__, self.t, 
                self.abd_c, self.description))


    def calculate_abd_c(self):
        """
        Method used by classes that inherit this base class to calculate the
        compliance matrix of the material based on the Classical Laminate 
        Theory.
        
        For this parent class, abd_c needs to be manually provided. 
        """
        return None


class Isotropic(Material):
    """
    An isotropic material that inherits the Material class.
    
    Attributes
    ----------
    t : float
        The thickness of the material.
    E : float
        The Young Modulus of the material.
    v: float
		The Poisson Ratio of the material.
    description: str
        The description of the material.
    abd_c : numpy.ndarray
        The material 6x6 compliance matrix based on CLT (Classical Laminate
        Theory).
        
    Methods
    -------
    calculate_abd_c()
        Calculates the compliance matrix of the isotropic material based on 
	    the Classical Laminate Theory.
		
    Examples
    --------
    .. code-block:: python

        mts = dict()
        mts[1] = ab.Isotropic(0.08, 10600000, 0.33)
        mts[1].calculate_abd_c()
	"""


    def __init__(self, t, E, v, description=''):
        """
        Creates an isotropic material instance.
        
        Parameters
        ----------
        t : float
            The thickness of the material.
	    E : float
            The Young Modulus of the material.
	    v: float
		    The Poisson Ratio of the material.
        description: str, default ''
            The description of the material.     
        """	

        super().__init__(t)
        self.t = t
        self.E = E
        self.v = v
        self.description = description


    def __repr__(self):
        return ('{}({}, {}, {}, {})'.format(self.__class__.__name__, self.t, 
                self.E, self.v, self.description))
        

    def calculate_abd_c(self):
        """
        Calculates the compliance matrix of the isotropic material based on 
	    the Classical Laminate Theory.
        """
        E=self.E
        t=self.t
        v=self.v
        abd_c=self.abd_c
        abd_c[0,0] = 1 / (E*t)
        abd_c[1,1] = 1 / (E*t)
        abd_c[0,1] = -v / (E*t)
        abd_c[1,0] = -v / (E*t)
        abd_c[2,2] = 2 * (1+v) / (E*t)
        abd_c[3,3] = 12 / (E*t**3)
        abd_c[4,4] = 12 / (E*t**3)
        abd_c[3,4] = -v*12 / (E*t**3)
        abd_c[4,3] = -v*12 / (E*t**3)
        abd_c[5,5] = 24*(1+v) / (E*t**3)


class ShearConnector(Material):
    """
    A shear connector that inherits the Material class.
	
	Shear connectors transfer only shear (only the compliance matrix term a66
    is not zero and equal to 1/(G*t)).
    
    Attributes
    ----------
    t : float
        The thickness of the shear connector material.
    G : float
        The Shear Modulus of the shear connector material.
    description: str
        The description of the material.
    abd_c : numpy.ndarray
        The material 6x6 compliance matrix based on CLT (Classical Laminate
        Theory).
        
    Methods
    -------
    calculate_abd_c()
        Calculates the compliance matrix of the shear connector based on the
	    Classical Laminate Theory.
		
    Examples
    --------
    .. code-block:: python

        mts = dict()
        mts[1] = ab.ShearConnector(0.25, 6380000, 'Quarter Inch Fastener')
        mts[1].calculate_abd_c()    
    """	


    def __init__(self, t, G, description=''):
        """
        Creates a shear connector material instance.
        
        Parameters
        ----------
        t : float
            The thickness of the shear connector material.
        G : float
            The Shear Modulus of the shear connector material.
        description: str, default ''
            The description of the material.  
        """	
        super().__init__(t)
        self.t = t
        self.G = G
        self.description = description


    def __repr__(self):
        return ('{}({}, {}, {}, {})'.format(self.__class__.__name__, self.t, 
                self.G, self.description))


    def calculate_abd_c(self):
        """
        Calculates the compliance matrix of the shear connector based on the
	    Classical Laminate Theory.
		
		Note: all terms of the compliance matrix are zero, except a66 = 
		1/(G*t).
        """
        self.abc_c[2,2] = 1.0 / (self.G*self.t)		


class PlyMaterial():
    """
    A ply material used by the Laminate class.
    
    Attributes
    ----------
    t : float
        The ply thickness.
    E1 : float
        The axial stiffness of the ply.
    E2 : float
        The transverse stiffness of the ply.
	G12: float
        The shear modulus of the ply.
    description: str
        The description of the ply.

    Examples
    --------
    .. code-block:: python

        ply_mat = ab.PlyMaterial(0.166666, 148000, 9650, 4550, 0.3)
        mts[1].ply_materials[1] = ply_mat   
    """	

    def __init__(self, t, E1, E2, G12, v12, description=''):
        """
        Creates a ply material instance.
        
        Parameters
        ----------
        t : float
            The ply thickness.
        E1 : float
            The axial stiffness of the ply.
        E2 : float
            The transverse stiffness of the ply.
	    G12: float
		    The shear modulus of the ply.
        description: str, default = ''
            The description of the ply.  
        """	
        self.t = t
        self.E1 = E1
        self.E2 = E2
        self.G12 = G12
        self.v12 = v12
        self.description = description


    def __repr__(self):
        return ('{}({}, {}, {}, {}, {}, {})'.format(self.__class__.__name__, 
                self.t, self.E1, self.E2, self.G12, self.v12, 
                self.description))
        

class Laminate(Material):
    """
    A composite laminate material that inherits the Materials class.
	
    Attributes
    ----------
    t : float
        The laminate thickness. Calculated by the calculate_abd_c() method.
	ply_materials : dict
        Of the form {int : abdbeam.PlyMaterial}		
    plies : list
        A list that defines the laminate stacking sequence. Plies are the 
        elements of this list, which in turn are represented as 2-elements 
        lists of angle and material ids the form [float, int]. The
        first element in the plies list is the bottom ply.
    symmetry : {'T', 'S', 'SM', 'SMEAR'}, default = 'T'
        'T' means all plies are defined in the plies list; 'S' means symmetry
        will be applied to the plies list; 'SM' means symmetry will be applied
        around the last item in the plies list; 'SMEAR' means the effects of 
        the plies stacking sequence will be ignored ([D]=t**2 / 12 * [A]).
    abd_c : numpy.ndarray
        The material 6x6 compliance matrix based on CLT (Classical Laminate
        Theory).
    abd : numpy.ndarray
        The material 6x6 stiffness matrix based on CLT (Classical Laminate
        Theory).


    Methods
    -------
    calculate_abd_c()
        Calculates the compliance matrix of the laminate based on the
	    Classical Laminate Theory. Will also retain the abd stiffness matrix
		array in self.abd.
		
    Examples
    --------
	Creating a symmetric and balanced 8 plies laminate: 
	
    .. code-block:: python
	
        mts = dict()
        mts[1] = ab.Laminate()
        ply_mat = ab.PlyMaterial(0.166666, 148000, 9650, 4550, 0.3)
        mts[1].ply_materials[1] = ply_mat
        mts[1].plies = [[45,1], [-45,1], [0,1], [90,1]]
        mts[1].symmetry = 'S'
        mts[1].calculate_abd_c()

    """


    def __init__(self):
        super().__init__(0.0)
        self.t = 0.0
        self.plies = list()
        self.ply_materials = dict()
        self.abd = np.zeros((6,6), float)
        self.symmetry = 'T'


    def __repr__(self):
        return ('{}()'.format(self.__class__.__name__))
    

    def calculate_abd_c(self):
        """
		Calculates the compliance matrix of the laminate based on the 
		Classical Laminate Theory. Will also retain the abd stiffness matrix
		array in self.abd.
        """
        self.t = 0
        abd = np.zeros((6,6), float)
        plies = self.plies[:]
        plies_ql = list()
        if self.symmetry == 'S':
            for p in reversed(self.plies):
                plies.append(p)
        elif self.symmetry == 'SM':
            for p in reversed(self.plies[:-1]):
                plies.append(p)
        # Calculate ply properties and laminate thickness
        for angle, ply_mat_id in plies:
            mat = self.ply_materials[ply_mat_id]
            self.t += mat.t
            E1 = mat.E1
            E2 = mat.E2
            v12 = mat.v12
            G12 = mat.G12
            v21 = (E2 * v12) / E1
            qp = np.zeros((3,3), float)
            ql = np.zeros((3,3), float)
            #Ply Stiffness in Ply Axis
            qp[0, 0] = E1 / (1 - (v12 * v21))
            qp[1, 1] = E2 / (1 - (v12 * v21))
            qp[0, 1] = (E2 * v12) / (1 - (v12 * v21))
            qp[1, 0] = qp[0, 1]
            qp[2, 2] = G12
            m = np.cos((angle) * math.pi / 180)
            n = np.sin((angle) * math.pi / 180)
            #Ply Stiffness in Laminate Axis
            ql[0, 0] = (m**4 * qp[0, 0] + n**4 * qp[1, 1] + 2*m**2 * n**2
                        * qp[0, 1] + 4*m**2 * n**2 * qp[2, 2])
            ql[1, 1] = (n**4 * qp[0, 0] + m**4 * qp[1, 1] + 2*m**2 * n**2
                        * qp[0, 1] + 4*m**2 * n**2 * qp[2, 2])
            ql[0, 1] = (m**2 * n**2 * qp[0, 0] + m**2 * n**2 * qp[1, 1]
                        + (m**4 + n**4)*qp[0, 1] - 4*m**2 * n**2 * qp[2, 2])
            ql[0, 2] = (m**3 * n*qp[0, 0] - m*n**3 * qp[1, 1] + (m*n**3
                        - m**3 * n)*qp[0, 1] + 2*(m*n**3 - m**3 * n)*qp[2, 2])
            ql[1, 2] = (m*n**3 * qp[0, 0] - n*m**3 * qp[1, 1] + (n*m**3 - n**3
                        * m)*qp[0, 1] + 2*(n*m**3 - n**3 * m) * qp[2, 2])
            ql[2, 2] = (m**2 * n**2 * qp[0, 0] + m**2 * n**2 * qp[1, 1]
                        - 2*m**2 * n**2 * qp[0, 1] + ((m**2 - n**2)**2)
                        * qp[2, 2])
            ql[1, 0] = ql[0, 1]
            ql[2, 1] = ql[1, 2]
            ql[2, 0] = ql[0, 2]
            plies_ql.append(ql)
        h = np.zeros((len(plies)+1), float)
        h[0] = - self.t / 2
        # Calculate the [abd_c] stiffness matrix
        for i, (angle, ply_mat_id) in enumerate(plies):
            mat = self.ply_materials[ply_mat_id]
            h[i+1] = h[i] + mat.t
            for j in range(3):
                for k in range(3):
                    # calculate the extensional matrix [A]
                    abd[j,k] += plies_ql[i][j,k] * mat.t
                    # calculate the coupling stiffness matrix [B]
                    abd[j,k+3] += 0.5*plies_ql[i][j,k]*(h[i+1]**2 - h[i]**2)
                    abd[k+3,j] = abd[j,k+3]
                    # calculate the bending stiffness matrix [D]
                    abd[j+3,k+3] += (1/3 * plies_ql[i][j,k]*(h[i+1]**3
                                     - h[i]**3))
        if self.symmetry == 'SMEAR':
            abd[3:6, 3:6] =  (self.t**2 / 12)*abd[0:3, 0:3]
            abd[0:3, 3:6] = 0.0
            abd[3:6, 0:3] = 0.0
        self.abd = abd		
        self.abd_c = np.linalg.inv(abd) # The compliance [abd_c] matrix