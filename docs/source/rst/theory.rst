Theory
======

*Abdbeam* uses a thin-walled anisotropic composite beam theory that includes closed cells with open branches and booms. For the detailed theory behind Abdbeam, Ref. [1]_ is the most complete reference. Ref [2]_ and Ref. [3]_ are also great references on its originating theory.


Hypothesis and Limitations
--------------------------

- Prismatic thin-walled beams undergoing small deformations;
- Materials behave in a linearly elastic manner;
- Bernoulli-Navier hypothesis: originally plane cross sections of a beam undergoing bending remain plane and perpendicular to the axis of the beam;
- The effects of shear deformation and restrained warping are neglected.
 

Sign Conventions
----------------

The figure below show the sign conventions for the cross section axis, its normal force, bending moments, torque and transverse shear forces:

.. image:: /images/abdbeam_sign_conventions.png
   :width: 400px

.. warning::

   The Mz sign convention used in *Abdbeam* requires special attention. As seen above, the sign of the bending moments is that they are positive when they induce tension in the positive yz quadrant of the beam cross section (as seen in Ref. [1]_ to Ref. [4]_). In contrast, Finite Element Analysis software packages commonly adopt a right-hand rule to define the tension and compression signs of a positive Mz bending moment. Remember to multiply by -1 cross sectional Mz loads obtained from these sources.

.. rubric:: References

.. [1] `Victorazzo DS, De Jesus A. A Kollár and Pluzsik anisotropic composite beam theory for arbitrary multicelled cross sections. Journal of Reinforced Plastics and Composites. 2016 Dec;35(23):1696-711. <https://journals.sagepub.com/doi/abs/10.1177/0731684416665493>`_
.. [2] `Kollár LP, Springer GS. Mechanics of composite structures. Cambridge university press; 2003 Feb 17. <https://www.amazon.com/Mechanics-Composite-Structures-L%C3%A1szl%C3%B3-Koll%C3%A1r/dp/0521126908/ref=sr_1_1?ie=UTF8&qid=1544936929&sr=8-1&keywords=Mechanics+of+composite+structures>`_
.. [3] `Kollár LP and Pluzsik A. Analysis of thin-walled composite beams with arbitrary layup. J Reinf Plast Compos 2002; 21: 1423–1465. <https://journals.sagepub.com/doi/abs/10.1177/0731684402021016928>`_
.. [4] `Megson TH. Aircraft structures for engineering students. Butterworth-Heinemann; 2016 Oct 17. <https://www.amazon.com/Aircraft-Structures-Engineering-Students-Aerospace/dp/0080969054/ref=sr_1_1?ie=UTF8&qid=1548602525&sr=8-1&keywords=Megson+TH.+Aircraft+structures+for+engineering+students>`_ 
