# -*- coding: utf-8 -*-
"""
Module defining the classes ``Section``, ``Segment``, ``Point`` and ``Load``.
"""

import numpy as np
import pandas as pd
import math
from .materials import (Material, Isotropic, ShearConnector, PlyMaterial,
                        Laminate)


class Section:
    """
    Class that defines a cross section, calculates its properties and internal
    loads.

    Attributes
    ----------
    materials : dict
        Of the form {int : abdbeam.Material}
    points : dict
        Of the form {int : abdbeam.Point}
    segments : dict
        Of the form {int : abdbeam.Segment}
    loads : dict
        Of the form {int : abdbeam.Load}
    cells : dict
        An output of the form {int : abdbeam.Cell}
    yc : float
        The centroid Y coordinate.
    zc : float
        The centroid Z coordinate.
    ys : float
        The shear center Y coordinate.
    zs : float
        The shear center Z coordinate.
    p_c : numpy.ndarray
        The 4x4 section stiffness matrix relative to the centroid.
    w_c : numpy.ndarray
        The 4x4 section compliance matrix relative to the centroid.
    p : numpy.ndarray
        The 4x4 section stiffness matrix relative to the section origin.
    w : numpy.ndarray
        The 4x4 section compliance matrix relative to the section origin.
    weight : float
        The section weight per unit length.
    principal_axis_angle : float
        The angle of the coordinate system Y'-Z' relative to Y-Z in which the
        moment of Inertia Iy'z' is zero. Only applicable to isotropic beams.
    sc_int_strains_df : pandas.DataFrame
        The pandas dataframe containing the axial strain, the Y curvature, the
        Z curvature and the rate of twist of the section relative to the
        centroid (to be implemented).
    sgs_int_lds_df: pandas.DataFrame
        A pandas dataframe containing the segments internal loads for all load
        cases in the loads dictionary. Populated by the
        calculate_internal_loads method.
    pts_int_lds_df: pandas.DataFrame
        A pandas dataframe containing the points internal loads for all load
        cases in the loads dictionary. Populated by the
        calculate_internal_loads method.

    Methods
    -------
    summary()
        Prints a summary of the section properties.
    calculate_properties()
        Calculates the section properties.
    calculate_internal_loads()
        Calculates internal loads for all load cases in the loads dictionary.
    print_internal_loads()
        Prints to the console segment and point internal loads for all load
        cases in the loads dictionary.

    Examples
    --------
    Creating a 2-cells beam cross section comprised of asymmetric laminate
    segments (see appendix example in reference theory paper):

    .. code-block:: python

        import abdbeam as ab
        sc = ab.Section()
        mts = dict()
        mts[1] = ab.Laminate()
        ply_mat = ab.PlyMaterial(0.166666, 148000, 9650, 4550, 0.3)
        mts[1].ply_materials[1] = ply_mat
        mts[1].plies = [[0,1], [0,1], [0,1], [0,1], [0,1], [0,1]] + [[45,1]]*6
        mts[1].symmetry = 'T'
        mts[1].calculate_properties()
        pts = dict()
        pts[1] = ab.Point(0, -35)
        pts[2] = ab.Point(-50, -35)
        pts[3] = ab.Point(-50, 35)
        pts[4] = ab.Point(0, 35)
        pts[5] = ab.Point(50, 35)
        pts[6] = ab.Point(50, -35)
        sgs = dict()
        sgs[1] = ab.Segment(1,2,1)
        sgs[2] = ab.Segment(2,3,1)
        sgs[3] = ab.Segment(3,4,1)
        sgs[4] = ab.Segment(4,1,1)
        sgs[5] = ab.Segment(4,5,1)
        sgs[6] = ab.Segment(5,6,1)
        sgs[7] = ab.Segment(6,1,1)
        sc.materials = mts
        sc.points = pts
        sc.segments = sgs
        sc.calculate_properties()
        sc.summary()

    Adding two load cases to the section above and printing their internal
    loads:

    .. code-block:: python

        Lds = dict()
        Lds[101] = ab.Load(1000.0,25000,-36000)
        Lds[102] = ab.Load(Px=1500.0)
        Lds[103] = ab.Load(Vz_s=1000.0)
        sc.loads = Lds
        sc.calculate_internal_loads()
        sc.print_internal_loads()
    """


    def __init__(self):
        self.materials = dict()
        self.points = dict()
        self.segments = dict()
        self.cells = dict()
        self.loads = dict()
        self.yc = 0.0
        self.zc = 0.0
        self.ys = 0.0
        self.zs = 0.0
        self.p_c = np.zeros((4, 4))
        self.w_c = np.zeros((4, 4))
        self.p = np.zeros((4, 4))
        self.w = np.zeros((4, 4))
        self.weight = 0.0
        self.principal_axis_angle = 0.0
        self.u_Px = None
        self.u_Px_pt = None
        self.u_My = None
        self.u_My_pt = None
        self.u_Mz = None
        self.u_Mz_pt = None
        self.u_Tx = None
        self.u_Tx_pt = None
        self.u_Vy = None
        self.u_Vy_pt = None
        self.u_Vz = None
        self.u_Vz_pt = None
        self.sc_int_strains_df = None
        self.sgs_int_lds_df = None
        self.pts_int_lds_df = None


    def __repr__(self):
        return ('{}()'.format(self.__class__.__name__))


    def summary(self):
        """Prints a summary of the section properties."""
        msg = ['']
        msg.append('Section Summary')
        msg.append('===============')
        msg.append('')
        msg.append('Number of points: {}'.format(len(self.points)))
        msg.append('Number of segments: {}'.format(len(self.segments)))
        msg.append('Number of cells: {}'.format(len(self.cells)))
        msg.append('')
        msg.append('Centroid')
        msg.append('--------')
        msg.append('yc\t = {:.8e}'.expandtabs(6).format(self.yc))
        msg.append('zc\t = {:.8e}'.expandtabs(6).format(self.zc))
        msg.append('')
        msg.append('Shear Center')
        msg.append('------------')
        msg.append('ys\t = {:.8e}'.expandtabs(6).format(self.ys))
        msg.append('zs\t = {:.8e}'.expandtabs(6).format(self.zs))
        msg.append('')
        msg.append('Replacement Stiffnesses')
        msg.append('-----------------------')
        EIyy = self.p_c[1,1]
        EIzz = self.p_c[2,2]
        EIyz = self.p_c[1,2]
        EImax = 0.5*(EIyy + EIzz) + (0.25*(EIyy-EIzz)**2 + EIyz**2)**0.5
        EImin = 0.5*(EIyy + EIzz) - (0.25*(EIyy-EIzz)**2 + EIyz**2)**0.5
        msg.append('EA\t = {:.8e}'.expandtabs(6).format(self.p_c[0,0]))
        msg.append('EIyy\t = {:.8e}'.expandtabs(6).format(EIyy))
        msg.append('EIzz\t = {:.8e}'.expandtabs(6).format(EIzz))
        msg.append('EIyz\t = {:.8e}'.expandtabs(6).format(EIyz))
        msg.append('GJ\t = {:.8e}'.expandtabs(6).format(self.p_c[3,3]))
        msg.append('EImax\t = {:.8e}'.expandtabs(6).format(EImax))
        msg.append('EImin\t = {:.8e}'.expandtabs(6).format(EImin))
        msg.append('Angle\t = {:.8e}'.expandtabs(6).format(
                                                  self.principal_axis_angle))
        msg.append('')
        msg.append('[P_c] - Beam Stiffness Matrix at the Centroid')
        msg.append('---------------------------------------------')
        msg.append(np.array_str(self.p_c))
        msg.append('')
        msg.append('[W_c] - Beam Compliance Matrix at the Centroid')
        msg.append('----------------------------------------------')
        msg.append(np.array_str(self.w_c))
        msg.append('')
        msg.append('[P] - Beam Stiffness Matrix at the Origin')
        msg.append('-----------------------------------------')
        msg.append(np.array_str(self.p))
        msg.append('')
        msg.append('[W] - Beam Compliance Matrix at the Origin')
        msg.append('------------------------------------------')
        msg.append(np.array_str(self.w))
        for line in msg:
            print(line)


    def calculate_properties(self):
        """Calculates the section properties."""
        for mat in self.materials.values():
            mat.calculate_properties()
        for pt in self.points.values():
            pt.adj_point_ids = set()
        for seg in self.segments.values():
            seg.calculate_properties(self.points, self.materials)
            self.weight += seg.density*seg.bk*seg.t
            self.points[seg.point_a_id].adj_point_ids.add(seg.point_b_id)
            self.points[seg.point_b_id].adj_point_ids.add(seg.point_a_id)
        self._perform_section_checks()
        # Add the contribution of each segment (as open) to the beam
        # stiffness matrix.
        for seg in self.segments.values():
            self.p += (np.dot(np.transpose(seg.rk),
                       np.dot(seg.wk_inv, seg.rk)))
        # Add the contributions from the booms
        for pt in self.points.values():
                self.p[0, 0] += pt.EA
                self.p[0, 1] += pt.z*pt.EA
                self.p[0, 2] += pt.y*pt.EA
                self.p[1, 0] += pt.z*pt.EA
                self.p[1, 1] += (pt.z**2)*pt.EA
                self.p[1, 2] += pt.y*pt.z*pt.EA
                self.p[2, 0] += pt.y*pt.EA
                self.p[2, 1] += pt.y*pt.z*pt.EA
                self.p[2, 2] += (pt.y**2)*pt.EA
                self.p[3, 3] += pt.GJ
        # Add the contribution from the cells
        self._detect_and_configure_cells()
        self._perform_post_cells_checks()
        if self.cells:
            self._calculate_cell_contributions()
        else:
            self.w = np.linalg.inv(self.p)
        # Calculate the section centroid location
        tmp = np.dot(np.linalg.inv(np.array([[-self.w[1, 1], -self.w[1, 2]],
              [-self.w[1, 2], -self.w[2, 2]]])), np.array([[self.w[0, 1]],
              [self.w[0, 2]]]))
        self.yc = tmp[1,0]
        self.zc = tmp[0,0]
        # Calculate the section stiffness and compliance matrices w.r.t.
        # the centroid
        rb =  np.identity(4)
        rb[1,0] = self.zc
        rb[2,0] = self.yc
        self.w_c = np.dot(np.transpose(rb), np.dot(self.w, rb))
        self.p_c = np.linalg.inv(self.w_c)
        # Calculate the principal axis angle theta
        self.principal_axis_angle = (0.5 *np.arctan(-2 * self.p_c[1,2] /
                                    (self.p_c[1,1]-self.p_c[2,2])))
        self.principal_axis_angle = math.degrees(self.principal_axis_angle)
        # Calculate and store the unitary internal loads
        self.u_Px, self.u_Px_pt = self._calc_unit_internal_loads(0)
        self.u_My, self.u_My_pt = self._calc_unit_internal_loads(1)
        self.u_Mz, self.u_Mz_pt = self._calc_unit_internal_loads(2)
        self.u_Tx, self.u_Tx_pt = self._calc_unit_internal_loads(3)
        self.u_Vy, self.u_Vy_pt = self._calc_unit_internal_loads(4)
        self.u_Vz, self.u_Vz_pt = self._calc_unit_internal_loads(5)


    def _calculate_cell_contributions(self):
        n = len(self.cells)
        coef = np.zeros((2*n + 4, 2*n + 4))
        for i, cell_id in zip(range(n), self.cells):
            for seg_id, seg in self.segments.items():
                f = seg.cell_factors[cell_id]
                coef[i, 2*n] += f*seg.ik[0, 0]
                coef[i, 2*n + 1] += f*seg.ik[0, 1]
                coef[i, 2*n + 2] += f*seg.ik[0, 2]
                coef[i, 2*n + 3] += f*seg.ik[0, 3]
                coef[i, i] += abs(f)*seg.fk[0, 0]
                coef[i, i+n] += abs(f)*seg.fk[0, 1]
                coef[i+n, 2*n] += f*seg.ik[1, 0]
                coef[i+n, 2*n+1] += f*seg.ik[1, 1]
                coef[i+n, 2*n+2] += f*seg.ik[1, 2]
                coef[i+n, 2*n+3] += f*seg.ik[1, 3]
                coef[i+n, i] += abs(f)*seg.fk[1, 0]
                coef[i+n, i+n] += abs(f)*seg.fk[1, 1]
                for k, cell_id_ in zip(range(n), self.cells):
                    if k == i:
                        continue
                    f_ = seg.cell_factors[cell_id_]
                    coef[i, k] += -abs(f)*abs(f_)*seg.fk[0, 0]
                    coef[i, k+n] += -abs(f)*abs(f_)*seg.fk[0, 1]
                    coef[i+n, k] += -abs(f)*abs(f_)*seg.fk[1, 0]
                    coef[i+n, k+n] += -abs(f)*abs(f_)*seg.fk[1, 1]
                # P equation
                coef[2*n, i] += -f*seg.ik[0, 0]
                coef[2*n, i+n] += -f  * seg.ik[1, 0]
                # My equation
                coef[2*n + 1, i] += -f*seg.ik[0, 1]
                coef[2*n + 1, i+n] += -f*seg.ik[1, 1]
                # Mz equation
                coef[2*n + 2, i] += -f*seg.ik[0, 2]
                coef[2*n + 2, i+n] += -f*seg.ik[1, 2]
                # T equation
                coef[2*n + 3, i] += -f*seg.ik[0, 3]
                coef[2*n + 3, i+n] += -f*seg.ik[1, 3]
            # The rotation rate coefficient
            coef[i, 2*n + 3] += -2*self.cells[cell_id].area
            # The (2An*X1n) equation
            coef[2*n + 3, i] += 2*self.cells[cell_id].area
        coef[2*n:(2*n + 4), 2*n:(2*n + 4)] = self.p
        # Add a small number to the diagonal to assure coef can be inverted
        # (needed for sections with connectors only)
        non_zero_diag = np.zeros_like(coef)
        np.fill_diagonal(non_zero_diag, 1e-99)
        coef+=non_zero_diag
        tmp_matrix = np.linalg.inv(coef)
        # Cycle load_i: unitary Px, then My, then Mz, then Tx
        #----------------------
        for load_i in range(4):
            vect = np.zeros((2*n + 4, 1))
            vect[2*n + load_i, 0] = 1
            tmp_matrix_ = np.dot(tmp_matrix, vect)
            # Fill the compliance matrix 1st column
            self.w[:, load_i] = tmp_matrix_[2*n:(2*n + 4), 0]
            # Store X1 and X2 calculated for each cell
            i = 0
            for cell in self.cells.values():
                cell.x[0, load_i] = tmp_matrix_[i, 0]
                cell.x[1, load_i] = tmp_matrix_[i+n, 0]
                i += 1
            # Fill the segment [S]k
            # Calculate Nxy and My
            for seg_id, seg in self.segments.items():
                Nxy_My_vect = np.zeros((2, 1))
                for cell_id, cell in self.cells.items():
                    f = seg.cell_factors[cell_id]
                    Nxy_My_vect[0, 0] += f*cell.x[0, load_i]
                    Nxy_My_vect[1, 0] += f*cell.x[1, load_i]
                Nx_Mx_Mxy_vector = np.dot(seg.vk, Nxy_My_vect)
                seg.sk[:, load_i] = -Nx_Mx_Mxy_vector[:, 0]
        self.p = np.linalg.inv(self.w)


    def _detect_and_configure_cells(self):
        """
        Detects the cross section cells (if any) based on segment-node
        connectivity. Also calculates segment to cell interface factors and
        identifies cell cuts for shear calculations.
        """
        tmp_cell = Cell()
        cell_id = 1
        for pt_id, ipt in self.points.items():
            for adj_pt_id in ipt.adj_point_ids:
                # if a free edge will not enter
                if adj_pt_id > pt_id: # to prevent repeated cells
                    attempt_pts_ids = [adj_pt_id]
                    tmp_cell = Cell()
                    tmp_cell.point_ids = [pt_id, adj_pt_id]
                    current_pt_id = adj_pt_id
                    previous_pt_id = pt_id
                    next_pt_id = 0
                    while next_pt_id >= 0:
                        next_pt_id = self._min_clockwise_angle_adj_point_id(
                                current_pt_id, previous_pt_id, attempt_pts_ids)
                        if next_pt_id > pt_id:
                            tmp_cell.point_ids.append(next_pt_id)
                            previous_pt_id = current_pt_id
                            current_pt_id = next_pt_id
                            attempt_pts_ids.append(next_pt_id)
                        elif next_pt_id == -1:
                            # there is no next_pt_id
                            del tmp_cell.point_ids[-1]
                            if pt_id == previous_pt_id:
                                # we are back to the starting point
                                break
                            else:
                                current_pt_id = tmp_cell.point_ids[-1]
                                previous_pt_id = tmp_cell.point_ids[-2]
                                next_pt_id = 0
                        elif next_pt_id == pt_id:
                            if tmp_cell._is_closed_polygon(self.points):
                                self.cells[cell_id] = tmp_cell
                                self.cells[cell_id]._calculate_area(
                                        self.points)
                                cell_id += 1
                            break
                        else:
                            break
        if not self.cells:
            return
        # Calculate segment interface factors
        # Note that cell directions are counter-clockwise
        for seg_id, segment in self.segments.items():
            for cell_id, cell in self.cells.items():
                cell_point_ids = cell.point_ids + [cell.point_ids[0]]
                for first, second in zip(cell_point_ids, cell_point_ids[1:]):
                    if (segment.point_a_id == first
                            and segment.point_b_id == second):
                        segment.cell_factors[cell_id] = -1.0
                        cell.segment_ids.append(seg_id)
                        segment.adj_cell_id_list.append(cell_id)
                        segment.tmp_adj_cell_id_list.append(cell_id)
                        break
                    elif (segment.point_a_id == second
                            and segment.point_b_id == first):
                        segment.cell_factors[cell_id] = +1.0
                        cell.segment_ids.append(seg_id)
                        segment.adj_cell_id_list.append(cell_id)
                        segment.tmp_adj_cell_id_list.append(cell_id)
                        break
                    else:
                        segment.cell_factors[cell_id] = 0.0
        # Identify cell cuts for shear calculations
        cell_tmp_count = len(self.cells)
        while cell_tmp_count > 0:
            erase_cell_id = -1
            # look for cells that contain a segment with only one adjacent cell
            for seg_id, segment in self.segments.items():
                if len(segment.tmp_adj_cell_id_list) == 1:
                    segment.shear_cut = True
                    erase_cell_id = segment.tmp_adj_cell_id_list[0]
                    break
            # Remove the cell id from all segment.tmp.adj_cell_id_list
            for seg_id, segment in self.segments.items():
                if erase_cell_id in segment.tmp_adj_cell_id_list:
                    segment.tmp_adj_cell_id_list.remove(erase_cell_id)
            cell_tmp_count -= 1


    def _perform_post_cells_checks(self):
        # Check for floating segments or cells
        # The number of points must be the number of segments + 1 - number of
        # cells
        correct_n_pts = len(self.segments) + 1 - len(self.cells)
        assert len(self.points) == correct_n_pts, ("The section contains "
                  "disconnected segments and/or cells.")


    def _perform_section_checks(self):
        # Check for orphan points
        for pt_id, pt in self.points.items():
            assert (pt.adj_point_ids), ("Point {} is not associated to any"
                 " segment. A section cannot have orphan points.").format(
                         pt_id)

        for s, (sg_id, sg) in enumerate(self.segments.items()):
            assert (sg.bk > 0), ("Segment {} has zero length.").format(sg_id)
            ya = self.points[sg.point_a_id].y
            za = self.points[sg.point_a_id].z
            yb = self.points[sg.point_b_id].y
            zb = self.points[sg.point_b_id].z
            dy1 = yb-ya
            cosalpha1 = dy1/sg.bk
            for i_s, (i_sg_id, i_sg) in enumerate(self.segments.items()):
                if i_s <= s:
                    continue
                assert (i_sg.bk > 0), ("Segment {} has zero length.").format(
                        i_sg_id)
                yc = self.points[i_sg.point_a_id].y
                zc = self.points[i_sg.point_a_id].z
                yd = self.points[i_sg.point_b_id].y
                zd = self.points[i_sg.point_b_id].z
                dy2 = yd-yc
                cosalpha2 = dy2/i_sg.bk
                # check for segments intersecting each other
                assert (_no_intersec_check(ya, za, yb, zb, yc, zc, yd,
                    zd, sg.point_a_id, sg.point_b_id, i_sg.point_a_id,
                    i_sg.point_b_id)), ("Segment {} intersects with segment "
                    "{}. A section cannot have intersecting segments.").format(
                    sg_id, i_sg_id)

                # check for coincident segments
                coincident = ((ya == yc and za == zc and yb == yd and zb == zd)
                        or (ya == yd and za == zd and yb == yc and zb == zc))
                assert not coincident, ("Segment {} overlaps segment {}. "
                        "Section segments cannot overlap each other.").format(
                        sg_id, i_sg_id)
                # check for partially overlapping points and segments
                if abs(cosalpha1) == abs(cosalpha2):
                    # if the segments are parallel
                    bool_chk = (_no_overlap(ya, za, yc, zc, yd, zd,
                            sg.point_a_id, i_sg.point_a_id, i_sg.point_b_id)
                            * _no_overlap(yb, zb, yc, zc, yd, zd,
                            sg.point_b_id, i_sg.point_a_id, i_sg.point_b_id)
                            * _no_overlap(yc, zc, ya, za, yb, zb,
                            i_sg.point_a_id, sg.point_a_id, sg.point_b_id)
                            * _no_overlap(yd, zd, ya, za, yb, zb,
                            i_sg.point_b_id, sg.point_a_id, sg.point_b_id))
                    assert (bool_chk == 1), ("Segment {} partially overlaps "
                           "segment {}. Section segments cannot overlap each "
                           "other.").format(sg_id, i_sg_id)


    def _calc_unit_internal_loads(self, unit_load):
        Px_c = 0.0
        My = 0.0
        Mz = 0.0
        Tx = 0.0
        Vy_s = 0.0
        Vz_s = 0.0
        if unit_load == 0:
            Px_c = 1.0
        elif unit_load == 1:
            My = 1.0
        elif unit_load == 2:
            Mz = 1.0
        elif unit_load == 3:
            Tx = 1.0
        elif unit_load == 4:
            Vy_s = 1.0
        elif unit_load == 5:
            Vz_s = 1.0
        # Create the Pandas dataframes that will store the unitary loads second
        # degree polynomial constants C2, C1, C0 in C2*N^2 + C1*N + C0
        col = pd.MultiIndex.from_product([['Nx','Nxy','Mx','My','Mxy'],
                                          ['C2','C1','C0']])
        u_sgs = pd.DataFrame(index=list(self.segments.keys()),columns=col)
        u_sgs.drop(u_sgs.columns[[0,6,9,12]], axis=1, inplace=True)
        u_sgs.index.name = 'Segment_Id'
        u_sgs[:] = 0.0
        u_pts = pd.DataFrame(index=list(self.points.keys()),
                            columns=['Px', 'Tx'])
        u_pts.index.name = 'Point_Id'
        # Calculate section loads at the section origin
        origin_loads = np.array([Px_c, My + Px_c*self.zc, Mz + Px_c
                                 *self.yc, Tx])
        # Centroid loads
        ct_loads = np.array([Px_c, My, Mz, Tx])
        # Calculate section strains
        #origin_strains = np.dot(self.w, origin_loads)
        o_s_c_strs = list(np.dot(self.w_c, ct_loads))
        # Calculate point loads and strains
        #----------------------------------
        shear_const = self.p_c[1,1]*self.p_c[2,2] - self.p_c[1,2]**2
        pt_Nxy_contribution = {pt_id: 0.0 for pt_id in self.points.keys()}
        for p, (pt_id, pt) in enumerate(self.points.items()):
            o_pt_Px = 0.0
            o_pt_Tx = 0.0
            if pt.EA > 0:
                o_pt_Px = (pt.EA*(o_s_c_strs[0] + o_s_c_strs[1]
                          *(pt.z-self.zc) + o_s_c_strs[2]*(pt.y-self.yc)))
                pt_Nxy_contribution[pt_id] = (-(self.p_c[2,2]*Vz_s
                              - self.p_c[1,2]*Vy_s)*pt.EA*(pt.z-self.zc)
                              - (self.p_c[1,1]*Vy_s - self.p_c[1,2]*Vz_s)
                              * pt.EA*(pt.y-self.yc))
                pt_Nxy_contribution[pt_id] /= shear_const
                #o_pt_ex = o_pt_Px/pt.EA * 1000000.0
            if pt.GJ > 0:
                o_pt_Tx  = pt.GJ * o_s_c_strs[3]
                #o_pt_v = o_pt_Tx / pt.GJ
            u_pts.loc[pt_id, 'Px'] = o_pt_Px
            u_pts.loc[pt_id, 'Tx'] = o_pt_Tx
        #----------------------------------
        # Calculate the segments loads
        #-----------------------------
        # Loads are recovered at three points (either linear or quadratic)
        Nx_k = np.zeros((len(self.segments),3))
        Nxy_k = np.zeros((len(self.segments),3))
        Mx_k = np.zeros((len(self.segments),3))
        My_k = np.zeros((len(self.segments),3))
        Mxy_k = np.zeros((len(self.segments),3))
        # And for the dNx/dx at the 3 recovery points + 2
        dqop = np.zeros((len(self.segments),5))
        for s, (sg_id, sg) in enumerate(self.segments.items()):
            # Calculate Nxy and My from the cells
            # The cells contribute with constant values
            Nxy_My = np.zeros((2), float)
            for cell_id, cell in self.cells.items():
                Nxy_My[0] += sg.cell_factors[cell_id]*(origin_loads[0]
                      * cell.x[0, 0] + origin_loads[1]*cell.x[0, 1]
                      + origin_loads[2]*cell.x[0,2] + origin_loads[3]
                      * cell.x[0, 3])
                Nxy_My[1] += sg.cell_factors[cell_id]*(origin_loads[0]
                      * cell.x[1, 0] + origin_loads[1]*cell.x[1, 1]
                      + origin_loads[2]*cell.x[1, 2] + origin_loads[3]
                      * cell.x[1, 3])
            # Calculate Mx, My and Mxy
            # Contribution from the cells
            Nx_Mx_Mxy = np.dot(sg.uk_inv, np.dot(-1*sg.vk,Nxy_My))
            # Calculate loads at the 3 recovery points
            n = np.array([-0.5*sg.bk,0.0,+0.5*sg.bk])
            for r in range(3): # cycle the recovery points
                rn = np.array([[1.0, 0.0, n[r], 0.0], \
                               [0.0, 1.0, 0.0, 0.0],\
                               [0.0, 0.0, 0.0, -2.0]])
                Nx_Mx_Mxy_recovery = np.dot(
                    np.dot(np.dot(sg.uk_inv,np.dot(rn,sg.rk)),self.w),
                    origin_loads
                    )
                Nx_k[s,r] = Nx_Mx_Mxy[0] + Nx_Mx_Mxy_recovery[0]
                Mx_k[s,r] = Nx_Mx_Mxy[1] + Nx_Mx_Mxy_recovery[1]
                Mxy_k[s,r] = Nx_Mx_Mxy[2] + Nx_Mx_Mxy_recovery[2]
                Nxy_k[s,r] = Nxy_My[0]
                My_k[s,r] = Nxy_My[1]
                # Shear flow calculations
                if unit_load > 3:
                    # Setup the derivative vectors and matrix
                    d = np.zeros((4))
                    d[1] = Vz_s
                    d[2] = Vy_s
                    d_a = np.zeros((3,4))
                    d_a = np.dot(np.dot(sg.uk_inv, np.dot(rn, sg.rk)), self.w)
                    d_a += sg.sk
                    d = np.dot(d_a, d)
                    # dNx/dx at recovery point r:
                    dqop[s,r] = d[0]
        # Calculate contributions from Vy and Vz
        if unit_load > 3:
            Nxy_k, My_k = self._calculate_shear_contributions(
                    pt_Nxy_contribution, dqop, Nxy_k, My_k)
            self._calculate_shear_center(Nxy_k, unit_load)
        # Fill the segment unitary load dataframe
        x1 = 0.0
        x2 = 0.5
        x3 = 1.0
        for s, (sg_id, sg) in enumerate(self.segments.items()):
            a, b, c = _quadratic_poly_coef_from_3_values(
                      x1, x2, x3, Nx_k[s, 0], Nx_k[s, 1], Nx_k[s, 2])
            u_sgs.loc[sg_id,('Nx','C1')] = b
            u_sgs.loc[sg_id,('Nx','C0')] = c

            a, b, c = _quadratic_poly_coef_from_3_values(
                      x1, x2, x3, Nxy_k[s, 0], Nxy_k[s, 1], Nxy_k[s, 2])
            u_sgs.loc[sg_id,('Nxy','C2')] = a
            u_sgs.loc[sg_id,('Nxy','C1')] = b
            u_sgs.loc[sg_id,('Nxy','C0')] = c

            a, b, c = _quadratic_poly_coef_from_3_values(
                      x1, x2, x3, Mx_k[s, 0], Mx_k[s, 1], Mx_k[s, 2])
            u_sgs.loc[sg_id,('Mx','C1')] = b
            u_sgs.loc[sg_id,('Mx','C0')] = c

            a, b, c = _quadratic_poly_coef_from_3_values(
                      x1, x2, x3, My_k[s, 0], My_k[s, 1], My_k[s, 2])
            u_sgs.loc[sg_id,('My','C1')] = b
            u_sgs.loc[sg_id,('My','C0')] = c

            a, b, c = _quadratic_poly_coef_from_3_values(
                      x1, x2, x3, Mxy_k[s, 0], Mxy_k[s, 1], Mxy_k[s, 2])
            u_sgs.loc[sg_id,('Mxy','C1')] = b
            u_sgs.loc[sg_id,('Mxy','C0')] = c
        return u_sgs, u_pts


    def calculate_internal_loads(self):
        """
        Calculates internal loads for all load cases in the loads dictionary.

        Results are loaded into two pandas dataframes: self.sgs_int_lds_df and
        self.pts_int_lds_df.
        Segment loads are represented as quadratic equations by outputting the
        coefficients C2, C1 and C0, where Load = C2*n**2 + C1*n + C0. "n" is
        the location in the segment length varying from 0.0 (point A) to 1.0
        (point B). Maximum and minimum segment values and their associated
        locations (0.0 - 1.0) inside the segment are also provided, along with
        the segment average and total (integrated) load.
        """
        i_load = ['Nx','Nxy','Mx','My','Mxy']
        r_dict={}
        sg_ids = []
        load_ids = []
        pt_load_ids = []
        pt_df_lst = []
        #r_dict = {'Segment_Id': [], 'Load_Id': []}
        for i_l in i_load:
            if i_l == 'Nxy':
                r_dict[(i_l,'C2')] = []
            r_dict[(i_l,'C1')] = []
            r_dict[(i_l,'C0')] = []
            if i_l == 'Nx':
                r_dict[(i_l,'Nb')] = []
            r_dict[(i_l,'Max')] = []
            r_dict[(i_l,'n_Max')] = []
            r_dict[(i_l,'Min')] = []
            r_dict[(i_l,'n_Min')] = []
            r_dict[(i_l,'Avg')] = []
            r_dict[(i_l,'Total')] = []
        for f_id, f in self.loads.items():
            # Bring all loads to the centroid and shear center
            Px_c = f.Px_c+f.Px
            My = f.My + f.Px*(f.zp-self.zc)
            Mz = f.Mz + f.Px*(f.yp-self.yc)
            Tx = f.Tx + f.Vz*(f.yv-self.ys) - f.Vy*(f.zv-self.zs)
            Vy_s = f.Vy_s+f.Vy
            Vz_s = f.Vz_s+f.Vz
            # Segments dataframe for the load case
            f_df = (self.u_Px[:]*Px_c + self.u_My[:]*My + self.u_Mz[:]*Mz
                     + self.u_Tx[:]*Tx + self.u_Vy[:]*Vy_s + self.u_Vz[:]*Vz_s)
            # Points dataframe for the load case
            f_p_df = (self.u_Px_pt[:]*Px_c + self.u_My_pt[:]*My
                      + self.u_Mz_pt[:]*Mz + self.u_Tx_pt[:]*Tx
                      + self.u_Vy_pt[:]*Vy_s + self.u_Vz_pt[:]*Vz_s)
            pt_df_lst.append(f_p_df)
            pt_load_ids += len(self.points)*[f_id]
            for sg_id, sg in self.segments.items():
                sg_ids.append(sg_id)
                load_ids.append(f_id)
                for i_l in i_load:
                    if i_l == 'Nxy':
                        a = f_df.loc[sg_id,(i_l,'C2')]
                        r_dict[(i_l,'C2')].append(a)
                    else:
                        a = 0.0
                    b = f_df.loc[sg_id,(i_l,'C1')]
                    c = f_df.loc[sg_id,(i_l,'C0')]
                    max_, n_max, min_, n_min = _max_min_quad_n_eq(
                            a, b, c)
                    if i_l == 'Nx':
                        Nb = abs(c-(a+b+c))/2
                        r_dict[(i_l,'Nb')].append(Nb)
                    r_dict[(i_l,'C1')].append(b)
                    r_dict[(i_l,'C0')].append(c)
                    r_dict[(i_l,'Max')].append(max_)
                    r_dict[(i_l,'n_Max')].append(n_max)
                    r_dict[(i_l,'Min')].append(min_)
                    r_dict[(i_l,'n_Min')].append(n_min)
                    x1 = 0
                    x2 = 0.5 * sg.bk
                    x3 = sg.bk
                    v1 = c
                    v2 = a*0.25 + b*0.5 + c
                    v3 = a+b+c
                    a, b, c = _quadratic_poly_coef_from_3_values(
                              x1, x2, x3, v1, v2, v3)
                    total = a*sg.bk**3 / 3 + b*sg.bk**2 / 2 + c*sg.bk
                    avg = total/sg.bk
                    r_dict[(i_l,'Avg')].append(avg)
                    r_dict[(i_l,'Total')].append(total)
        self.sgs_int_lds_df = pd.DataFrame(r_dict)
        self.sgs_int_lds_df.insert(0, 'Segment_Id', sg_ids)
        self.sgs_int_lds_df.insert(1, 'Load_Id', load_ids)
        self.pts_int_lds_df = pd.concat(pt_df_lst)
        self.pts_int_lds_df.insert(0, 'Load_Id', pt_load_ids)
        self.pts_int_lds_df = self.pts_int_lds_df.reset_index()


    def print_internal_loads(self, break_columns=True):
        """
        Prints to the console segment and point internal loads for all load
        cases in the loads dictionary.

        Warning
        -------
        This method outputs a significant amount of data per load
        case and segment. Depending on your number of segments and load
        cases, manipulate the data stored in self.sgs_int_lds_df and
        self.pts_int_lds_df using pandas methods directly.

        Parameters
        ----------
        break_columns : bool, default True
            Of the form {int : abdbeam.Material}
        """
        print('')
        print('Internal Loads for Segments')
        print('---------------------------')
        print('')
        if break_columns:
            with pd.option_context('display.max_rows', None,
                                   'display.max_columns', None):
                print(self.sgs_int_lds_df)
        else:
            print(self.sgs_int_lds_df.to_string())
        if ((self.pts_int_lds_df['Px'] != 0).any() or
           (self.pts_int_lds_df['Tx'] != 0).any()):
            print('')
            print('Internal Loads for Points')
            print('-------------------------')
            print('')
            if break_columns:
                with pd.option_context('display.max_rows', None,
                                       'display.max_columns', None):
                    print(self.pts_int_lds_df)
            else:
                print(self.pts_int_lds_df.to_string())


    def _calculate_shear_contributions(self, pt_Nxy_contribution, dqop, Nxy_k,
                                       My_k):
        # Nxy at a location is qop = -integral(dNx/dx) up to this point
        for s, (sg_id, sg) in enumerate(self.segments.items()):
            dqop[s, 3] = -0.5*(dqop[s, 0] + dqop[s, 1])*0.5*sg.bk
            dqop[s, 4] = -0.5*(dqop[s, 1] + dqop[s, 2])*0.5*sg.bk

        # Calculate Nxy for open beams with branches
        # Cells are also treated as open branches based on the segments
        # shear_cut atributte
        qop = np.zeros((len(self.segments), 3))
        sg_ids = list(self.segments.keys())
        pt_adj_count = [len(pt.adj_point_ids) for pt in self.points.values()]
        pt_adj_count = dict((pt_id, s) for pt_id, s in zip(self.points.keys(),
                             pt_adj_count))
        pt_acc_shear = dict((pt_id, 0.0) for pt_id in self.points.keys())

        # Calculate the Nxy for all segments with shear_cut = True at recovery
        # points, subtracting 1 from the adjacent point a and b counts
        for s, (sg_id, sg) in enumerate(self.segments.items()):
            if not sg.shear_cut:
                continue
            # calculate the shear from this cut free edge
            # Integrate shear from the sg point b to point a
            # qop[i, 2] is at point b free edge, qop[i, 1] is mid and qop[i,0]
            # is point a.
            # Note: the point pt_Nxy_contribution is not added to point b, as
            # the cut is made immediately after the point
            qop[s, 2] = 0.0
            qop[s, 1] = qop[s, 2] - dqop[s, 4]
            qop[s, 0] = qop[s, 1] - dqop[s, 3]
            # Dump qop[i,1] at point a. Negative means shear added.
            pt_acc_shear[sg.point_a_id] -= qop[s, 0]
            pt_adj_count[sg.point_a_id] -= 1
            pt_adj_count[sg.point_b_id] -= 1
            # Remove the seg_id from the list, since its shear is calculated
            sg_ids.remove(sg_id)

        # calculate the Nxy for the remaining segments at recovery points
        while sg_ids:
            for sg_id in sg_ids:
                idx = list(self.segments.keys()).index(sg_id)
                pt_a_id = self.segments[sg_id].point_a_id
                pt_b_id = self.segments[sg_id].point_b_id
                if pt_adj_count[pt_a_id] == 1:
                    # point a is a free edge
                    qop[idx, 0] = (pt_acc_shear[pt_a_id]
                                  + pt_Nxy_contribution[pt_a_id])
                    qop[idx, 1] = qop[idx, 0] + dqop[idx, 3]
                    qop[idx, 2] = qop[idx, 1] + dqop[idx, 4]
                    pt_acc_shear[pt_b_id] += qop[idx, 2]
                    pt_adj_count[pt_a_id] -= 1
                    pt_adj_count[pt_b_id] -= 1
                    sg_ids.remove(sg_id)
                    break
                elif pt_adj_count[pt_b_id] == 1:
                    # point b is a free edge
                    qop[idx, 2] = (-pt_acc_shear[pt_b_id]
                                  - pt_Nxy_contribution[pt_b_id])
                    qop[idx, 1] = qop[idx, 2] - dqop[idx, 4]
                    qop[idx, 0] = qop[idx, 1] - dqop[idx, 3]
                    pt_acc_shear[pt_a_id] -= qop[idx, 0]
                    pt_adj_count[pt_a_id] -= 1
                    pt_adj_count[pt_b_id] -= 1
                    sg_ids.remove(sg_id)
                    break
        # Calculate the total shear load at the segment based on the integral
        # of the quadratic equation using the 3 points
        Nxy_open_k = np.zeros(len(self.segments))

        for s, (sg_id, sg) in enumerate(self.segments.items()):
            x1 = 0.0
            x2 = sg.bk/2
            x3 = sg.bk
            a, b, c = _quadratic_poly_coef_from_3_values(
                      x1, x2, x3, qop[s, 0], qop[s, 1], qop[s, 2])
            # Calculate the average shear on the segment
            Nxy_open_k[s] = (a*sg.bk**3 / 3 + b*sg.bk**2 / 2 + c*sg.bk)/sg.bk

        # Find the balancing torque
        Nxy_bal_k = np.zeros((len(self.segments)))
        My_bal_k = np.zeros((len(self.segments)))
        if self.cells:
            Nxy_bal_k, My_bal_k = self._calculate_balancing_torque(Nxy_open_k)
        for s, sg_id in enumerate(self.segments):
            for r in range(3):
                Nxy_k[s, r] += qop[s, r] + Nxy_bal_k[s]
                My_k[s, r] += My_bal_k[s]

        return Nxy_k, My_k


    def _calculate_shear_center(self, Nxy_sgs, unit_load):
        Mx = 0.0
        for s, (sg_id, sg) in enumerate(self.segments.items()):
            x1 = 0
            x2 = sg.bk/2
            x3 = sg.bk
            a, b, c = _quadratic_poly_coef_from_3_values(
                      x1, x2, x3, Nxy_sgs[s, 0], Nxy_sgs[s, 1], Nxy_sgs[s, 2])
            # Calculate the total shear load in the segment by its integral
            v = a*sg.bk**3 / 3 + b*sg.bk**2 / 2 + c*sg.bk
            # Calculate moment arms
            ya = self.points[sg.point_a_id].y
            za = self.points[sg.point_a_id].z
            yb = self.points[sg.point_b_id].y
            zb = self.points[sg.point_b_id].z
            arm_y = sg.yk - self.yc
            arm_z = sg.zk - self.zc
            vy = (yb-ya)/sg.bk
            vz = (zb-za)/sg.bk
            fy = v * vy
            fz = v * vz
            # Torsion at centroid
            Mx += arm_y * fz - arm_z * fy
        if unit_load == 4:
            self.zs = -Mx + self.zc
        elif unit_load == 5:
            self.ys = Mx + self.yc


    def _calculate_balancing_torque(self, Nxy_open_sgs):
        m = np.zeros((2*len(self.cells), 2*len(self.cells)))
        v = np.zeros((2*len(self.cells)))
        Nxy_balancing = np.zeros((len(self.segments)))
        My_balancing = np.zeros((len(self.segments)))
        ncells = len(self.cells)
        for c, c_id in enumerate(self.cells.keys()):
           for s, (sg_id, sg) in enumerate(self.segments.items()):
                f = sg.cell_factors[c_id]
                # 1st equation
                m[c, c] += abs(f)*sg.fk[0,0]
                m[c, c+ncells] += abs(f)*sg.fk[0,1]
                # 2nd equation
                m[c+ncells, c] += abs(f)*sg.fk[1,0]
                m[c+ncells, c+ncells] += abs(f)*sg.fk[1,1]
                for cc, cc_id in enumerate(self.cells.keys()):
                    ff = sg.cell_factors[cc_id]
                    if cc == c:
                        continue
                    # qm_c
                    m[c, cc] -= abs(f)*abs(ff)*sg.fk[0,0]
                    m[c+ncells, cc] -= abs(f)*abs(ff)*sg.fk[1,0]
                    # Mm_c
                    m[c, cc+ncells] -= abs(f)*abs(ff)*sg.fk[0,1]
                    m[c+ncells, cc+ncells] -= abs(f)*abs(ff)*sg.fk[1,1]
                # q_op
                v[c] -= f*sg.fk[0,0] * Nxy_open_sgs[s]
                v[c+ncells] -= f*sg.fk[1,0] * Nxy_open_sgs[s]
        # Add a small number to the diagonal to assure m can be inverted
        # (needed for sections with connectors only)
        non_zero_diag = np.zeros_like(m)
        np.fill_diagonal(non_zero_diag, 1e-99)
        m += non_zero_diag
        tmp_m = np.dot(np.linalg.inv(m), v)
        for c, c_id in enumerate(self.cells.keys()):
           for s, (sg_id, sg) in enumerate(self.segments.items()):
                f = sg.cell_factors[c_id]
                Nxy_balancing[s] += f*tmp_m[c]
                My_balancing[s] += f*tmp_m[c+ncells]
        return Nxy_balancing, My_balancing


    def _min_clockwise_angle_adj_point_id(self, pivot_pt_id, from_pt_id,
                                          attempt_pt_ids):
        output_min_angle = math.inf
        x1 = self.points[from_pt_id].y
        y1 = self.points[from_pt_id].z
        x2 = self.points[pivot_pt_id].y
        y2 = self.points[pivot_pt_id].z
        out_pt_id = -1
        # cycle adjacent points
        for adj_pt_id in self.points[pivot_pt_id].adj_point_ids:
            if adj_pt_id != from_pt_id and adj_pt_id not in attempt_pt_ids:
                x3 = self.points[adj_pt_id].y
                y3 = self.points[adj_pt_id].z
                angle = _clockwise_angle_from_3_points(x1, y1, x2,
                                                               y2, x3, y3)
                if angle < output_min_angle:
                    output_min_angle = angle
                    out_pt_id = adj_pt_id
        return out_pt_id


class Point:
    """
    A cross section point in the section (Y,Z) coordinate system, optionally
    having an EA and GJ associated to it.

    Attributes
    ----------
    y: float
        The Y location of the point.
    z: float
        The Z location of the point.
    EA: float
        The axial stiffness of the point.
    GJ: float
        The torsional stiffness of the point.
    description: str
        The point description.

    Examples
    --------
    Create two points and associate them to a section:

    .. code-block:: python

        import abdbeam as ab
        sc = ab.Section()
        pts = dict()
        pts[1] = ab.Point(0.5, 1.0, 10000000.0, 4500000.0, 'Stringer 105')
        pts[2] = ab.Point(0.0, 0.0)
        sc.points = pts

    """


    def __init__(self, y=0.0, z=0.0, EA=0.0, GJ=0.0, description=''):
        """
        Creates a Point instance.

        Parameters
        ----------
        y: float, default 0.0
            The Y location of the point.
        z: float, default 0.0
            The Z location of the point.
        EA: float, default 0.0
            The axial stiffness of the point.
        GJ: float, default 0.0
            The torsional stiffness of the point.
        description: str, default ''
            The point description.
        """
        self.y = y
        self.z = z
        self.EA = EA
        self.GJ = GJ
        self.description = description
        self.adj_point_ids = set() # filled by Section class instances


    def __repr__(self):
        return (('{}({}, {}, {}, {}, {})'.format(self.__class__.__name__,
                self.y, self.z, self.EA, self.GJ, self.description)))


class Segment:
    """
    Class that defines a section segment and calculates its properties.

    Attributes
    ----------
    point_a_id : int
        The first point id of the segment.
    point_b_id : int
        The second point id of the segment.
    material_id : int
        The material id of the segment.
    description : str
        The segment description.
    bk : float
        The segment length.
    t : float
        The segment thickness (based on material data).


    Methods
    -------
    calculate_properties(points, materials)
        Prints a summary of the section properties.


    Examples
    --------
    Creating 3 segments and associating them to a section.

    .. code-block:: python

        import abdbeam as ab
        sc = ab.Section()
        sgs = dict()
        sgs[1] = ab.Segment(1,2,1)
        sgs[2] = ab.Segment(2,3,1)
        sgs[3] = ab.Segment(3,4,1)
        sc.segments = sgs
    """


    def __init__(self, point_a_id, point_b_id, material_id, description=''):
        """
        Instantiates a Segment object.

        Attributes
        ----------
        point_a_id : int
            The first point id of the segment.
        point_b_id : int
            The second point id of the segment.
        material_id : int
            The material id of the segment.
        description : str, default = ''
            The segment description.
        """
        self.point_a_id = point_a_id
        self.point_b_id = point_b_id
        self.material_id = material_id
        self.description = description
        self.adj_cell_id_list=[]
        self.tmp_adj_cell_id_list=[]
        self.wk = np.zeros((4,4))
        self.fk = np.zeros((2,2))
        self.ik = np.zeros((2,4))
        self.bk = 0.0
        self.t = 0.0
        self.uk_inv = np.zeros((3,3))
        self.wk_inv = np.zeros((4,4))
        self.rk = np.zeros((4,4))
        self.density = 0.0
        self.shear_cut = False
        self.cell_factors = dict()
        self.sk = np.zeros((3,4))
        self.vk = np.zeros((3,2))
        self.yk = 0.0
        self.zk = 0.0


    def __repr__(self):
        return ('{}({}, {}, {}, {})'.format(self.__class__.__name__,
                self.point_a_id, self.point_b_id, self.material_id,
                self.description))


    def calculate_properties(self, points, materials):
        """
        Calculates the segment properties.

        This method is normally called by a Section object.

        Parameters
        ----------
        points :  dict
            Of the form {int : abdbeam.Point}.
        materials :  dict
            Of the form {int : abdbeam.Material}.
        """
        ya = points[self.point_a_id].y
        za = points[self.point_a_id].z
        yb = points[self.point_b_id].y
        zb = points[self.point_b_id].z
        mat = materials[self.material_id]
        abd_c = mat.abd_c
        deltay = yb - ya
        deltaz = zb - za
        bk = (deltay**2 + deltaz**2)**0.5
        self.bk = bk
        self.t = mat.t
        self.yk = (ya+yb) / 2
        self.zk = (za+zb) / 2
        cosalpha = deltay/bk
        sinalpha = deltaz/bk
        self.rk = np.array([[1, self.zk, self.yk, 0],
                            [0, cosalpha, -sinalpha, 0],
                            [0, sinalpha, cosalpha, 0],
                            [0, 0, 0, 1]])
        aik =np.array([[abd_c[0,0], abd_c[0,3], abd_c[0,5]],
                       [abd_c[0,3], abd_c[3,3], abd_c[3,5]],
                       [abd_c[0,5], abd_c[3,5], abd_c[5,5]]])
        aik_inv = np.linalg.inv(aik)
        self.wk =(1/bk)*np.array(
           [[abd_c[0,0], abd_c[0,3], 0, -0.5*abd_c[0,5]],
            [abd_c[0,3], abd_c[3,3], 0, -0.5 *abd_c[3,5]],
            [0, 0, 12 / (aik_inv[0,0] * bk**2), 0],
            [-0.5*abd_c[0,5], -0.5*abd_c[3,5], 0, 0.25*abd_c[5,5]]]
            )
        self.wk_inv = np.linalg.inv(self.wk)
        #Ik
        m_tmp = np.array(
            [[abd_c[0,2], abd_c[0,5], 0, -0.5*abd_c[2,5]],
             [abd_c[0,4], abd_c[3,4], 0, -0.5*abd_c[4,5]]]
             )
        self.ik =np.dot(m_tmp, np.dot(self.wk_inv, self.rk))
        #Fk
        m_tmp_2 = bk * np.array([[abd_c[2,2], abd_c[1,5]],
                                 [abd_c[1,5], abd_c[4,4]]])
        self.fk = (m_tmp_2 -1 * np.dot(m_tmp, np.dot(self.wk_inv,
                   np.transpose(m_tmp))))
        uk =np.array([[abd_c[0,0], abd_c[0,3], abd_c[0,5]],
                      [abd_c[0,3], abd_c[3,3], abd_c[3,5]],
                      [abd_c[0,5], abd_c[3,5], abd_c[5,5]]])
        self.uk_inv = np.linalg.inv(uk)
        self.vk = np.array([[abd_c[0,2], abd_c[0,4]],
                       [abd_c[0,5], abd_c[3,4]],
                       [abd_c[2,5], abd_c[4,5]]])


class Cell:
    """
    Class instantiated internally by a abdbeam.Section object containing
    attributes and utilities related to a detected cell.

    Attributes
    ----------
    point_ids : list
        A list of point ids associated to the cell. Of the form [int].
    segment_ids : list
        A list of segment ids associated to the cell. Of the form [int].
    area : float
        The cell area.
    """


    def __init__(self):
        self.point_ids = []
        self.segment_ids = []
        self.area = 0.0
        self.x = np.zeros((2,4), float)


    def __repr__(self):
        return ('{}()'.format(self.__class__.__name__))


    def _calculate_area(self, points_dict):
        self.area = 0.0
        points = self.point_ids + [self.point_ids[0]]
        for first, second in zip(points, points[1:]):
            self.area += (points_dict[second].y*points_dict[first].z
                          - points_dict[second].z*points_dict[first].y)
        self.area = abs(self.area/2)


    def _is_closed_polygon(self, points_dict):
        n_segs = len(self.point_ids)
        angle_sum = 0.0
        points = self.point_ids + [self.point_ids[0], self.point_ids[1]]
        for first, second, third in zip(points, points[1:], points[2:]):
            angle_sum += _clockwise_angle_from_3_points(
                        points_dict[first].y, points_dict[first].z,
                        points_dict[second].y, points_dict[second].z,
                        points_dict[third].y, points_dict[third].z
                        )
        if (180*(n_segs-2)*0.99) < angle_sum < (180*(n_segs-2)*1.01):
            return True
        else:
            return False


class Load:
    """
    A single section load case.

    Attributes
    ----------
    Px_c : float
        The axial load at the centroid of the cross section. Positive sign
        induces tension in the cross section.
    My : float
        The moment around the Y axis. Positive sign induces tension in the
        positive yz quadrant of the beam cross section.
    Mz : float
        The moment around the Z axis. Positive sign induces tension in the
        positive yz quadrant of the beam cross section.
    Tx : float
        The torque around the X axis. Positive sign is counterclockwise.
    Vy_s: float
        The shear force oriented with the Y axis at the shear center.
    Vz_s: float
        The shear force oriented with the section Z axis at the shear.
    Px: float
        The axial force located at (yp, zp). Positive sign induces tension in
        the cross section.
    yp: float
        The Y axis location of the Px axial force.
    zp: float
        The Z axis location of the Px axial force.
    Vy: float
        The shear force oriented with the Y axis at zv.
    Vz: float
        The shear force oriented with the Z axis at yv.
    yv: float
        The Y axis location of the Vz shear force.
    zv: float
        The Z axis location of the Vy shear force.

    Examples
    --------
    Creating 3 load cases and associating them to a section:

    .. code-block:: python

        import abdbeam as ab
        sc = ab.Section()
        Lds = dict()
        Lds[101] = ab.Load(My=5e6)
        Lds[102] = ab.Load(Tx=250000, Vz=5000.0)
        Lds[103] = ab.Load(0, 0, 0, 0, 0, 1000.0)
        sc.loads = Lds
    """


    def __init__(self, Px_c=0.0, My=0.0, Mz=0.0, Tx=0.0, Vy_s=0.0, Vz_s=0.0,
             Px=0.0, yp=0.0, zp=0.0, Vy=0.0, Vz=0.0, yv=0.0, zv=0.0):
        """
        Instantiates a section Load.

        Attributes
        ----------
        Px_c : float, default 0.0
            The axial load at the centroid of the cross section. Positive sign
            induces tension in the cross section.
        My : float, default 0.0
            The moment around the Y axis. Positive sign induces tension in the
            positive yz quadrant of the beam cross section.
        Mz : float, default 0.0
            The moment around the Z axis. Positive sign induces tension in the
            positive yz quadrant of the beam cross section.
        Tx : float, default 0.0
            The torque around the X axis. Positive sign is counterclockwise.
        Vy_s: float, default 0.0
            The shear force oriented with the Y axis at the shear center.
        Vz_s: float, default 0.0
            The shear force oriented with the section Z axis at the shear.
        Px: float, default 0.0
            The axial force located at (yp, zp). Positive sign induces tension
            in the cross section.
        yp: float, default 0.0
            The Y axis location of the Px axial force.
        zp: float, default 0.0
            The Z axis location of the Px axial force.
        Vy: float, default 0.0
            The shear force oriented with the Y axis at zv.
        Vz: float, default 0.0
            The shear force oriented with the Z axis at yv.
        yv: float, default 0.0
            The Y axis location of the Vz shear force.
        zv: float, default 0.0
            The Z axis location of the Vy shear force.
        """
        self.Px_c = Px_c
        self.My = My
        self.Mz = Mz
        self.Tx = Tx
        self.Vy_s = Vy_s
        self.Vz_s = Vz_s
        self.Px = Px
        self.yp = yp
        self.zp = zp
        self.Vy = Vy
        self.Vz = Vz
        self.yv = yv
        self.zv = zv


    def __repr__(self):
        return ('{}({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})'
                .format(self.__class__.__name__, self.Px_c, self.My, self.Mz,
                self.Tx, self.Vy_s, self.Vz_s, self.Px, self.yp, self.zp,
                self.Vy, self.Vz, self.yv, self.zv))


def _quadratic_poly_coef_from_3_values(x1, x2, x3, n1, n2, n3):
    """
    Returns the coeficients a, b and c of a quadratic polynomial of the form
    n = a*x**2 * b*x + c from 3 known points.

    Parameters
    ----------
    x1, x2, x3 : float
        The positions of the values.
    n1, n2, n3 : float
        The values at positions x1, x2 and x3 respectively.

    Returns
    -------
    a, b, c : float
        The coefficients.
    """
    a = (((n2 - n1) * (x1 - x3) + (n3 - n1) * (x2 - x1)) /
        ((x1 - x3) * (x2**2 - x1**2) + (x2 - x1) * (x3**2 - x1**2)))
    b = ((n2 - n1) - a * (x2**2 - x1**2)) / (x2 - x1)
    c = n1 - a * x1**2 - b * x1
    return a,b,c


def _clockwise_angle_from_3_points(x1, y1, x2, y2, x3, y3):
    ref_start_angle = _angle_from_x_axis(x2 - x1, y2 - y1, 0.0)
    angle = 180 - _angle_from_x_axis(x3 - x2, y3 - y2,
                                             ref_start_angle )
    angle = _fix_angle_within_360(angle)
    angle = 360 - angle
    return angle


def _angle_from_x_axis(x , y, ref_angle):
    if x == 0 and y < 0:
        local_angle = 270
    elif x == 0 and y > 0:
        local_angle = 90
    elif x > 0 and y == 0:
        local_angle = 0
    elif x < 0 and y == 0:
        local_angle = 180
    elif x == 0 and y == 0:
        local_angle = 0
    else:
        local_angle = math.degrees(math.atan(abs(y)/abs(x)))
        if x < 0 and y > 0:
            local_angle = 180 - local_angle
        elif x < 0 and y < 0:
            local_angle = 180 + local_angle
        elif x > 0 and y < 0 :
            local_angle = 360 - local_angle
    ref_angle = _fix_angle_within_360(ref_angle)
    angle = _fix_angle_within_360(local_angle - ref_angle)
    return angle


def _fix_angle_within_360(angle):
    if angle < 0:
        angle = 360 - abs(angle)
    elif angle > 360:
        angle = abs(angle) - 360
    return angle


def _max_min_quad_n_eq(a, b, c):
    # Since the quadratic equation has x between 0 and 1:
    f1 = c
    f3 = a + b + c
    f2 = f1
    if a != 0:
        if 0.0 < -b/(2*a) < 1.0:
            f2 = a*(-b/(2*a))**2 + b*(-b/(2*a)) + c
    max_ = f1
    n_max = 0.0
    min_ = f1
    n_min = 0.0
    if f3 > max_:
        max_ = f3
        n_max = 1.0
    if f3 < min_:
        min_ = f3
        n_min = 1.0
    if f2 > max_:
        max_ = f2
        n_max = -b/(2*a)
    if f2 < min_:
        min_ = f2
        n_min = -b/(2*a)
    return max_, n_max, min_, n_min


def _no_overlap(y, z, y1, z1, y2, z2, pt_id, pt_1_id, pt_2_id):
    """
    Checks if a point (y,z) is inside a line given by (y1,z1) and
    (y2,z2). Also check if the point id matches with the ids from the line.

    Returns
    -------
    Boolean
    """
    if (y < min(y1, y2) or y > max(y1, y2) or z < min(z1, z2) or
            z > max(z1, z2)):
        return True
    elif pt_id == pt_1_id or pt_id == pt_2_id:
        return True
    else:
        return False


def _no_intersec_check(y1, z1, y2, z2, y3, z3, y4, z4, pt_1_id, pt_2_id,
                        pt_3_id, pt_4_id):
    if ((y1*z3 - y3*z1 - y1*z4 - y2*z3 + y3*z2 + y4*z1 + y2*z4 - y4*z2)
            == 0):
        return True
    y = ((y1*y3*z2 - y2*y3*z1 - y1*y4*z2 + y2*y4*z1 - y1*y3*z4 + y1*y4*z3
          + y2*y3*z4 - y2*y4*z3) / (y1*z3 - y3*z1 - y1*z4 - y2*z3 + y3*z2
          + y4*z1 + y2*z4 - y4*z2))
    z = ((y1*z2*z3 - y2*z1*z3 - y1*z2*z4 + y2*z1*z4 - y3*z1*z4 + y4*z1*z3
          + y3*z2*z4 - y4*z2*z3) / (y1*z3 - y3*z1 - y1*z4 - y2*z3 + y3*z2
          + y4*z1 + y2*z4 - y4*z2))
    if (y < min(y1, y2) or y < min(y3, y4) or y > max(y1, y2) or
        y > max(y3, y4) or z < min(z1, z2) or z < min(z3, z4) or
        z > max(z1, z2) or z > max(z3, z4)):
        return True
    if (pt_1_id == pt_3_id or pt_1_id == pt_4_id or pt_2_id == pt_3_id
        or pt_2_id == pt_4_id):
        return True
    return False
