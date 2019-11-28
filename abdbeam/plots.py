# -*- coding: utf-8 -*-
"""
Module containing methods to plot cross sections and their internal loads.
"""

import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.patches import Polygon
import math
import numpy as np
from abdbeam.core import _clockwise_angle_from_3_points
import warnings


def plot_section(section, segment_coord=False, thickness=True, mid_plane=True,
                 top_bottom=False, centroid=True, shear_center=True,
                 origin=True, princ_dir=True, show_axis=True, prop_color='r',
                 pt_size=4, filter_sgs=[], plot_sgs=[], legend=True, title='',
                 figsize=(6.4, 4.8), dpi=80):
    """
    Uses matplolib to plot the section geometry and its properties (centroid,
    shear center and principal axis).

    Note
    ----
    Section properties need to be calculated using the method
    abdbeam.Section.calculate_properties() before using this function.

    Parameters
    ----------
    section : abdbeam.Section
        The section object to be plotted.
    segment_coord : bool, default False
        If True, will plot the segments local coordinate systems.
    thickness : bool, default True
        If True, will plot the segments thickness.
    mid_plane : bool, default True
        If True, will plot the segments mid-plane.
    top_bottom : bool, default False
        If True and thickness is also True, will identify the bottom side of
        the material with the red color and the top side with the green color.
    centroid : bool, default True
        If True, will plot the centroid location with an 'o' marker.
    shear_center : bool, default True
        If True, will plot the shear center location with a 'x' marker.
    origin : bool, default True
        If True, will plot the section origin location with a '+' marker.
    princ_dir : bool, default True
        If True, will plot the moment of inertia principal axes.
    show_axis : bool, default True
        If True, will show plot dimensions.
    prop_color : string, default 'r'
        The matplotlib color name to be used when plotting centroid, shear
        center and principal axes.
    pt_size : int, default 4
        The size in pixels of the marker. Booms (points with EAs and GJs) will
        have 2 times this size.
    filter_sgs : list, default []
        The list of segment ids that will not be plotted. Of the form [int].
        Will also respect the filter imposed by the plot_sgs parameter.
    plot_sgs : list, default []
        The list containing the only segment ids that will be plotted. Of the
        form [int]. If left empty will plot all segments. Will also respect the
        filter imposed by the filter_sgs parameter.
    legend : bool, default True
        If True, will show a legend for the centroid, shear center and
        principal axis.
    title : str, default ''
        The title to be added at the top of the figure.
    figsize : tuple, default (6.4, 4.8)
        Width and height of the figure in inches. Of the form (float, float).
    dpi : integer, default 80
        The resolution of the figure.

    Examples
    --------
    The example below creates a "C" section and plots it. The optional
    attribute prop_color is changed to purple usinh the HTML color code
    #800080:

    .. code-block:: python

        import abdbeam as ab
        sc = ab.Section()
        mts = dict()
        mts[1] = ab.Isotropic(0.08, 10600000, 0.33)
        pts = dict()
        pts[1] = ab.Point(0, -1.5)
        pts[2] = ab.Point(-1, -1.5)
        pts[3] = ab.Point(-1, 1.5)
        pts[4] = ab.Point(0.5, 1.5)
        sgs = dict()
        sgs[1] = ab.Segment(1,2,1)
        sgs[2] = ab.Segment(2,3,1)
        sgs[3] = ab.Segment(3,4,1)
        sc.materials = mts
        sc.points = pts
        sc.segments = sgs
        sc.calculate_properties()
        ab.plot_section(sc, prop_color='#800080')
    """
    fig, axs = plt.subplots(1, 1, squeeze=False)
    _plot_section_to_ax(section, axs[0,0], segment_coord, thickness, mid_plane,
                top_bottom, centroid, shear_center, origin, princ_dir,
                show_axis, prop_color, pt_size, filter_sgs, plot_sgs, legend,
                title)
    fig.set_size_inches(figsize)
    fig.set_dpi(dpi)
    plt.ion()
    plt.show()


def _plot_section_to_ax(section, ax, segment_coord=True, thickness=True,
                mid_plane=True, top_bottom=True, centroid=True,
                shear_center=True, origin=True, princ_dir=True, show_axis=True,
                prop_color='r', pt_size=4, filter_sgs=[], plot_sgs=[],
                legend=True, title=''):
    sc = section
    ax.axis('equal')
    for sg_id, sg in sc.segments.items():
        if sg_id in filter_sgs: continue
        if len(plot_sgs) > 0:
            if sg_id not in plot_sgs: continue
        # Extract point locations
        ya = sc.points[sg.point_a_id].y
        za = sc.points[sg.point_a_id].z
        yb = sc.points[sg.point_b_id].y
        zb = sc.points[sg.point_b_id].z
        #The segment rotation
        angle = _clockwise_angle_from_3_points(ya+1,za,ya,za,yb,zb)
        base = ax.transData
        rot = transforms.Affine2D().rotate_deg_around(ya, za, angle)
        # Plot Segment mid-plane line
        if mid_plane:
            if thickness:
                lstyle = '--'
            else:
                lstyle = '-'
            ax.plot([ya, yb], [za, zb], marker='o', markersize=pt_size,
                     linestyle=lstyle, color='k', linewidth=1)
        else:
            ax.plot([ya], [za], marker='o', markersize=pt_size, color="k")
            ax.plot([yb], [zb], marker='o', markersize=pt_size, color="k")
        # Plot segment direction with a 25% segment length
        if segment_coord:
            y = np.array([ya+sg.bk*0.5, ya+sg.bk*0.5, ya+sg.bk*0.65])
            z = np.array([za+sg.bk*0.15, za, za])
            ax.plot(y, z, linestyle='-', color='r',
                     linewidth=1, transform = rot + base)
        #Set top/bottom colors
        if top_bottom:
            top_color = 'g'
            bot_color = 'r'
        else:
            top_color = 'k'
            bot_color = 'k'
        # Plot thickness
        if thickness == True:
            y = np.array([0, 0, sg.bk, sg.bk])
            z = np.array([0, -sg.t/2, -sg.t/2, 0])
            ax.plot(y + ya, z + za, linestyle='-', color=bot_color,
                     linewidth=1, transform = rot + base)
            y = np.array([0, 0, sg.bk, sg.bk])
            z = np.array([0, +sg.t/2, +sg.t/2, 0])
            ax.plot(y + ya, z + za, linestyle='-', color=top_color,
                     linewidth=1, transform = rot + base)
        # If point is a boom, add marker with double the size:
        if sc.points[sg.point_a_id].EA > 0 or sc.points[sg.point_a_id].GJ > 0:
            ax.scatter([ya], [za], s=20*pt_size, facecolors='none',
                       edgecolors='k')
            #plt.plot([ya], [za], marker='o', markersize=10, color="g")
        if sc.points[sg.point_b_id].EA > 0 or sc.points[sg.point_b_id].GJ > 0:
            ax.scatter([yb], [zb], s=20*pt_size, facecolors='none',
                       edgecolors='k')
            #plt.plot([yb], [zb], marker='o', markersize=10, color="g")
        # Plot origin
    if origin:
        ax.plot([0], [0], marker='+', markersize=20, color='k')
    # Plot centroid
    if centroid == True:
        ax.plot([sc.yc], [sc.zc], marker='o', markersize=5,
                 color=prop_color, label='Centroid', linestyle="None")
    # Plot shear center
    if shear_center:
        ax.plot([sc.ys], [sc.zs], marker='x', markersize=8,
                 color=prop_color, label='Shear Center', linestyle="None")
    if type(title) == str and title != '':
        ax.set_title(title)
    if not show_axis:
        ax.axis('off')
    #Fix the axis limits at this point
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    max_axis = max(abs(x_limits[1]-x_limits[0]), abs(y_limits[1]
                      -y_limits[0]))
    # plot principal directions
    if princ_dir:
        angle = sc.principal_axis_angle
        rot = transforms.Affine2D().rotate_deg_around(sc.yc, sc.zc, angle)
        y = np.array([sc.yc - 2*max_axis, sc.yc + 2*max_axis])
        z = np.array([sc.zc, sc.zc])
        ax.plot(y, z, linestyle='--', color=prop_color,
                 linewidth=1, transform = rot + base, label='Princ. Axis')
        y = np.array([sc.yc, sc.yc])
        z = np.array([sc.zc - 2*max_axis, sc.zc + 2*max_axis])
        ax.plot(y, z, linestyle='--', color=prop_color,
                 linewidth=1, transform = rot + base)
    # restore axis limits
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    if legend:
        ax.legend()


def plot_section_loads(section, load_id, int_load_list=['Nx', 'Nxy', 'Mx',
            'My', 'Mxy'], title_list=[], thickness=True, pt_size=4,
            segment_contour=True, diagram=True, diagram_contour=False,
            diagram_alpha=0.15, diagram_scale=1.0, diagram_factor_list=[],
            contour_color='jet_r', contour_levels=10, filter_sgs=[],
            plot_sgs=[], no_result_sgs=[], result_sgs=[],
            figsize=(6.4, 4.8), dpi=80):
    """
    Deprecated method. Use plot_section_results method instead.
    """
    msg = "".join((
            "The 'plot_section_loads' method is deprecated in favor of",
            " 'plot_section_results'.\n",
            "Backward compatibility will be droped in package version 2."))
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    plot_section_results(section, load_id, int_load_list,
                title_list, thickness, pt_size, segment_contour, diagram,
                diagram_contour, diagram_alpha, diagram_scale,
                diagram_factor_list, contour_color, contour_levels, filter_sgs,
                plot_sgs, no_result_sgs, result_sgs, figsize, dpi)


def plot_section_results(section, load_id, result_list=['Nx', 'Nxy', 'Mx',
            'My', 'Mxy'], title_list=[], thickness=True, pt_size=4,
            segment_contour=True, diagram=True, diagram_contour=False,
            diagram_alpha=0.15, diagram_scale=1.0, diagram_factor_list=[],
            contour_color='jet_r', contour_levels=10, filter_sgs=[],
            plot_sgs=[], no_result_sgs=[], result_sgs=[],
            figsize=(6.4, 4.8), dpi=80):
    """
    Uses matplotlib to plot the results associated to a section and load case
    id.

    Note
    ----
    Internal loads and strains need to be calculated using the method
    abdbeam.Section.calculate_results() before using this function;
    If two or more results are plotted, the plots will be presented in two
    columns; figure sizes are for individual plots and not the entire
    figure.

    Parameters
    ----------
    section : abdbeam.Section
        The section object to be plotted.
    load_id : int
        The load case id key in the abdbeam.Section.loads dictionary.
    result_list : list, default ['Nx', 'Nxy', 'Mx', 'My','Mxy']
        The segment internal loads and/or strains list to be plotted for the
        selected load case. Options are: 'Nx', 'Nxy', 'Mx', 'My','Mxy','ex_o',
        'ey_o', 'gxy_o', 'kx','ky','kxy'.
    title_list : list, default []
        A list containing all the plot titles to be added. An empty list (the
        default) will use the list result_list as titles. If the length of
        this list is smaller than result_list's length, None values will be
        assumed for the last items.
    thickness : bool, default True
        If True, will plot the segments thickness.
    pt_size : int, default 4
        The size in pixels of the marker. Booms (points with EAs and GJs) will
        have 2 times this size.
    segment_contour : bool, default True
        If True, will plot the internal load contour inside a segment
        thickness.
    diagram : bool, default True
        If True, will plot result diagrams at each segment. Positive
        values are plotted towards the segment top side and negative towards
        the bottom side.
    diagram_contour: bool, default False
        If True, will replace the standard gray diagram color, with each
        segments' result contour.
    diagram_alpha: float, default 0.15
        The diagram transparency alpha.
    diagram_scale: float, default 1.0
        A scale factor to be applied to the diagram plot. Negative values will
        reverse its plot direction. Does not affect the result values, only the
        diagram plot.
    diagram_factor_list : list, default []
        A list containing factors to multiply each segment's diagram. An empty
        list (the default) is a list with factors=1. If the length of
        this list is smaller than the number of segments, 1.0 values will be
        assumed for the last items. The factors' order is the same as the order
        in which the segments were entered in the section segments dictionary.
    contour_color: st, default 'jet_r'
        The matplotlib's colormap name to be used in all contours.
    contour_levels: int, default 10
        The number of contour level color areas to be used.
    filter_sgs : list, default []
        The list of segment ids that will not be plotted. Of the form [int].
        Will also respect the filter imposed by the plot_sgs parameter.
    plot_sgs : list, default []
        The list containing the only segment ids that will be plotted. Of the
        form [int]. If left empty will plot all segments. Will also respect the
        filter imposed by the filter_sgs parameter.
    no_result_sgs : list, default []
        The list of segment ids that will not have results plotted. Of the form
        [int]. Will also respect the filter imposed by the result_sgs parameter.
    result_sgs : list, default []
        The list containing the only segment ids that will have results
        plotted. Of the form [int]. If left empty will plot contours for all
        segments. Will also respect the result_sgs.
    figsize : tuple, default (6.4, 4.8)
        Width and height of each result plot in inches. Of the form (float,
        float). Note that this is not the size of the entire matplotlib figure,
        but the size of each result plot.
    dpi : integer, default 100
        The resolution of the figure.

    Examples
    --------
    The example below creates a "C" section, creates load case id 100 with a
    vertical shear at the shear center of 150 and plots the 'Nxy' internal
    loads:

    .. code-block:: python

        import abdbeam as ab
        sc = ab.Section()
        mts = dict()
        mts[1] = ab.Isotropic(0.08, 10600000, 0.33)
        pts = dict()
        pts[1] = ab.Point(0, -1.0)
        pts[2] = ab.Point(-1, -1.0)
        pts[3] = ab.Point(-1, 1.0)
        pts[4] = ab.Point(0, 1.0)
        sgs = dict()
        sgs[1] = ab.Segment(1,2,1)
        sgs[2] = ab.Segment(2,3,1)
        sgs[3] = ab.Segment(3,4,1)
        sc.materials = mts
        sc.points = pts
        sc.segments = sgs
        sc.calculate_properties()
        Lds = dict()
        Lds[100] = ab.Load(Vz_s=150)
        sc.loads = Lds
        sc.calculate_results()
        ab.plot_section_results(sc, 100, result_list=['Nxy'])
    """
    sc = section
    title_list = title_list[:len(result_list)]
    title_list = title_list + (len(result_list) - len(title_list))*[None]
    # Check if the result exists in the pandas dataframe, removing it otherwise
    chkd_result_list = []
    chkd_title_list = []
    for result, title in zip(result_list, title_list):
        if result not in sc.sgs_results_df.columns:
            msg = ("No result " + result + " calculated for the section. "
                   + "Skipping...")
            warnings.warn(msg, UserWarning, stacklevel=2)
        else:
            chkd_result_list.append(result)
            chkd_title_list.append(title)
    result_list = chkd_result_list
    title_list = chkd_title_list

    if len(result_list)>1:
        rows = (len(result_list) - 1)//2 + 1
        fig, axs = plt.subplots(rows, 2, squeeze=False)
        figsize = (figsize[0]*2, figsize[1]*rows)
    elif len(result_list)==1:
        fig, axs = plt.subplots(1, 1, squeeze=False)
    col = 0
    for i, result, title in zip(range(0, len(result_list)), result_list,
                              title_list):
        if result not in sc.sgs_results_df.columns:
            msg = ("No result " + result + " calculated for the section. "
                   + "Skipping...")
            warnings.warn(msg, UserWarning, stacklevel=2)
            continue
        if title==None:
            fig_title = '{}'.format(result)
        else:
            fig_title = title
        _plot_section_to_ax(sc, axs[i//2, col], segment_coord=False,
                thickness=thickness, pt_size=pt_size, mid_plane=thickness,
                top_bottom=False, centroid=False, shear_center=False,
                origin=False, princ_dir=False, show_axis=False,
                filter_sgs=filter_sgs, plot_sgs=plot_sgs, legend=False,
                title=fig_title)
        axs[i//2, col].autoscale()
        _plot_result_curves(sc, fig, axs[i//2, col], load_id, result,
                thickness=thickness, segment_contour=segment_contour,
                diagram=diagram, diagram_contour=diagram_contour,
                diagram_alpha=diagram_alpha, contour_color=contour_color,
                contour_levels=contour_levels, diagram_scale=diagram_scale,
                diagram_factor_list=diagram_factor_list, filter_sgs=filter_sgs,
                plot_sgs= plot_sgs, no_result_sgs=no_result_sgs,
                result_sgs=result_sgs)
        col = 1-col
    if len(result_list)>=1:
        axs[-1,-1].axis('off')
        if figsize != None:
            fig.set_size_inches(figsize)
        if dpi != None:
            fig.set_dpi(dpi)
        plt.tight_layout()
        if len(result_list) > 1:
            plt.subplots_adjust(top=0.85)
        plt.ion()
        plt.show()


def _plot_result_curves(section, fig, ax, load_id, result, thickness=True,
            segment_contour=True, diagram=True, diagram_contour=False,
            diagram_alpha=0.15, contour_color='jet_r', diagram_scale=1.0,
            diagram_factor_list=[], filter_sgs=[], plot_sgs=[],
            no_result_sgs=[], result_sgs=[], contour_levels=10):
    sc = section
    include_sgs = []
    if result_sgs and plot_sgs:
        include_sgs = _intersect(plot_sgs,result_sgs)
    elif result_sgs:
        include_sgs = result_sgs
    elif plot_sgs:
        include_sgs = plot_sgs
    exclude_sgs = _union(no_result_sgs, filter_sgs)
    if include_sgs:
        dff = sc.sgs_results_df[(sc.sgs_results_df['Load_Id']==load_id) &
                    (~sc.sgs_results_df['Segment_Id'].isin(exclude_sgs)) &
                    (sc.sgs_results_df['Segment_Id'].isin(include_sgs))]
    else:
        dff = sc.sgs_results_df[(sc.sgs_results_df['Load_Id']==load_id) &
                    (~sc.sgs_results_df['Segment_Id'].isin(exclude_sgs))]
    # Set the gradient levels
    max_load = dff[(result, 'Max')].max()
    min_load = dff[(result, 'Min')].min()
    max_abs_load = max(abs(max_load), abs(min_load))
    if max_load<min_load or (max_load == 0 and min_load == 0):
        levels=[-1e-99, +1e-99]
    elif math.isclose(max_load, min_load, rel_tol = 1e-8):
        if math.isclose(max_load, 0, rel_tol=1e-8):
            levels = [-1e-99, +1e-99]
        elif max_load > 0:
            levels = np.linspace(0, max_load, contour_levels+1).tolist()
        elif max_load < 0:
            levels = np.linspace(min_load, 0, contour_levels+1).tolist()
    else:
        levels = np.linspace(min_load, max_load, contour_levels+1).tolist()
    # Format the diagram factor list
    diagram_factor_list = diagram_factor_list[:len(sc.segments)]
    diagram_factor_list = diagram_factor_list + (len(sc.segments)
                          - len(diagram_factor_list))*[1.0]
    # Calculate result value to diagram units ratio
    # This expects only a section plot with no properties
    x0, y0, width, height = ax.dataLim.bounds
    dgm_max = max(width, height)*0.20*diagram_scale
    if max_abs_load > 0:
        dgm_f = dgm_max/max_abs_load
    else:
        dgm_f = 1.0
    # Cycle segments
    for (sg_id, sg), diagf in zip(sc.segments.items(), diagram_factor_list):
        if sg_id in exclude_sgs: continue
        if include_sgs:
            if sg_id not in include_sgs: continue
        # Extract point locations
        ya = sc.points[sg.point_a_id].y
        za = sc.points[sg.point_a_id].z
        yb = sc.points[sg.point_b_id].y
        zb = sc.points[sg.point_b_id].z
        #The segment rotation
        angle = _clockwise_angle_from_3_points(ya+1,za,ya,za,yb,zb)
        base = ax.transData
        rot = transforms.Affine2D().rotate_deg_around(ya, za, angle)
        #Filter the results dataframe:
        df = sc.sgs_results_df[(sc.sgs_results_df['Segment_Id']==sg_id) &
                               (sc.sgs_results_df['Load_Id']==load_id)]
        # Transparent bounding box for plot auto-size consistency between
        # diagrams. To do: account for colorbar auto-sizing
        y = np.array([0, 0, sg.bk, sg.bk])
        z = np.array([dgm_max, -dgm_max, -dgm_max, dgm_max])
        ax.plot(y + ya, z + za, linestyle='-', color='m',
                 linewidth=1, transform=(rot+base), alpha=0.0)
        # Extract the second degree terms:
        a = 0
        if (result, 'C2') in df.columns:
            a = df.iloc[0][(result, 'C2')]
        b = df.iloc[0][(result, 'C1')]
        c = df.iloc[0][(result, 'C0')]
        # Do the contour inside the thickness
        if segment_contour:
            if thickness:
                t = sg.t
            else:
                box = ax.get_window_extent().transformed(
                        fig.dpi_scale_trans.inverted())
                width_pixels = box.width*fig.dpi
                x_limits = ax.get_xlim()
                x_delta_value = abs(x_limits[1]-x_limits[0])
                t_per_pixel = x_delta_value/width_pixels
                t = 1.0*t_per_pixel
            xi, yi = np.meshgrid(np.linspace(ya, ya+sg.bk, 25),
                                 np.linspace(za - t/2, za + t/2, 2))
            zi = _f((xi-ya)/sg.bk, a, b, c)
            cf = ax.contourf(xi, yi, zi, levels=levels,
                           cmap=plt.get_cmap(contour_color), alpha=1.00,
                           transform=(rot+base))
        # Plot contoured load diagrams
        if diagram_contour:
            xi, yi = np.meshgrid(np.linspace(ya, ya+sg.bk, 25),
                            np.linspace(za-dgm_max, za+dgm_max, 2))
            zi = _f((xi-ya)/sg.bk, a, b, c)
            cf=ax.contourf(xi, yi, zi, levels=levels,
                           cmap=plt.get_cmap(contour_color),
                           alpha=diagram_alpha, transform=(rot+base))
            xi=np.linspace(ya, ya+sg.bk, 25)
            yi=_f((xi-ya)/sg.bk, a, b, c)
            yi = za+yi*dgm_f*diagf
            verts = [(ya, za), *zip(xi, yi), (ya+sg.bk, za)]
            poly = Polygon(verts, fc='none', ec='gray', transform=(rot+base),
                          alpha=0.5)
            for col in cf.collections:
                col.set_clip_path(poly)
            ax.add_patch(poly)
        elif diagram:
            # Plot gray load diagrams
            y = np.linspace(0, 1, 25)
            z = _f(y,a,b,c)
            y *= sg.bk
            z *= dgm_f*diagf
            y = np.insert(y, 0, 0.0)
            z = np.insert(z, 0, 0.0)
            y = np.append(y, sg.bk)
            z = np.append(z, 0.0)
            ax.fill(y+ ya,z+ za, color="black", alpha=diagram_alpha,
                    transform=(rot+base))
    if segment_contour or diagram_contour:
        cb = fig.colorbar(cf, ax=ax, ticks=levels)
        cb.ax.tick_params(direction='in')


def _f(x, a, b, c):
    return a*x**2 + b*x + c


def _intersect(lst_a, lst_b):
    """ return the intersection of two lists """
    return list(set(lst_a) & set(lst_b))


def _union(lst_a, lst_b):
    """ return the union of two lists """
    return list(set(lst_a) | set(lst_b))