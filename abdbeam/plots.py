# -*- coding: utf-8 -*-
"""
Module containing methods to plot cross sections and their internal loads.
"""

import matplotlib.pyplot as plt
from matplotlib import transforms
import pandas as pd
import math
import numpy as np
from abdbeam.core import _clockwise_angle_from_3_points

def plot_section(section, orientation=False, thickness=True, mid_plane=True, top_bottom=False,
                 centroid=True, shear_center=True, origin=True,
                 princ_dir=True, show_axis=True, prop_color='r', filter_sgs=[],
                 plot_sgs=[], figsize=None, dpi=None, legend=True, title=''):

    fig, axs = plt.subplots(1, 1, squeeze=False)
    _plot_section_to_ax(section, axs[0,0], orientation, thickness, mid_plane, top_bottom, centroid,
                  shear_center, origin, princ_dir, show_axis, prop_color,
                  filter_sgs, plot_sgs, legend, title)
    if figsize != None:
        fig.set_size_inches(figsize)
    if dpi != None:
        fig.set_dpi(dpi)
    plt.show()


def _plot_section_to_ax(section, ax,  orientation=True, thickness=True, mid_plane=True, top_bottom=True,
                 centroid=True, shear_center=True, origin=True,
                 princ_dir=True, show_axis=True, prop_color='r', filter_sgs=[],
                 plot_sgs=[], legend=True, title=''):
    """
    """
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
        # Calculate segment versors
        vry = (yb-ya)/sg.bk
        vrz = (zb-za)/sg.bk
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
            ax.plot([ya, yb], [za, zb], marker='o', markersize=4,
                     linestyle=lstyle, color='k', linewidth=1)
        else:
            ax.plot([ya], [za], marker='o', markersize=4, color="k")
            ax.plot([yb], [zb], marker='o', markersize=4, color="k")
        # Plot segment direction with a 25% segment length
        if orientation == True:
            y = np.array([ya+sg.bk*0.5, ya+sg.bk*0.5, ya+sg.bk*0.65])
            z = np.array([za+sg.bk*0.15, za, za])
            ax.plot(y, z, linestyle='-', color='r',
                     linewidth=1, transform = rot + base)
        #Set top/bottom colors
        if top_bottom == True:
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
            ax.scatter([ya], [za], s=80, facecolors='none', edgecolors='k')
            #plt.plot([ya], [za], marker='o', markersize=10, color="g")
        if sc.points[sg.point_b_id].EA > 0 or sc.points[sg.point_b_id].GJ > 0:
            ax.scatter([yb], [zb], s=80, facecolors='none', edgecolors='k')
            #plt.plot([yb], [zb], marker='o', markersize=10, color="g")
        # Plot origin
    if origin == True:
        ax.plot([0], [0], marker='+', markersize=20, color='k')
    # Plot centroid
    if centroid == True:
        ax.plot([sc.yc], [sc.zc], marker='o', markersize=5,
                 color=prop_color, label='Centroid', linestyle="None")
    # Plot shear center
    if shear_center == True:
        ax.plot([sc.ys], [sc.zs], marker='x', markersize=8,
                 color=prop_color, label='Shear Center', linestyle="None")
    if type(title) == str and title != '':
        ax.set_title(title)
    if show_axis == False:
        ax.axis('off')
    #Fix the axis limits at this point
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    max_axis = max(abs(x_limits[1]-x_limits[0]), abs(y_limits[1]
                      -y_limits[0]))
    # plot principal directions
    if princ_dir == True:
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
    if legend == True:
        ax.legend()


def plot_section_loads(section, load_id, load_list = ['Nx', 'Nxy', 'Mx', 'My', 'Mxy'], thickness=True, filter_sgs=[],
                 plot_sgs=[], figsize=None, dpi=None):
    sc = section
    if load_id not in sc.loads.keys(): return
    min_x = math.inf
    max_x = -math.inf
    min_y = math.inf
    max_y = -math.inf
    fig, axs = plt.subplots(len(load_list),1, squeeze=False)
    for i, load in zip(range(1,len(load_list)+1), load_list):
        # Left side plot with contour:
        _plot_section_to_ax(sc, axs[i-1,0], orientation=False, thickness=thickness, mid_plane=thickness, top_bottom=False,
                 centroid=False, shear_center=False, origin=False,
                 princ_dir=False, show_axis=False, prop_color='r', filter_sgs=filter_sgs,
                 plot_sgs=plot_sgs, legend=False, title=load)
        _plot_internal_load_curves(sc, fig, axs[i-1,0], load_id, load, thickness=thickness, contour=True, diagram=True, filter_sgs=filter_sgs, plot_sgs=plot_sgs)
        # record the auto-scale limits to later set all plots to the same scale:
        axs[i-1,0].autoscale()
        x_limits= axs[i-1,0].get_xlim()
        y_limits = axs[i-1,0].get_ylim()
        min_x = min(min_x,x_limits[0])
        max_x = max(max_x,x_limits[1])
        min_y = min(min_y,y_limits[0])
        max_y = max(max_y,y_limits[1])
    # Make all plots axis scales consistent:
    for i in range(1,len(load_list)+1):
        axs[i-1,0].set_xlim(min_x, max_x)
        axs[i-1,0].set_ylim(min_y, max_y)

    if figsize != None:
        fig.set_size_inches(figsize)
    if dpi != None:
        fig.set_dpi(dpi)

    #plt.gcf().suptitle('Load Case {}'.format(load_id), y=1.01)
    plt.tight_layout()
    plt.show()
    return None


def _func(x, a, b, c):
    return a*x**2 + b*x + c


def _plot_internal_load_curves(section, fig, ax, load_id, load, thickness=True, contour=True, diagram=True,filter_sgs=[], plot_sgs=[]):
    sc = section
    # Find the maximum and minimum for the selected load case and segments:
    if len(plot_sgs) > 0:
        dff = sc.sgs_int_lds_df[(sc.sgs_int_lds_df['Load_Id']==load_id) &
                           (~sc.sgs_int_lds_df['Segment_Id'].isin(filter_sgs)) &
                           (sc.sgs_int_lds_df['Segment_Id'].isin(plot_sgs))]
    else:
        dff = sc.sgs_int_lds_df[(sc.sgs_int_lds_df['Load_Id']==load_id) &
                           (~sc.sgs_int_lds_df['Segment_Id'].isin(filter_sgs))]
    # Set the gradient levels
    max_load = dff[(load, 'Max')].max()
    min_load = dff[(load, 'Min')].min()
    if max_load<min_load or (max_load==0 and min_load==0):
        levels=[-1e-99,+1e-99]
    elif math.isclose(max_load, min_load, rel_tol=1e-8):
        if math.isclose(max_load,0, rel_tol=1e-8):
            levels = [-1e-99, +1e-99]
        elif max_load>0:
            levels = np.linspace(0, max_load,21).tolist()
        elif max_load<0:
            levels = np.linspace(max_load, 0, 21).tolist()
    else:
        levels = np.linspace(min_load, max_load,21).tolist()

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
        #Filter the results dataframe:
        df = sc.sgs_int_lds_df[(sc.sgs_int_lds_df['Segment_Id']==sg_id) &
                               (sc.sgs_int_lds_df['Load_Id']==load_id)]
        # Extract the second degree terms:
        a = 0
        if load == 'Nxy':
            a = df.iloc[0][(load, 'C2')]
        b = df.iloc[0][(load, 'C1')]
        c = df.iloc[0][(load, 'C0')]
        # Do the contour inside the thickness
        if contour:
            if thickness:
                t = sg.t
            else:
                box = ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
                width_pixels = box.width * fig.dpi
                x_limits = ax.get_xlim()
                x_delta_value = abs(x_limits[1]-x_limits[0])
                t_per_pixel = x_delta_value/width_pixels
                t = 2.0*t_per_pixel
            xi, yi = np.meshgrid(np.linspace(ya,ya+sg.bk,25), np.linspace(za-t/2,za+t/2,2))
            zi = _func((xi-ya)/sg.bk, a, b, c)
            cf=ax.contourf(xi, yi, zi, levels=levels, cmap=plt.get_cmap('jet'), alpha = 1.00, transform = (rot + base))

        # Plot load diagrams
        if diagram==True:
            y = np.linspace(0, 1, 25)
            z = a*y**2 +b*y + c
            y *= sg.bk
            scale = 0.00001
            z *= scale
            y = np.insert(y, 0, 0.0)
            z = np.insert(z, 0, 0.0)
            y = np.append(y, sg.bk)
            z = np.append(z, 0.0)
            ax.fill(y+ ya,z+ za, color="black", alpha=0.15, transform = rot + base)

    if contour == True:
        cb = plt.colorbar(cf, ax=ax, ticks=levels)

    return None

