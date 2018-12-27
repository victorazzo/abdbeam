# -*- coding: utf-8 -*-
"""
Module containing methods to plot cross sections and their internal loads.
"""

import matplotlib.pyplot as plt
from matplotlib import transforms
import pandas as pd
import math

import abdbeam as ab
import numpy as np

def plot_section(section, orientation=True, thickness=True, mid_plane=True, top_bottom=True,
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
        angle = ab.core._clockwise_angle_from_3_points(ya+1,za,ya,za,yb,zb)
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
