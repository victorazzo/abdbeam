import abdbeam as ab

sc = ab.Section()
# Create a dictionary to store ply materials shared by laminates
ply_mts = dict()
ply_mts[1] = ab.PlyMaterial(0.0075, 1.149e7, 1.149e7, 6.6e5, 0.04)
# Create the materials dictionary for the laminates and shear connector:
mts = dict()
mts[1] = ab.Laminate()
mts[1].ply_materials[1] = ply_mts[1]
mts[1].plies = [[45,1], [-45,1]]*2 + [[0,1]]*3
mts[1].symmetry = 'S'
mts[2] = ab.ShearConnector(0.25, 6381760)
# Create a points dictionary based on Y and Z point coordinates:
pts = dict()
pts[1] = ab.Point(1,-4)
pts[2] = ab.Point(2,-4)
pts[3] = ab.Point(12,-4)
pts[4] = ab.Point(14,-4)
pts[5] = ab.Point(15,-4)
pts[11] = ab.Point(1,1.21)
pts[12] = ab.Point(2,1.21)
pts[13] = ab.Point(12,1.21)
pts[14] = ab.Point(14,1.21)
pts[15] = ab.Point(15,1.21)
pts[21] = ab.Point(1,-3.895)
pts[22] = ab.Point(2,-3.895)
pts[23] = ab.Point(3,-3.895)
pts[24] = ab.Point(3,1.105)
pts[25] = ab.Point(2,1.105)
pts[26] = ab.Point(1,1.105)
pts[31] = ab.Point(11,-3.895)
pts[32] = ab.Point(12,-3.895)
pts[33] = ab.Point(13,-3.895)
pts[34] = ab.Point(14,-3.895)
pts[35] = ab.Point(15,-3.895)
pts[36] = ab.Point(11,1.105)
pts[37] = ab.Point(12,1.105)
pts[38] = ab.Point(13,1.105)
pts[39] = ab.Point(14,1.105)
pts[40] = ab.Point(15,1.105)
# Create a segments dictionary referencing point and material ids:
sgs = dict()
sgs[1] = ab.Segment(1,2,1,'Bottom Skin')
sgs[2] = ab.Segment(2,3,1,'Bottom Skin')
sgs[3] = ab.Segment(3,4,1,'Bottom Skin')
sgs[4] = ab.Segment(4,5,1,'Bottom Skin')
sgs[11] = ab.Segment(11,12,1,'Top Skin')
sgs[12] = ab.Segment(12,13,1,'Top Skin')
sgs[13] = ab.Segment(13,14,1,'Top Skin')
sgs[14] = ab.Segment(14,15,1,'Top Skin')
sgs[21] = ab.Segment(21,22,1,'Fwd Spar')
sgs[22] = ab.Segment(22,23,1,'Fwd Spar')
sgs[23] = ab.Segment(23,24,1,'Fwd Spar')
sgs[24] = ab.Segment(24,25,1,'Fwd Spar')
sgs[25] = ab.Segment(25,26,1,'Fwd Spar')
sgs[31] = ab.Segment(31,32,1,'Rear Spar')
sgs[32] = ab.Segment(32,33,1,'Rear Spar')
sgs[33] = ab.Segment(33,34,1,'Rear Spar')
sgs[34] = ab.Segment(34,35,1,'Rear Spar')
sgs[35] = ab.Segment(33,38,1,'Rear Spar')
sgs[36] = ab.Segment(36,37,1,'Rear Spar')
sgs[37] = ab.Segment(37,38,1,'Rear Spar')
sgs[38] = ab.Segment(38,39,1,'Rear Spar')
sgs[39] = ab.Segment(39,40,1,'Rear Spar')
sgs[91] = ab.Segment(2,22,2,'Connector')
sgs[92] = ab.Segment(3,32,2,'Connector')
sgs[93] = ab.Segment(4,34,2,'Connector')
sgs[94] = ab.Segment(25,12,2,'Connector')
sgs[95] = ab.Segment(37,13,2,'Connector')
sgs[96] = ab.Segment(39,14,2,'Connector')
# Point the dictionaries to the section
sc.materials = mts
sc.points = pts
sc.segments = sgs
# Calculate section properties
sc.calculate_properties()
# Plot the section
ab.plot_section(sc, pt_size=2, title='Abdbeam - Torque-box Example',
                figsize=(6.4*1.5, 4.8*1.5))
# Create load cases and calculate their internal loads
sc.loads[8] = ab.Load(Px=17085,My=-140914,Mz=-7208,Tx=1595,Vy=4727,Vz=-1661)
sc.loads[4] = ab.Load(Px=11854,My=-89211,Mz=-33716,Tx=-57488,Vy=5684,Vz=394)
sc.loads[1] = ab.Load(Px=2395,My=-83206,Mz=210099,Tx=-43162,Vy=1316,Vz=407)
sc.loads[10] = ab.Load(Px=-7458,My=-15571,Mz=-96370,Tx=-3615,Vy=564,Vz=-369)
sc.loads[2] = ab.Load(Px=1000,My=-30865,Mz=180498,Tx=11653,Vy=-7001,Vz=-189)
sc.loads[3] = ab.Load(Px=-281,My=133314,Mz=-123966,Tx=324,Vy=9389,Vz=-1514)
sc.loads[6] = ab.Load(Px=299,My=40658,Mz=101677,Tx=7102,Vy=9214,Vz=-3545)
sc.calculate_results()
# Use Pandas methods to get info on the critical spar compressive Nx
df = sc.sgs_int_lds_df
spar_sgs = range(31,40)
df = df[df['Segment_Id'].isin(spar_sgs)]
idx = df[('Nx', 'Min')].idxmin()
min_Nx = round(df.loc[idx, ('Nx', 'Min')],1)
min_sg = int(df.loc[idx, 'Segment_Id'])
min_lc = int(df.loc[idx, 'Load_Id'])
print(('Minimum rear spar Nx is {}, from segment {}, load case {}'
       ).format(min_Nx, min_sg, min_lc))
# Plot the critical compressive case Nx internal loads
ab.plot_section_results(sc, min_lc, result_list=['Nx'],
                        title_list=['Critical Compressive Nx, LC '+
                        str(min_lc)], plot_sgs=range(31,40),
                        figsize=(6.4*0.8, 4.8*0.8))
