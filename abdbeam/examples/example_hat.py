import abdbeam as ab

sc = ab.Section()
# Create a dictionary to store ply materials shared by laminates
ply_mts = dict()
ply_mts[1] = ab.PlyMaterial(0.0075, 2.147e7, 1.4e6, 6.6e5, 0.3)
ply_mts[2] = ab.PlyMaterial(0.0075, 1.149e7, 1.149e7, 6.6e5, 0.04)
# Create the materials dictionary for the laminates and shear connector:
mts = dict()
mts[1] = ab.Laminate()
mts[1].ply_materials[2] = ply_mts[2]
mts[1].plies = [[45,2], [-45,2]] + [[0,2]]*3
mts[1].symmetry = 'S'
mts[2] = ab.Laminate()
mts[2].ply_materials[2] = ply_mts[2]
mts[2].plies = [[45,2], [-45,2]]*2 + [[0,2]]
mts[2].symmetry = 'S'
mts[3] = ab.Laminate()
mts[3].ply_materials[1] = ply_mts[1]
mts[3].ply_materials[2] = ply_mts[2]
mts[3].plies = [[45,2], [-45,2]] + [[0,1]]*3 + [[0,2]] + [[0,1]]*2
mts[3].symmetry = 'SM'
mts[4] = ab.ShearConnector(0.075, 2605615)
# Create a points dictionary based on Y and Z point coordinates:
pts = dict()
pts[1] = ab.Point(-2, 0)
pts[2] = ab.Point(-1, 0)
pts[3] = ab.Point(1, 0)
pts[4] = ab.Point(2, 0)
pts[5] = ab.Point(-2, 0.075)
pts[6] = ab.Point(-1, 0.075)
pts[7] = ab.Point(-0.35, 0.8)
pts[8] = ab.Point(0.35, 0.8)
pts[9] = ab.Point(1, 0.075)
pts[10] = ab.Point(2, 0.075)
# Create a segments dictionary referencing point and material ids:
sgs = dict()
sgs[1] = ab.Segment(1,2,1,'Skin_Left')
sgs[2] = ab.Segment(2,3,1,'Skin_Center')
sgs[3] = ab.Segment(3,4,1,'Skin_Right')
sgs[10] = ab.Segment(5,6,2,'Hat_Left_Foot')
sgs[11] = ab.Segment(6,7,2,'Hat_Left_Web')
sgs[12] = ab.Segment(7,8,3,'Hat_Top')
sgs[13] = ab.Segment(8,9,2,'Hat_Right_Web')
sgs[14] = ab.Segment(9,10,2,'Hat_Right_Foot')
sgs[91] = ab.Segment(1,5,4,'Connector_1')
sgs[92] = ab.Segment(2,6,4,'Connector_1')
sgs[93] = ab.Segment(3,9,4,'Connector_1')
sgs[94] = ab.Segment(4,10,4,'Connector_1')
# Point the dictionaries to the section
sc.materials = mts
sc.points = pts
sc.segments = sgs
# Calculate section properties
sc.calculate_properties()
sc.summary()
# Plot the section
ab.plot_section(sc, filter_sgs=[91,92,93,94], title='Abdbeam - Hat Example',
                prop_color='#471365')
# Create load cases and calculate their internal loads
sc.loads[1] = ab.Load(My=100, Vy_s=1000)
sc.loads[2] = ab.Load(Tx=100)
sc.calculate_results()
# Plot internal loads
ab.plot_section_results(sc, 1, contour_color = 'viridis',
                      result_sgs=[10,11,12,13,14], figsize=(6.4*0.8, 4.8*0.8),
                      diagram_scale=0.5, filter_sgs=[91,92,93,94])