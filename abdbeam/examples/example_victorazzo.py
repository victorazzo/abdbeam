import abdbeam as ab

sc = ab.Section()
# Create a materials dictionary:
mts = dict()
mts[1] = ab.Laminate()
ply_mat = ab.PlyMaterial(0.166666, 148000, 9650, 4550, 0.3)
mts[1].ply_materials[1] = ply_mat
mts[1].plies = [[0,1]]*6 + [[45,1]]*6
# Create a points dictionary based on Y and Z point coordinates:
pts = dict()
pts[1] = ab.Point(0, -35)
pts[2] = ab.Point(-50, -35)
pts[3] = ab.Point(-50, 35)
pts[4] = ab.Point(0, 35)
pts[5] = ab.Point(50, 35)
pts[6] = ab.Point(50, -35)
# Create a segments dictionary referencing point and material ids:
sgs = dict()
sgs[1] = ab.Segment(1,2,1)
sgs[2] = ab.Segment(2,3,1)
sgs[3] = ab.Segment(3,4,1)
sgs[4] = ab.Segment(4,1,1)
sgs[5] = ab.Segment(4,5,1)
sgs[6] = ab.Segment(5,6,1)
sgs[7] = ab.Segment(6,1,1)
# Point the dictionaries to the section
sc.materials = mts
sc.points = pts
sc.segments = sgs
# Calculate and output section properties
sc.calculate_properties()
sc.summary()
# Create and calculate internal loads
sc.loads = dict()
sc.loads[101] = ab.Load(My=5e6)
sc.loads[102] = ab.Load(Tx=250000, Vz=5000.0)
sc.calculate_results()
# Access Pandas dataframe with internal loads
df = sc.sgs_int_lds_df
# Plot the section and its properties
ab.plot_section(sc, segment_coord=True, title='Abdbeam - Example',
                legend=False, prop_color='#471365', figsize=(5.12, 3.84))
# Plot Nx and Nxy internal loads for case 101
ab.plot_section_results(sc, 101, contour_color='viridis', diagram_scale=0.7,
                        result_list=['Nx', 'Nxy'], figsize=(5.12, 3.84))