import abdbeam as ab

sc = ab.Section()
# Create a dictionary for the isotropic material
mts = dict()
mts[1] =  ab.Isotropic(2, 70000, 0.3)
# Create a points dictionary based on Y and Z point coordinates
pts = dict()
pts[1] = ab.Point(0,-200)
pts[2] = ab.Point(-100,-200)
pts[3] = ab.Point(-100,0)
pts[4] = ab.Point(-200,0)
pts[5] = ab.Point(-200,-100)
pts[6] = ab.Point(0,0)
pts[7] = ab.Point(100,-200)
pts[8] = ab.Point(100,0)
pts[9] = ab.Point(200,0)
pts[10] = ab.Point(200,-100)
# Create a segments dictionary referencing point and material ids
sgs = dict()
sgs[1] = ab.Segment(1,2,1)
sgs[2] = ab.Segment(2,3,1)
sgs[3] = ab.Segment(3,4,1)
sgs[4] = ab.Segment(4,5,1)
sgs[5] = ab.Segment(3,6,1)
sgs[6] = ab.Segment(1,7,1)
sgs[7] = ab.Segment(7,8,1)
sgs[8] = ab.Segment(8,9,1)
sgs[9] = ab.Segment(9,10,1)
sgs[10] = ab.Segment(8,6,1)
# Point the dictionaries to the section
sc.materials = mts
sc.points = pts
sc.segments = sgs
# Calculate section properties
sc.calculate_properties()
# Plot the section
ab.plot_section(sc, segment_coord=True, title='Abdbeam - Megson Example')
#Create load case and calculate its internal loads:
sc.loads[1] = ab.Load(Vz_s=100000)
sc.calculate_results()
ab.plot_section_results(sc, 1, segment_contour=False, diagram=True,
                        diagram_contour=True, diagram_alpha=1.0,
                        contour_levels=20, contour_color='coolwarm',
                        diagram_factor_list=[1,1,-1,-1,1,-1,-1,1,1,-1],
                        thickness=False, result_list=['Nxy'],
                        title_list=['Nxy (N/mm)'])
