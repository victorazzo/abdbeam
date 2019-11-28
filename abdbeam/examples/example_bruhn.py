import abdbeam as ab

sc = ab.Section()
# Create a dictionary for the shear connector materials
mts = dict()
mts[1] =  ab.ShearConnector(0.03, 3846154)
mts[2] =  ab.ShearConnector(0.04, 3846154)
mts[3] =  ab.ShearConnector(0.05, 3846154)
mts[4] =  ab.ShearConnector(0.064, 3846154)
# Create a points dictionary based on Y and Z point coordinates
pts = dict()
pts[1] = ab.Point(0,0,2e7,0,'a')
pts[2] = ab.Point(10,0,1e7,0,'b')
pts[3] = ab.Point(20,0,5e6,0,'c')
pts[4] = ab.Point(30,0,5e6,0,'d')
pts[5] = ab.Point(40,0,5e6,0,'e')
pts[6] = ab.Point(50,0,1e7,0,'f')
pts[11] = ab.Point(0,10,2e7,0,'a_')
pts[12] = ab.Point(10,10,1e7,0,'b_')
pts[13] = ab.Point(20,10,5e6,0,'c_')
pts[14] = ab.Point(30,10,5e6,0,'d_')
pts[15] = ab.Point(40,10,5e6,0,'e_')
pts[16] = ab.Point(50,10,1e7,0,'f_')
# Create a segments dictionary referencing point and material ids
sgs = dict()
sgs[1] = ab.Segment(1,2,2,'Bottom 1')
sgs[2] = ab.Segment(2,3,2,'Bottom 2')
sgs[3] = ab.Segment(3,4,2,'Bottom 3')
sgs[4] = ab.Segment(4,5,1,'Bottom 4')
sgs[5] = ab.Segment(5,6,1,'Bottom 5')
sgs[11] = ab.Segment(11,12,2,'Top 1')
sgs[12] = ab.Segment(12,13,2,'Top 2')
sgs[13] = ab.Segment(13,14,2,'Top 3')
sgs[14] = ab.Segment(14,15,1,'Top 4')
sgs[15] = ab.Segment(15,16,1,'Top 5')
sgs[21] = ab.Segment(1,11,4,'Web 1')
sgs[22] = ab.Segment(2,12,3,'Web 2')
sgs[23] = ab.Segment(3,13,2,'Web 3')
sgs[24] = ab.Segment(4,14,2,'Web 4')
sgs[25] = ab.Segment(5,15,1,'Web 5')
sgs[26] = ab.Segment(6,16,1,'Web 6')
# Point the dictionaries to the section
sc.materials = mts
sc.points = pts
sc.segments = sgs
# Calculate section properties
sc.calculate_properties()
# Plot the section
ab.plot_section(sc, centroid =False, princ_dir=False, thickness=False,
                segment_coord=True, title='Abdbeam - Bruhn Example')
#Create load case and calculate its internal loads:
sc.loads[1] = ab.Load(Vz_s=1000)
sc.calculate_results()
# Print the shear flows Nxy for all segments
df = sc.sgs_int_lds_df
print(df[[('Segment_Id', ''),('Nxy','Avg')]])
# Print the shear center location
print('')
print('Shear center is at y = {:.8e}'.format(sc.ys))