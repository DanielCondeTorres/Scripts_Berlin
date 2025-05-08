
#EXAMPLES: 
#python area_per_lipid.py -p popc_sim_0.pdb -x popc_sim_0.xtc  -a 'PO4' -r 'POPC' -s 1 

import numpy as np
from MDAnalysis import *
import optparse
import fileinput
from math import *
import matplotlib.pyplot as plt
desc="Calculation of lateral diffusion coefficients for molecules in the XY plane (typically lipids in a bilayer). Example: ./ldc_00.py -p pc_pg_g_180ns_P8.pdb -x pc_pg_g_180ns_P8.xtc -a 'P8' -t 2 -r 'LPOG' -o LPOG_2.ldc -s 20"

def dist2D(p,q):
    d=0
    for i in range(0,2,1): d=d+(p[i]-q[i])*(p[i]-q[i])
    return sqrt(d)

def GetH(d, nbins):
    i=0; up=1.0
# A loop to guarantee that we have a complete distribution with zeros at the end, without a fix range (that could be risky)
    while True:
        i=i+1; 
        h=np.histogram(d, bins=nbins, density=True, range=(0,up)) # determination of histogram
        up=up+0.2; stop='y'
        for j in range (1,6,1): # This guarantees at least 5 zeros at the end of the distribution. Then the distribution is complete
            if (h[0][-j]!=0): stop='n'
        if (stop=='y' or i>100): break # two conditions to avoid infinite loops
    r=[]; P=h[0];
    for i in range(0,len(P),1): r.append((h[1][i]+h[1][i+1])/2.) # this is because np.histogram gives one more dimension for r than for P
    return ([r,P])

def calcf (r, P, D, dt ) :
    f=0;
    for j in range(0, len(r), 1) :
        factor=r[j]/(2.*dt*D)
        #print factor, r[j], P[j]
        f=f+((factor*exp(-0.5*r[j]*factor))-P[j])*((factor*exp(-0.5*r[j]*factor))-P[j])
    return f

def calcfp (f, P, D, r, dt) :
    fp=0
    for j in range(0, len(r), 1) :
        factor=r[j]/(2.*dt*D)
        fp=fp+(factor*exp(-0.5*r[j]*factor)/D*(0.5*r[j]*factor-1.0))*2.*(factor*exp(-0.5*r[j]*factor)-P[j])
    return fp

def FitLDC(r,P, dt):
    dt=float(dt)/1.e3
    fmin=1.e30; D=Dmin=1; # D is the seed for the iterative calculation of the Diffusion Coefficient
    for j in range(0, 5000, 1) : # 1D dumped Newton-Raphson to get the diffusion coefficient
        if (D<0): 
            D=-D
        else:
            f=calcf(r, P, D, dt)
            fp=calcfp(f, P, D, r, dt)
        if (f<fmin) :
            fmin=f; Dmin=D
        if ((fp*fp)>0): D=D-0.005*f/fp # 0.005 is a dumping factor to avoid divergences
    return Dmin

def main() :
    parser = optparse.OptionParser(description=desc, version='%prog version 1.0')
    parser.add_option('-p', '--pdb_file', help='input pdb file', action='store')
    parser.add_option('-x', '--xtc_file', help='input xtc file', action='store')
    parser.add_option('-n', '--nbins', help='number of bins employed to generate the distribution (default=30)', action='store')
    parser.add_option('-a', '--atomname', help='atom name', action='store')
    parser.add_option('-r', '--resname', help='residue name', action='store')
    parser.add_option('-b', '--begin', help='begin: first frame to be analyzed', action='store')
    parser.add_option('-e', '--end', help='end: last frame to be analyzed', action='store')
    parser.add_option('-s', '--skip', help='skip: only write every nr-th frame', action='store')
    parser.add_option('-t', '--dt', help='time step to determine the displacements (in ns)', action='store')
    parser.add_option('-o', '--outfile_1', help='output file for the displacements distribution of all the lipids', action='store')
    parser.add_option('-d', '--outfile_2', help='output file for the diffusion constants distribution of every individual lipid', action='store')
    parser.set_defaults(pdb_file='file.pdb', xtc_file='none', nbins=30, atomname='P', resname='POPC', begin=0, end=-1, skip=1, dt=2, outfile_1='out_1.ldc', outfile_2='out_2.ldc')
    options, arguments = parser.parse_args()

#**************************************************************************************
    pdb_file=str(options.pdb_file)
    xtc_file=str(options.xtc_file)
    nbins=int(options.nbins)
    begin=int(options.begin)
    end=int(options.end)
    skip=int(options.skip)
    outfile_1=str(options.outfile_1)
    outfile_2=str(options.outfile_2)
    atomname=str(options.atomname)
    resname=str(options.resname)
    dt=int(float(options.dt)*1000)

    trj=Universe(pdb_file, xtc_file)

    atlist=[]; fr=[]; j=0
    # Reading pdb file and storing the reference atom together with its XY coordinates in the matrix atlist
    for frame in trj.trajectory[begin:end:skip]: # Iterate over frames
        j=j+1; 
        if (j%500)==0: print("Frame",j)
        fr.append([frame.time, frame.dimensions[0], frame.dimensions[1]])
        atlist_i=[]; i=-1
        for at in trj.atoms: # Iterate over atoms
            if (at.name==atomname and at.residue.resname==resname):
                i=i+1; 
                x=at.position[0]; y=at.position[1]; fx=frame.dimensions[0]; fy=frame.dimensions[1]
                if len(fr)>1: # Correction of positions by PBC. This may fail for large displacements (usually large dt)
                    aux_x=x-atlist[-1][i][0]; dx=aux_x*aux_x
                    aux_y=y-atlist[-1][i][1]; dy=aux_y*aux_y
                    dx0=aux_x+fx; dx0=dx0*dx0; dx1=aux_x-fx; dx1=dx1*dx1
                    dy0=aux_y+fy; dy0=dy0*dy0; dy1=aux_y-fy; dy1=dy1*dy1
                    if (dx0<dx and dx0<dx1): x=x+fx
                    elif (dx1<dx and dx1<dx0): x=x-fx
                    if (dy0<dy and dy0<dy1): y=y+fy
                    elif (dy1<dy and dy1<dy0): y=y-fy
                atlist_i.append([x, y])
        atlist.append(atlist_i)  
      
    if len(atlist[0])==0: print("\n\nERROR!!!! I have not found lipidname", resname, "or atom", atomname, "in the xtc file\n\n")
    timestep=int(fr[1][0]-fr[0][0]) # time between 2 frames (in pdb units: picoseconds in gromacs)
    frstep=dt/timestep # translation of dt for the determination of diffusion coefficients into number of frames
    d=[]; Dmin_i=[]
   
 # determination of lateral displacements
    print("\nCalculating displacements over the trajectory every", dt, "ps for lipid:")
    for lipid in range(0, len(atlist[0]), 1): # loop over lipids
        d_i=[]
        if ((lipid+1)%10)==0: print(lipid+1)
        for frame in range (0, len(fr), 1): # loop over frames
            if (frame>=frstep): 
                aux=dist2D(atlist[frame-frstep][lipid], atlist[frame][lipid])
                d.append(aux); d_i.append(aux)
        h=GetH(d_i, nbins);
        r=h[0]; P=h[1];
        aux=FitLDC(r, P, dt);
        Dmin_i.append(aux)
    i=0; up=1.0

# A loop to guarantee that we have a complete distribution with zeros at the end, without a fix range (that could be risky)
    h=GetH(d, nbins); r=h[0]; P=h[1];

# Get Lateral diffusion coefficient from the fitting to the distribution of displacements
    Dmin=FitLDC(r,P, dt)

# Print experimental data together with the results of the fitting
    fout=open(outfile_1, "w")
    fout.write("#r/(nm)\t P(Simulation)\t P(Fitted)\n")
    for j in range(0, len(r), 1) :
        factor=r[j]/(2.*dt*1.e-3*Dmin)
        fout.write("%f \t %f \t %f \n" % (r[j], P[j], factor*exp(-0.5*r[j]*factor)))
    fout.close

# Print diffusion coefficients for the individual lipids
    fout=open(outfile_2, "w")
    fout.write("#Diffusion constants (cm2/s)\n")
    for j in range(0, len(atlist[0]), 1):
        fout.write("%f \n" % (Dmin_i[j]))
    fout.close

    print ("\nDiffusion constant= %f 10^-7 +/- %f 10^-7 cm2/s\n" % (Dmin, np.std(Dmin_i)))
    print ("\nCheck file %s to see the distribution of displacements for all lipids\n" % (outfile_1))
    print ("\nCheck file %s to see the distribution of difussion constants for every lipid molecule\n" % (outfile_1))
    print ("\nType 'xmgrace -nxy %s' if you are using xmgrace \n" % (outfile_1))
    # Carga los datos (sustituye el nombre si es otro)
    data = np.loadtxt('out_1.ldc', comments='#')
    r = data[:, 0]       # posición (r en nm)
    P_sim = data[:, 1]   # probabilidad simulada
    P_fit = data[:, 2]   # probabilidad ajustada

    # Graficar
    plt.figure()
    plt.plot(r, P_sim, label='P(simulation)')
    plt.plot(r, P_fit, label='P(adjust)', linestyle='-')
    plt.xlabel('r (nm)')
    plt.ylabel('P(r)')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__=="__main__" :
    main()
