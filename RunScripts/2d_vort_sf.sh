# Set the number of MPI process to execute with
num_run_procs=64

# Executable
execut="./src/fastSF.out"

# scalar/vector switch
scal_vec_switch="true"

# 2D Dim switch
dim="true"

# Switch to run my code -> set to false
fast_code_switch="false"

# Longitudinal sf only swithc
long_sf_switch="false"

# Number of processors in the x direction
procs_x=8

# checkpoints
chkpts=10

# Dimensions
Nx=1024
Ny=1
Nz=1024 # this is Ny

# grid size
Lx=6.283185307179586477
Ly=1.0
Lz=6.283185307179586477 # grid size in y direction

# Powers to compute
q1=1
q2=6

# data directory
data_dir="/root/2DNS/Data/SF/NAV_AB4CN_FULL_N[1024,1024]_T[0.0,0.00025,100.000]_NU[5e-20,1,4.0]_DRAG[0.1,0.1,1,0.0]_FORC[BODY_FORC_COS,2,1]_u0[RANDOM]_TAG[SFTest]/"

# Path to input file - use the -I switch for this
input_file="$data_dir/Main_HDF_Data.h5"

# Path to output files
scalar_out_file="Vort_StrFunc_Data"
perp_out_file="Vel_TransStrFunc_Data"
long_out_file="Vel_LongStrFunc_Data"

# Run command
cmd="mpirun -np $num_run_procs $execut -F $fast_code_switch -c $chkpts -s $scal_vec_switch -d $dim -l $long_sf_switch -p $procs_x -X $Nx -Y $Ny -Z $Nz -x $Lx -y $Ly -z $Lz -1 $q1 -2 $q2 -D $data_dir -I $input_file -M $scalar_out_file -P $perp_out_file -L $long_out_file"
echo -e "\n$cmd\n"
$cmd
