# Set the number of MPI process to execute with
num_run_procs=1

# Executable
execut="./src/fastSF.out"

# scalar/vector switch
scal_vec_switch="true"

# 2D Dim switch
dim="true"

# Longitudinal sf only swithc
long_sf_switch="true"

# Number of processors in the x direction
procs_x=1

# Dimensions
Nx=64
Ny=1
Nz=64 # this is Ny

# grid size
Lx=1.0
Ly=1.0
Lz=1.0 # grid size in y direction

# Powers to compute
q1=1
q2=6

# data directory
data_dir="./test/MyTestData"

# Path to input file - use the -I switch for this
input_file="$data_dir/Main_HDF_Data.h5"

# Path to output files
scalar_out_file="Vort_StrFunc_Data"
perp_out_file="Vel_TransStrFunc_Data"
long_out_file="Vel_LongStrFunc_Data"

# Run command
cmd="mpirun -np $num_run_procs $execut -s $scal_vec_switch -d $dim -l $long_sf_switch -p $procs_x -X $Nx -Y $Ny -Z $Nz -x $Lx -y $Ly -z $Lz -1 $q1 -2 $q2 -D $data_dir -I $input_file -M $scalar_out_file -P $perp_out_file -L $long_out_file"
echo -e "\n$cmd\n"
$cmd
