L_AMD=-L/lib/amdlibm-3.1-lin64/lib/dynamic
I_AMD=-I/lib/amdlibm-3.1-lin64/include
# for gcc:
L_MKL=-Wl,--no-as-needed -L/opt/intel/compilers_and_libraries_2017.0.098/linux/mkl/lib/intel64
# maybe for icc:
# L_MKL=-L/opt/intel/compilers_and_libraries_2017.0.098/linux/mkl/lib/intel64
l_MKL=-lmkl_intel_ilp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl
I_MKL=-I/opt/intel/compilers_and_libraries_2017.0.098/linux/mkl/include

# Might need to export the above L paths to LD_LIBRARY_PATH
# "source /opt/intel/paralle_studio_xe_2017.0.035/psxevars.sh" to use icc
# also, running mklvars.sh might help if want use $(MKLROOT)

math:
	gcc -O3 math_speed.c -lm -o math_libm.out
	gcc -O3 -DAMD math_speed.c $(I_AMD) $(L_AMD) -lamdlibm -o math_amd.out
	gcc -O3 -DMKL_ILP64 -m64 math_vml.c $(I_MKL) $(L_MKL) $(l_MKL) -o math_vml.out
	icc -O3 math_speed.c -lm -o math_svml.out

all: math

clean:
	rm math_libm.out math_amd.out math_vml.out math_svml.out
