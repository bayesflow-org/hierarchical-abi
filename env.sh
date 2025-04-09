
module purge

module load gnu12/12.2.0

module load \
     linux-rocky8-x86_64/gcc/12.2.0/python/3.11.9-ofsncoj \
     linux-rocky8-x86_64/gcc/12.2.0/swig/4.1.1-qarng36 \
     linux-rocky8-x86_64/gcc/12.2.0/openblas/0.3.21-zy2y7i4 \
     linux-rocky8-x86_64/gcc/12.2.0/hdf5/1.12.1-cdxzvhd \
     linux-rocky8-x86_64/gcc/12.2.0/boost/1.81.0-5k3pf6l
module load linux-rocky8-x86_64/gcc/12.2.0/cuda/11.8.0-kbkq5ya 

source .venv/bin/activate

