import sys, os
import shutil
import subprocess
import shlex
from string import Template

qs_file_content = Template("""
#
# Template:  OpenMPI, Default (OpenIB Infiniband) Variant
# Revision:  $$Id: openmpi-ib.qs 525 2014-10-14 17:34:23Z frey $$
#
# Usage:
# 1. Modify "NPROC" in the -pe line to reflect the number
#    of processors desired.
# 2. Modify the value of "MY_EXE" to be your MPI program and any
#    arguments to be passed to it.
# 3. Uncomment the WANT_CPU_AFFINITY line if you want Open MPI to
#    bind workers to processor cores.  Can increase your program's
#    efficiency.
# 4. Uncomment the SHOW_MPI_DEBUGGING line if you want very verbose
#    output written to the Grid Engine output file by OpenMPI.
# 5. If you use exclusive=1, please be aware that NPROC will be
#    rounded up to a multiple of 20.  In this case, set the
#    WANT_NPROC variable to the actual core count you want.  The
#    script will "load balance" that core count across the N nodes
#    the job receives.
# 6. Jobs default to using 1 GB of system memory per slot.  If you
#    need more than that, set the m_mem_free complex.
# 
#$$-pe mpi $procs
#
#
# The standby flag asks to run the job in a standby queue.
#$$ -l standby=1
#
# Change the following to #$$ and set the amount of memory you need
# per-slot if you're getting out-of-memory errors using the
# default:
#$$ -l m_mem_free=$memory
#
#
# Change the following to #$$ if you want exclusive node access
# (see 5. above):
# -l exclusive=1
#
# If you want an email message to be sent to you when your job ultimately
# finishes, edit the -M line to have your email address and change the
# next two lines to start with #$$ instead of just #
#$$ -m eas
#$$ -M franklyn@udel.edu 
#

#
# Setup the environment; choose the OpenMPI version that's
# right for you:
#
vpkg_require openmpi/1.8.2
vpkg_require python-matplotlib
vpkg_require python-scipy
vpkg_require python-mpi4py

#
# The MPI program to execute and any arguments to it:
#
MY_EXE="$s"

#
# By default the slot count granted by Grid Engine will be
# used, one MPI worker per slot.  Set this variable if you
# want to use fewer cores than Grid Engine granted you (e.g.
# when using exclusive=1):
#
#WANT_NPROC=0

#
# Ask Open MPI to do processor binding?
#
WANT_CPU_AFFINITY=YES

#
# Uncomment to enable lots of additional information as OpenMPI executes
# your job:
#
#SHOW_MPI_DEBUGGING=YES

##
## You should NOT need to change anything after this comment.
##
OPENMPI_FLAGS="--display-map --mca btl ^tcp"
if [ "$${WANT_CPU_AFFINITY:-NO}" = "YES" ]; then
  OPENMPI_FLAGS="$${OPENMPI_FLAGS} --bind-to core"
fi
if [ "$${WANT_NPROC:-0}" -gt 0 ]; then
  OPENMPI_FLAGS="$${OPENMPI_FLAGS} --np $${WANT_NPROC} --map-by node"
fi
if [ "$${SHOW_MPI_DEBUGGING:-NO}" = "YES" ]; then
  OPENMPI_FLAGS="$${OPENMPI_FLAGS} --debug-devel --debug-daemons --display-devel-map --display-devel-allocation --mca mca_verbose 1 --mca coll_base_verbose 1 --mca ras_base_verbose 1 --mca ras_gridengine_debug 1 --mca ras_gridengine_verbose 1 --mca btl_base_verbose 1 --mca mtl_base_verbose 1 --mca plm_base_verbose 1 --mca pls_rsh_debug 1"
  if [ "$${WANT_CPU_AFFINITY:-NO}" = "YES" ]; then
    OPENMPI_FLAGS="$${OPENMPI_FLAGS} --report-bindings"
  fi
fi


echo "GridEngine parameters:"
echo "  mpirun        = "`which mpirun`
echo "  nhosts        = $$NHOSTS"
echo "  nproc         = $$NSLOTS"
echo "  executable    = $$MY_EXE"
echo "  MPI flags     = $$OPENMPI_FLAGS"
echo "-- begin OPENMPI run --"
mpirun $${OPENMPI_FLAGS} $$MY_EXE
echo "-- end OPENMPI run --"


""")

python_script_content = Template("""
from mpi4py import MPI
import subprocess

comm = MPI.COMM_WORLD

if comm.rank == 0:
   pass

for r in xrange(1, comm.size):
    if comm.rank == r:
        subprocess.call(['python', $s])
""")

class MPI():
    def __init__(self):
        pass

    def output_batch_qs_file(self, fn, command, _procs=120, _memory='2G'):
        #print fn, command
        with open(fn, 'wb') as f:
            f.write(qs_file_content.substitute(s=command, procs=_procs, memory=_memory))

    def output_batch_MPI_python_script(self, fn, script_path_and_options):
        #print fn, script_path_and_options
        with open(fn, 'wb') as f:
            f.write(python_script_content.substitute(s=script_path_and_options))


    def gen_batch_framework(self, batch_script_root, label, real_program_file_name, 
      batch_pythonscript_para, paras, para_dir, para_alreay_split=False, add_node_to_para=False, 
      node_para_prefix="", max_nodes=119, memory=2, run_after_gen=False):
      """
      para_alreay_split: parameters has already split into the format: [[para1, para2], [para1, para2]]
      add_node_to_para: add the node number, e.g. node1, node2 as the last parameter
      node_para_prefix: add the path prefix to add_node_to_para(node1), this is mainly for intermediate output results
      """


      if not os.path.exists(batch_script_root):
          os.makedirs(batch_script_root)

      para_dir = os.path.abspath(para_dir)
      if os.path.exists(para_dir):
          shutil.rmtree(para_dir)
      os.makedirs(para_dir)

      total_nodes = min( max_nodes, len(paras) ) + 1

      batch_script_fn = os.path.abspath(os.path.join( batch_script_root, 'batch_%s.qs' % label ))
      python_file_fn = os.path.abspath(os.path.join( batch_script_root, 'batch_%s.py' % label ))

      self.output_batch_qs_file( batch_script_fn, 'python %s' % python_file_fn, total_nodes, memory )
      self.output_batch_MPI_python_script(python_file_fn, "'%s', '-%s', '%s%%d' %% r" \
          % ( os.path.abspath(real_program_file_name), \
              batch_pythonscript_para, os.path.join(para_dir, 'node') ))

      if para_alreay_split:
        all_paras = paras
      else:
        all_paras = [paras[i::total_nodes-1] for i in xrange(total_nodes-1)]

      #print all_paras
      #print len(all_paras)
      for idx, para in enumerate(all_paras):
          if para:
              with open( os.path.join(para_dir, 'node'+str(idx+1)), 'wb') as f:
                  for ele in para:
                      #print ele
                      f.write(' '.join(ele))
                      if add_node_to_para:
                        f.write(' '+node_para_prefix+'/node'+str(idx+1))
                      f.write('\n')

      run_batch_gen_query_command = 'qsub -l standby -q standby.q %s' % batch_script_fn
      if run_after_gen:
        subprocess.call( shlex.split(run_batch_gen_query_command) )

