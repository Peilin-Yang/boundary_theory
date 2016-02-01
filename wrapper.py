import sys,os
import math
import argparse
import json
import ast
import inspect


def gen_batch_framework(batch_script_root, label, 
    batch_pythonscript_para, paras, para_dir, max_nodes=119):

    if not os.path.exists(batch_script_root):
        os.makedirs(batch_script_root)
    if os.path.exists(para_dir):
        shutil.rmtree(para_dir)
    os.makedirs(para_dir)

    total_nodes = min( max_nodes, len(paras) ) + 1

    batch_script_fn = os.path.join( batch_script_root, 'batch_%s.qs' % label )
    python_file_fn = os.path.join( batch_script_root, 'batch_%s.py' % label )

    mills_script = MPI.MPI()
    mills_script.output_batch_qs_file( batch_script_fn, 'python %s' % python_file_fn, total_nodes )
    mills_script.output_batch_MPI_python_script(python_file_fn, "'%s', '-%s', '%s%%d' %% r" \
        % ( inspect.getfile(inspect.currentframe()), \
            batch_pythonscript_para, os.path.join(para_dir, 'node') ))

    all_paras = [paras[i::total_nodes-1] for i in xrange(total_nodes-1)]
    #print all_paras
    #print len(all_paras)
    for idx, para in enumerate(all_paras):
        if para:
            with open( os.path.join(para_dir, 'node'+str(idx+1)), 'wb') as f:
                for ele in para:
                    f.write(' '.join(ele)+'\n')

    run_batch_gen_query_command = 'qsub -l standby -q standby.q %s' % batch_script_fn
    subprocess.call( shlex.split(run_batch_gen_query_command) )


