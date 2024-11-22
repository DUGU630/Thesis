# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 13:55:42 2022
@author: Rahman Khorramfar
"""

import multiprocessing
import sys;
from subprocess import PIPE, Popen;
import  os;
import subprocess;
import numpy as np;


def system(command):
    process = Popen(
        args=command,
        stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
        bufsize=0,
        universal_newlines=True,
        shell=True);

    while True:
        nextline = process.stdout.readline()
        if nextline == '' and process.poll() is not None:
            break
        sys.stdout.write(nextline)
        sys.stdout.flush()

    output = process.communicate()[0]
    exitCode = process.returncode

    if (exitCode == 0):
        return output
    else:
        return -1#raise ProcessException(command, exitCode, output)

    return process.communicate()[0]


folder = 'Representative_Days';
net_size = 17;
BY = np.arange(2001,2021);
DY = np.arange(2001,2021);
rep_days = [24];
#rep_days = np.insert(rep_days,len(rep_days),365);
case = [4];
elec_scen = ['ME','MX','HE','HX','RF'];
emis_reduc_goal = [0.8,0.95];
VRE_share = [0.0];
solver_gap = 0.01;
wall_clock_time_lim = 1; #hours;
UC_active = 1;
relax_UC_vars = 1;
relax_int_vars = 0;
solver_thread_num = 1;
metal_air_cost = ['no-metal-air'];
CCS_allowed = [1];
hydro_QC = 1;
param_list=[];
SuperClound_Thread = 6;
# for i0 in BY:
#     for i2 in case:
#         for i1 in rep_days:
#             for i4 in emis_reduc_goal:
#                 for i3 in elec_scen:            
#                     for i5 in VRE_share:
#                         for i6 in metal_air_cost:
#                             for i7 in CCS_allowed:                                   
#                                 param = str(net_size)+'-'+str(i1)+'-'+i3+\
#                                     '-'+str(i2)+'-'+str(i4)+'-'+str(i5)+'-BY'+str(i0)+\
#                                     '-DY'+str(i0)+'-CCS'+str(i7)+'-hydro'+str(1)+'.csv';
#                                 param_list.append(param);                                
   
                   
system('cd '+os.getcwd());
system('scp -r rkhorramfar@txe1-login.mit.edu:/home/gridsan/rkhorramfar/JPoNG-CRS/JPoNG_Results.csv ./')
system('scp -r rkhorramfar@txe1-login.mit.edu:/home/gridsan/rkhorramfar/JPoNG-CRS/log.txt ./')
# system('scp -r rkhorramfar@txe1-login.mit.edu:/home/gridsan/rkhorramfar/JPoNG_R1/17-24-HX-4-0.95-0.0-BY2003-DY2003-CCS1-hydro1.csv ./')

# for param in param_list:
#     system('scp -r rkhorramfar@txe1-login.mit.edu:/home/gridsan/rkhorramfar/JPoNG/'+param+' ./');
    
    
