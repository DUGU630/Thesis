import multiprocessing
import sys
from subprocess import PIPE, Popen
import  os
import subprocess
import numpy as np;

 
def system(command):  
    process = Popen(
        args=command,
        stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
        bufsize=0,
        universal_newlines=True,
        shell=True
    )

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

if __name__ == "__main__":
    import time;
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor;
    from datetime import datetime;
    
    folder = 'Representative_Days';
    net_size = 17;
    BY = np.arange(2003,2004);
    DY = np.arange(2003,2004);
    rep_days = np.arange(30,31);
    #rep_days = np.insert(rep_days,len(rep_days),365);
    case = [4];
    elec_scen = ['ME','HE','RF','MX','HX'];
    # elec_scen = ['HX'];

    emis_reduc_goal = [0.8,0.95];
    # emis_reduc_goal = [0.95];

    VRE_share = [0.0];
    solver_gap = 0.01;
    wall_clock_time_lim = 12; #hours;
    UC_active = 0;
    relax_UC_vars = 1;
    relax_int_vars = 1;    
    metal_air_cost = ['no-metal-air'];
    CCS_allowed = [1];
    hydro_QC = 1;
    
    param_list=[];
    solver_thread_num = 4;
    SuperClound_Thread = 48;

    for i0 in BY:
        for i8 in DY:
            for i2 in case:
                for i1 in rep_days:
                    for i4 in emis_reduc_goal:
                        for i3 in elec_scen:            
                            for i5 in VRE_share:
                                for i6 in metal_air_cost:
                                    for i7 in CCS_allowed: 
                                        if i0!=i8:continue;
                                        param = 'python Main.py '+folder+' '+str(net_size)+' ' +str(i0)+' ' +str(i8) +' '+ str(i1)+' '+str(i2)+' '+str(i3)+' '+str(i4)+' '+str(i5)+' '+str(solver_gap)+' '+str(wall_clock_time_lim)+' '+str(UC_active)+' '+str(relax_UC_vars)+' '+str(relax_int_vars)+' '+str(solver_thread_num)+' '+i6+' '+str(i7)+' '+str(hydro_QC);
                                        param_list.append(param);   

    # del i1,i2,i3,i4,i5;       
    #print(param_list)
    for i in range(int(np.ceil(solver_thread_num*len(param_list)/SuperClound_Thread))):
        j = int(SuperClound_Thread/solver_thread_num);
        with ThreadPoolExecutor() as executor:
            results = executor.map(system, param_list[i*j:(i+1)*j]);
    # for param in param_list:
    #      tmp = system(param)
    # print(tmp)

#%%
# import time
# from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
# from datetime import datetime

# now = datetime.now().time() # time object

# print("now =", now)
# def sleep_secs(seconds):
#   time.sleep(seconds)
#   print(f'{seconds} has been processed')

# secs_list = [2,4, 6, 8, 10, 12];
# with ThreadPoolExecutor() as executor:
#   results = executor.map(sleep_secs, secs_list)
#   print(results)

# now = datetime.now().time() # time object

# print("now =", now)