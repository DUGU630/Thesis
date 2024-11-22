# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 10:41:06 2023
This script containt the 2-step approach where the problem is eventually decoupled 
Step 1: solve a relaxed JPoNG, and get the coupling decisions (daily flow to power, emission budget for each system).
Step 2: Solve the power system with the given parameters from Step 1. 
@author: Rahman Khorramfar
"""


import time;
import gurobipy as gp;
from gurobipy import GRB, quicksum,LinExpr;
from Setting import Setting;
from SystemClasses import EV,GV;
from Setting import Setting;
from SystemClasses import EV,GV, cut;
import sys;import os;import time;import csv;
from ProblemData import Enodes,Gnodes,Branches;
from ProblemData import PipeLines,Plants,eStore,Other_input,state2zone_id,plant2sym;
from ProblemData import sym2plant,time_weights,zone_id2state,Exist_SVL,SVLs;
import numpy as np;
# from BendersFuncs import SP, MP;
import pandas as pd;

s_time = time.time();

#%% fetch data
e_time_weight, g_time_weight, g_rep_days, e_rep_hrs, days_in_cluster,days2Medoid = time_weights(Setting.num_rep_days,Setting.rep_day_folder);

nE = len(Enodes); 
nG = len(Gnodes);
nBr = len(Branches);
nPipe = len(PipeLines);
nPlt = len(Plants);
neSt = len(eStore);
Te = range(len(e_time_weight));
Tg = range(len(g_time_weight));
nPipe = len(PipeLines);
nSVL = len(Exist_SVL);
nG = len(Gnodes);
FY = np.arange(365);
thermal_units = ["ng","OCGT","CCGT","CCGT-CCS","nuclear","nuclear-new"];
NG_units = ["ng","OCGT","CCGT","CCGT-CCS"];
VRE = ["solar","wind","wind-offshore-new","solar-UPV","wind-new"];# hydro not included

step_info = np.zeros((6,3));  # columns: elapsed time, total cost, power cost
Setting.is_dispatch_model = False;
#%% Step 1: no UC, copper-plate, no integer except pipeline inv. vars
Setting.UC_active = 0;
Setting.relax_UC_vars = True;
Setting.relax_int_vars = 1;
Setting.copper_plate_approx = 1; 
init_hydro = Setting.hydro_QC;
Setting.hydro_QC = 0;
MP_LB = 0;
Model = gp.Model();
import Module;
EV.Ze = Model.addVars(nBr,vtype=GRB.CONTINUOUS);
Module.Power_System_Model(Model); # Power System
GV.ZgOp = Model.addVars(nPipe,vtype=GRB.BINARY);
Module.NG_System_Model(Model);# NG System
Module.Coupling_constraints(Model);# Coupling Constraints
Model.modelSense = GRB.MINIMIZE;
if Setting.emis_case==1:
    Model.setObjective(EV.e_system_cost);    
else:
    Model.setObjective(GV.g_system_cost+ EV.e_system_cost);
Model.setParam('OutputFlag', 0); 
Model.setParam('MIPGap', 0.02);  
Model.setParam('Timelimit', 1*3600);
Model.setParam('Presolve',2); # -1 to 2
Model.optimize();
MP_LB = EV.e_system_cost.X;
print(sys.argv);
# print(f"Power cost: {EV.e_system_cost.X}");

EV.est_cost_val = EV.est_cost.X;

EV.Xop_val = Model.getAttr('x',EV.Xop);
EV.Ze_val = Model.getAttr('x',EV.Ze);
GV.ZgOp_val = Model.getAttr('x',GV.ZgOp);

step1_time = time.time();
step_info[0,0] = step1_time-s_time;
step_info[0,1] = Model.ObjVal;
step_info[0,2] = EV.e_system_cost.X;
print(f"\t First step objective value: {np.round(Model.ObjVal/10e8,3)}, \t CPU time (sec): {np.round(step_info[0,0])}\n ");
#%% Step 2: UC added, copper-plate, no integer
Setting.UC_active = 1;
Setting.relax_UC_vars = True;
Setting.relax_int_vars = 1;
Setting.copper_plate_approx = 1; 
Setting.hydro_QC = init_hydro;
MP_LB = 0;
Model = gp.Model();
import Module;
EV.Ze = Model.addVars(nBr,vtype=GRB.CONTINUOUS);
Module.Power_System_Model(Model); # Power System
GV.ZgOp = Model.addVars(nPipe,vtype=GRB.CONTINUOUS);
Module.NG_System_Model(Model);# NG System
Module.Coupling_constraints(Model);# Coupling Constraints
Model.modelSense = GRB.MINIMIZE;
if Setting.emis_case==1:
    Model.setObjective(EV.e_system_cost);    
else:
    Model.setObjective(GV.g_system_cost+ EV.e_system_cost);
Model.setParam('OutputFlag', 0);
Model.setParam('MIPGap', 0.02);
Model.setParam('Timelimit', 1*3600);
Model.setParam('Presolve',2); # -1 to 2
for n in range(nE):
    for i in range(nPlt):
        if plant2sym[i] in thermal_units: 
            Model.addConstr(EV.Xop[n,i]>= EV.Xop_val[n,i]);

for b in range(nPipe):
    Model.addConstr(GV.ZgOp[b]==np.round(GV.ZgOp_val[b]));
    if PipeLines[b].is_exist==1:
        Model.addConstr(GV.ZgDec[b]==1-np.round(GV.ZgOp_val[b]));
Model.optimize();
MP_LB = EV.e_system_cost.X;

EV.Xop_val = Model.getAttr('x',EV.Xop);
EV.Ze_val = Model.getAttr('x',EV.Ze);
GV.ZgOp_val = Model.getAttr('x',GV.ZgOp);

step2_time = time.time();
step_info[1,0] = step2_time-step1_time;
step_info[1,1] = Model.ObjVal;
step_info[1,2] = EV.e_system_cost.X; 
print(f"\t Second step objective value: {np.round(Model.ObjVal/10e8,3)}, \t CPU time (sec): {np.round(step_info[1,0])}\n ");

#%% step 3: no UC, set thermal variables
Setting.UC_active = 0;
Setting.relax_UC_vars = True;
Setting.relax_int_vars = 1;
Setting.copper_plate_approx = 0; 
MP_LB = 0;
Model = gp.Model();
import Module;
EV.Ze = Model.addVars(nBr,vtype=GRB.CONTINUOUS);
Module.Power_System_Model(Model); # Power System
GV.ZgOp = Model.addVars(nPipe,vtype=GRB.CONTINUOUS);
Module.NG_System_Model(Model);# NG System
Module.Coupling_constraints(Model);# Coupling Constraints
Model.modelSense = GRB.MINIMIZE;
if Setting.emis_case==1:
    Model.setObjective(EV.e_system_cost);    
else:
    Model.setObjective(GV.g_system_cost+ EV.e_system_cost);
Model.setParam('OutputFlag', 0); 
Model.setParam('MIPGap', 0.01);
Model.setParam('Timelimit', 1*3600);
Model.setParam('Presolve',2); # -1 to 2
for n in range(nE):
    for i in range(nPlt):
        if plant2sym[i] in thermal_units:
            Model.addConstr(EV.Xop[n,i] == np.round(EV.Xop_val[n,i]));
        if Plants[i].is_exist==0: 
            Model.addConstr(EV.Xop[n,i] >= np.round(EV.Xop_val[n,i]));
     
for b in range(nPipe):
    Model.addConstr(GV.ZgOp[b]==np.round(GV.ZgOp_val[b]));
    if PipeLines[b].is_exist==1:
        Model.addConstr(GV.ZgDec[b]==1-np.round(GV.ZgOp_val[b]));
Model.optimize();
print(sys.argv);
# print(f"Power cost: {EV.e_system_cost.X}");    
           
# MIP_gap = 100*Model.MIPGap
# print(f"MIP Gap (%): {MIP_gap}");
EV.Xop_val = Model.getAttr('x',EV.Xop);
EV.Ze_val = Model.getAttr('x',EV.Ze);
step4_time = time.time();
step_info[2,0] = step4_time-step2_time;
step_info[2,1] = Model.ObjVal;
step_info[2,2] = EV.e_system_cost.X;
print(f"\t Third step objective value: {np.round(Model.ObjVal/10e8,3)}, \t CPU time (sec): {np.round(step_info[2,0])}\n ");
#%% Step 4: feasible solution. set all integer variables
Setting.UC_active = 1;
Setting.relax_UC_vars = True;
Setting.relax_int_vars = 1;
Setting.copper_plate_approx = 0; 
MP_LB = 0;
Model = gp.Model();
import Module;
EV.Ze = Model.addVars(nBr,vtype=GRB.CONTINUOUS);
Module.Power_System_Model(Model); # Power System
GV.ZgOp = Model.addVars(nPipe,vtype=GRB.CONTINUOUS);
Module.NG_System_Model(Model);# NG System
Module.Coupling_constraints(Model);# Coupling Constraints
Model.modelSense = GRB.MINIMIZE;
if Setting.emis_case==1:
    Model.setObjective(EV.e_system_cost);    
else:
    Model.setObjective(GV.g_system_cost+ EV.e_system_cost);
    
Model.setParam('OutputFlag', 0);
Model.setParam('MIPGap', 0.01);    
Model.setParam('Timelimit', 6*3600);
Model.setParam('Presolve',2); # -1 to 2
for n in range(nE):
    for i in range(nPlt):            
        if (plant2sym[i] in thermal_units):             
            Model.addConstr(EV.Xop[n,i] == np.round(EV.Xop_val[n,i]));
        if (Plants[i].is_exist==1): 
            Model.addConstr(EV.Xop[n,i] == EV.Xop_val[n,i]);
        
        
for b in range(nPipe):
    Model.addConstr(GV.ZgOp[b]==np.round(GV.ZgOp_val[b]));
    if PipeLines[b].is_exist==1:
        Model.addConstr(GV.ZgDec[b]==1-np.round(GV.ZgOp_val[b]));
for b in range(nBr):
    if EV.Ze_val[b]>0.3:
        Model.addConstr(EV.Ze[b] == 1);
    else:
        Model.addConstr(EV.Ze[b] == 0);
Model.optimize();
print(sys.argv);
           
# MIP_gap = 100*Model.MIPGap
# print(f"MIP Gap (%): {MIP_gap}");

step5_time = time.time();
step_info[3,0] = step5_time-step4_time;
step_info[3,1] = Model.ObjVal;
step_info[3,2] = EV.e_system_cost.X;
print(f"Fourth step objective value: {np.round(Model.ObjVal/10e8,3)}, \t CPU time (sec): {np.round(step_info[3,0])}\n ");# print(f"Power cost: {EV.e_system_cost.X}");    



#%% Step 5: solve for dispatch year
# if Setting.base_year==Setting.dispatch_year:
Module.Get_var_vals(Model);
Module.Publish_results_Approx(s_time,0,step_info);
# else:    
#     Setting.CRM_reserve = 0.0;
#     Setting.is_dispatch_model = True;
#     base_year = Setting.base_year;
#     Setting.base_year = Setting.dispatch_year;
    
#     # solve the dispatch problem of the dispatch year
#     EV.Xop_val = Model.getAttr('x',EV.Xop);
#     # EV.Xest_val = Model.getAttr('x',EV.Xest);
#     # EV.Xdec_val = Model.getAttr('x',EV.Xdec);
#     EV.Ze_val = Model.getAttr('x',EV.Ze);
#     GV.Zg_val = Model.getAttr('x',GV.Zg);
#     GV.ZgOp_val = Model.getAttr('x',GV.ZgOp);
#     GV.ZgDec_val = Model.getAttr('x',GV.ZgDec);
#     parent = os.path.join(os.getcwd(), os.pardir);    
#     parent = os.path.abspath(parent);
#     df_eDem = pd.read_csv(parent+'/Power-Gas-Load-CFs/Electricity_Load_'+Setting.electrification_scenario+ '_BaseYear'+str(Setting.dispatch_year)+'.csv');
#     df_solar = pd.read_csv(parent+'/Power-Gas-Load-CFs/AvailabilityFactors_Solar_'+str(Setting.dispatch_year)+'.csv'); 
#     df_wind = pd.read_csv(parent+'/Power-Gas-Load-CFs/AvailabilityFactors_Wind_Onshore_'+str(Setting.dispatch_year)+'.csv'); 
#     df_wind_offshore = pd.read_csv(parent+'/Power-Gas-Load-CFs/AvailabilityFactors_Wind_Offshore_'+str(Setting.dispatch_year)+'.csv'); 
#     df_ng_dem = pd.read_csv(parent+'/Power-Gas-Load-CFs/NG_Load_'+Setting.electrification_scenario+'_BaseYear'+str(Setting.dispatch_year)+'.csv');
    
#     for i in range(nE):
#         Enodes[i].demand = np.array(df_eDem[str(i)]); # directly using 2050 load
#         Enodes[i].cap_factors = np.ones((8760,len(sym2plant.keys())));
#         Enodes[i].cap_factors[:,sym2plant['solar']] = np.array(df_solar.iloc[:,i]);
#         Enodes[i].cap_factors[:,sym2plant['solar-UPV']] = np.array(df_solar.iloc[:,i]);
#         Enodes[i].cap_factors[:,sym2plant['wind']] = np.array(df_wind.iloc[:,i]);
#         Enodes[i].cap_factors[:,sym2plant['wind-new']] = np.array(df_wind.iloc[:,i]);
#         Enodes[i].cap_factors[:,sym2plant['wind-offshore-new']] = np.array(df_wind_offshore.iloc[:,i]); 
#     for i in range(nG):
#         Gnodes[i].demand = np.array(df_ng_dem[str(i)]);
#     Setting.UC_active = 1;
#     Setting.relax_UC_vars = True;
#     Setting.relax_int_vars = 1;
#     Setting.copper_plate_approx = 0; 
#     MP_LB = 0;
#     Model = gp.Model();
#     import Module;
#     EV.Ze = Model.addVars(nBr,vtype=GRB.CONTINUOUS);
#     Module.Power_System_Model(Model); # Power System
#     GV.ZgOp = Model.addVars(nPipe,vtype=GRB.CONTINUOUS);
#     Module.NG_System_Model(Model);# NG System
#     Module.Coupling_constraints(Model);# Coupling Constraints
#     Model.modelSense = GRB.MINIMIZE;
#     if Setting.emis_case==1:
#         Model.setObjective(EV.e_system_cost);    
#     else:
#         Model.setObjective(GV.g_system_cost+ EV.e_system_cost);
        
#     Model.setParam('OutputFlag', 0);
#     Model.setParam('MIPGap', 0.01);    
#     Model.setParam('Timelimit', 4*3600);
#     Model.setParam('Presolve',2); # -1 to 2
#     for n in range(nE):
#         for i in range(nPlt):
#             Model.addConstr(EV.Xop[n,i] == EV.Xop_val[n,i]);            
#             # if (plant2sym[i] in thermal_units):             
#             #     Model.addConstr(EV.Xop[n,i] == np.round(EV.Xop_val[n,i]));
#             # if (Plants[i].is_exist==1): 
#             #     Model.addConstr(EV.Xop[n,i] == EV.Xop_val[n,i]);
#             # else:
#             #     Model.addConstr(EV.Xop[n,i] == np.round(EV.Xop_val[n,i]));
            
#     for b in range(nPipe):
#         Model.addConstr(GV.ZgOp[b]==np.round(GV.ZgOp_val[b]));
#         if PipeLines[b].is_exist==1:
#             Model.addConstr(GV.ZgDec[b]==1-np.round(GV.ZgOp_val[b]));
#     for b in range(nBr):
#         Model.addConstr(EV.Ze[b] == EV.Ze_val[b]);

#     Model.optimize();
#     print(sys.argv);
               
#     # MIP_gap = 100*Model.MIPGap
#     # print(f"MIP Gap (%): {MIP_gap}");

#     step5_time = time.time();
#     step_info[4,0] = step5_time-step4_time;
#     step_info[4,1] = Model.ObjVal;
#     step_info[4,2] = EV.e_system_cost.X;
#     print(f"Dispatch problem (Step 5) objective value: {np.round(Model.ObjVal/10e8,3)}, \t CPU time (sec): {np.round(step_info[3,0])}\n ");# print(f"Power cost: {EV.e_system_cost.X}");    
#     Setting.base_year = base_year;
#     Module.Get_var_vals(Model);
#     Module.Publish_results_Approx(s_time,0,step_info);

    