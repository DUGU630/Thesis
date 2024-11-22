# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 11:20:38 2022

@author: Rahman Khorramfar
"""

class Setting:    
    rep_day_folder = str();    
    Power_network_size = [];
    num_rep_days = int();
    emis_case = int();
    electrification_scenario = str();  #HM or RM (High-moderate or reference-moderate);
    emis_reduc_goal = float();
    VRE_share = float();
    CRM_reserve = float(); # capacity reserve margin [fraction]
    Metal_air_storage_cost = str(); # high, medium, low
    solver_gap = float();
    wall_clock_time_lim = int();
    solver_thread_num = int(); # number of thread solver uses
    base_year = int();  # the year on which the load projection is carried out [2000-2020]
    expansion_allowed = bool();
    CCS_allowed = bool();  # whether CCGC-CCS plants are allowed to be established
    hydro_QC = bool();  # hydro import/export from/to Quebeck allowed 
    poss_gas_consump = float();
    #poss_emis = float();
    CO2_emission_1990 = float();
    e_emis_lim = float();
    g_emis_lim = float();
    
    UC_active = bool();
    relax_int_vars = bool();
    relax_UC_vars = bool();
    print_result_header = bool();
    copper_plate_approx = bool();
    print_all_vars = bool();
    e_shed_penalty = float();
    g_shed_penalty = float();
    print_all_vars = bool();
    
    
    
#emission_power_plants = ; # without coupling. Just run case1 and see the power emission
#total_ng_yearly_demand = 5.32E+08;# (4.86 in 2016) 5.49E+08;# MMBtu in 2016
#total_yearly_gen_demand = 1.50E+07 / 0.053; # 1.23E+09
#Setting.poss_gas_consump = total_ng_yearly_demand + total_yearly_gen_demand;#1.00E+09 MMBTu possible gas consumption
#Setting.poss_emis = Setting.poss_gas_consump * 0.053; #5.3e7 tons  (previsouly 8.2e7 kg)

Setting.e_emis_lim = 43.9e6; #tons of CO2
Setting.g_emis_lim = 23.6e6;
Setting.CO2_emission_1990 =Setting.e_emis_lim+ Setting.g_emis_lim; #tons of CO2 for electricity and NG

# default values
Setting.Power_network_size=6;
Setting.num_rep_days = 2;











    

