import sys
import pandas as pd
import os
sys.path.append('../Models_mincost/')
import NCDataCost_mincost as NCDataCost

import gurobipy as gp
import gurobipy as gp
from gurobipy import GRB
import time
import itertools

class Setting:
    demandfile = str()
    RE_cell_size = dict()  # degree
    RE_plant_types = list()  # set of RE plant types considered
    REfile = dict()
    landusefile = dict()
    solver_gap = 0.001  # x100 percent
    wall_clock_time_lim = 100000  # seconds
    weather_model = str()
    print_results_header = 1
    print_detailed_results = 1
    test_name = str()
    datadir = str()
    UB_dispatchable_cap = dict()
    lost_load_thres = float()
    gas_price = float()
    storage_types = list()
    plant_types = list()
    wake = int()
    gas_plant_types = list()
    val_lost_load = float()
    val_curtail = float()
    num_y = int()
    test_year = list()
    ens_id = int()
    year_list=list()
    minCF = 0.005 # remove small numbers
    
    
Setting.RE_plant_types = ['solar-UPV', 'wind-onshore']
Setting.gas_plant_types = ['ng', 'CCGT']
Setting.plant_types = Setting.RE_plant_types
Setting.REfile['solar-UPV'] = 'DATA/Dev/CapacityFactors_ISONE/Solar/cf_Solar_0.22m_'
Setting.REfile['wind-onshore'] = 'DATA/Dev/CapacityFactors_ISONE/Wind/cf_Wind_0.22m_'
Setting.weather_model = "WR"
Setting.wake = 0 # if wake=1, then wake effect is considered
Setting.landr = 0 # if landr=1, then land restriction is considered
Setting.RE_cell_size['wind-onshore'] = 0.06 # OR wind
Setting.RE_cell_size['solar-UPV'] = 0.14 # OR solar
Setting.outputdir = '../Result_minCost/'

dat = NCDataCost.Data(Setting)