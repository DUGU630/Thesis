# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 19:29:25 2022

@author: Rahman Khorramfar
Allah sene tevekkul
"""
# from IPython import get_ipython;
# get_ipython().magic('reset -f') # to clear the namespace
# get_ipython().magic('clear');
import numpy as np;
from Setting import Setting;
# from SystemClasses import EV,GV;
import pandas as pd;
import os;
# Setting.Power_network_size = 6; 
# Setting.base_year=2011;
# Setting.rep_day_folder = 'joint_CF_with_extreme_days';#extreme_days
# Setting.electrification_scenario = 'HE';  
# from geopy.distance import distance;
def time_weights(nD,fName):   
    #df = pd.read_csv(os.getcwd()+r'\Rep_Days\num_rep_days='+str(nD)+'.csv');
    #df = pd.read_csv(os.getcwd() + '/' + fName+'/'+fName +'_medians_k='+str(nD)+'.csv');
    
    df = pd.read_csv(os.getcwd() + '/' + fName+'/Rep-Days-Year='+str(Setting.base_year)+'/cluster_year='+str(Setting.base_year)+'_days='+str(nD)+'.csv');

    s1 = df[df['is_medoid?']==1];
    s2 = s1.sort_values(['Cluster']);
    g_rep_days = np.array(s2.index);    
    e_time_weight = np.zeros(24*nD,dtype=int);
    e_rep_hrs = np.zeros(24*nD,dtype=int);
    g_time_weight = np.zeros(nD,dtype=int);    
    days_in_cluster = dict();
    days2Medoid = dict();
    
    # sort g_rep_days and g_weights
    g_rep_days = sorted(g_rep_days);
    for i in range(nD):
        s1 = np.array(df['Cluster']);
        i2 = df['Cluster'].iloc[g_rep_days[i]];
        s2 = np.where(s1==i2);
        g_time_weight[i]= len(s2[0]);
        for d in s2[0]:
            days2Medoid[d] = g_rep_days[i];
        days_in_cluster[i] = s2[0];
        e_time_weight[i*24:(i+1)*24] += g_time_weight[i];
        for j in range(24):
             e_rep_hrs[i*24+j] =g_rep_days[i]*24+j;
    return e_time_weight,g_time_weight,g_rep_days,e_rep_hrs,days_in_cluster,days2Medoid;


# Pow_dir = os.getcwd()+r'\Power_System_Data';
# Gas_dir = os.getcwd()+r'\Gas_System_Data';
parent = os.path.join(os.getcwd(), os.pardir);    
parent = os.path.abspath(parent);
Pow_dir = os.getcwd()+'/Power_System_Data';
Gas_dir = os.getcwd()+'/Gas_System_Data';
dfo = pd.read_csv('Other_Parameters.csv');
plant2sym = {0:'ng',1:'solar',2:'wind', 3:'hydro',4:'nuclear',5:'OCGT',6:'CCGT',7:'CCGT-CCS',8:'solar-UPV',9:'wind-new',10:'wind-offshore-new',11:'nuclear-new'};
sym2plant = {'ng':0,'solar':1,'wind':2,'hydro':3,'nuclear':4,'OCGT':5,'CCGT':6,'CCGT-CCS':7,'solar-UPV':8,'wind-new':9,'wind-offshore-new':10,'nuclear-new':11};

# this is the order in Breakthrough data
zone_id2state = {0:'Maine', 1:'New Hampshire',2:'Vermont',
               3:'Massachusetts',4:'Rhode Island',5:'Connecticut'};
state2zone_id = {'Maine':0, 'New Hampshire':1,'Vermont':2,
               'Massachusetts':3,'Rhode Island':4,'Connecticut':5};



df_ebus = pd.read_csv(Pow_dir+'/Power_Nodes.csv');
#df_eAdj = pd.read_csv(Pow_dir+'/bus_adj_Nodes-6-nodes.csv');
df_ePlt = pd.read_csv(Pow_dir+'/Plants_Nodes.csv');
df_br = pd.read_csv(Pow_dir+'/Transmission_Lines.csv');
df_eDem = pd.read_csv(parent+'/Power-Gas-Load-CFs/Electricity_Load_'+Setting.electrification_scenario+ '_BaseYear'+str(Setting.base_year)+'.csv');
df_solar = pd.read_csv(parent+'/Power-Gas-Load-CFs/AvailabilityFactors_Solar_'+str(Setting.dispatch_year)+'.csv'); 
df_wind = pd.read_csv(parent+'/Power-Gas-Load-CFs/AvailabilityFactors_Wind_Onshore_'+str(Setting.dispatch_year)+'.csv'); 
df_wind_offshore = pd.read_csv(parent+'/Power-Gas-Load-CFs/AvailabilityFactors_Wind_Offshore_'+str(Setting.dispatch_year)+'.csv'); 
df_reg_mult = pd.read_csv(Pow_dir+'/Regional_multipliers.csv');
df_plt = pd.read_csv(Pow_dir+'/Plant_params.csv');
df_str = pd.read_csv(Pow_dir+'/Storage_params.csv');
df_ccs = pd.read_csv(Pow_dir+'/CCS_params.csv');

df_gnode = pd.read_csv(Gas_dir+'/NG_Nodes.csv');
# df_ng_Lexp = pd.read_csv(Gas_dir+'/ng_L_exp.csv');
# df_ng_Limp = pd.read_csv(Gas_dir+'/ng_L_imp.csv');      
df_ng_adjE = pd.read_csv(Gas_dir+'/NG_AdjE_Nodes.csv');
df_g2g_br = pd.read_csv(Gas_dir+'/NG2NG_Pipelines.csv');
df_ng_dem = pd.read_csv(parent+'/Power-Gas-Load-CFs/NG_Load_'+Setting.electrification_scenario+'_BaseYear'+str(Setting.base_year)+'.csv');
df_exis_svl = pd.read_csv(Gas_dir+'/SVL_data.csv');
df_svl_params = pd.read_csv(Gas_dir+'/SVL_params.csv');

#%% other parameters
class other_prm():
    type_prod_lim = np.array(dfo['resource_avail (solar,wind,offshore,nuclear,hydro)[MW]'].iloc[:5]);# [22e3, 10e3, 280e3,3.5e3,2.6e3]; # 10e6 means no limit
    SVL_lifetime = int(dfo['SVL_lifetime'][0]);
    pipeline_lifetime = int(dfo['pipeline_lifetime'][0]);
    WACC = float(dfo['WACC'][0]);
    RNG_price = float(dfo['LCDF_price ($/MMBtu)'][0]);
    trans_unit_cost = float(dfo['trans_line_inv_cost (&/MW/mile)'][0]);
    trans_line_lifespan = int(dfo['trans_line_lifetime'][0]);
    NG_price = float(dfo['NG_price ($/MMBtu)'][0]);
    Nuclear_price = float(dfo['Uranium_price ($/MMBtu)'][0]);
    pipe_per_mile = float(dfo['pipeline_inv_cost ($/mile)'][0]);
    NG_emission = float(dfo['NG_emission (ton/MW)'][0]); 
    trans_FOM = float(dfo['trans_line_FOM ($/MW/mile)'][0]);
    Pennsylvania_carbon_storage_location =np.array(dfo['Pennsylvania_carbon_storage_location (lat-lon)'].iloc[:2]);
    pipline_FOM = float(dfo['pipeline_FOM ($/mile)'][0]);
    pipeline_decom = float(dfo['pipeline_decom_cost ($/mile)'][0]);
    carbon_str_cap = float(dfo['carbon_storage_cap (ton/year)'][0]);
Other_input = other_prm();


#%% enode
class enode:    
    # def __init__(self,n)    
    num = [];  # scalar
    adj_buses = np.array([]);
    Init_plt_type = np.array([]);
    Init_plt_count = np.array([]);
    demand  = np.array([]); 
    cap_factors = np.array([]);
    arcs = [];
    arc_sign= [];
    lat_lon = np.array([]);
    offshore_allowd = bool();  

Enodes = list();

for i in range(len(df_ebus)):
    en = enode();
    en.num = i;
    en.lat_lon = np.array([df_ebus['Lat'].iloc[i],df_ebus['Lon'].iloc[i]]);
    en.Init_plt_type = sym2plant.keys();
    en.offshore_allowd = df_ebus['Offshore_wind_allowed'].iloc[i];
    en.Init_plt_count = np.zeros(len(sym2plant.keys()),dtype=int);    
    en.demand = np.array(df_eDem[str(i)]); # directly using 2050 load
    en.cap_factors = np.ones((8760,len(sym2plant.keys())));
    en.cap_factors[:,sym2plant['solar']] = np.array(df_solar.iloc[:,i]);
    en.cap_factors[:,sym2plant['solar-UPV']] = np.array(df_solar.iloc[:,i]);
    en.cap_factors[:,sym2plant['wind']] = np.array(df_wind.iloc[:,i]);
    en.cap_factors[:,sym2plant['wind-new']] = np.array(df_wind.iloc[:,i]);
    en.cap_factors[:,sym2plant['wind-offshore-new']] = np.array(df_wind_offshore.iloc[:,i]); 
    Enodes.append(en);
    
for i in range(len(df_ePlt)):
    s1  = df_ePlt['type'][i];
    s2 = int(df_ePlt['node_id'][i]);
    if s1 in sym2plant.keys():
        Enodes[s2].Init_plt_count[sym2plant[s1]] = df_ePlt['count'][i];
            
    
#%% plants
class plant:
    Type = '';
    num = [];
    is_exist = [];
    capex = [];
    VOM = [];
    FOM = [];
    co2_capture_rate = [];
    heat_rate=[];
    lifetime= 0;
    decom_cost = [];
    nameplate_cap = [];
    min_output = [];
    ramp_rate= [];
    startup_cost=[];
    startup_fuel=[];
    est_cost_coef = np.array([]);
    emission=[];
    
    regional_mult = np.array([]);


Plants = list();
for i in range(len(sym2plant.keys())):
    plt = plant();
    plt.Type = plant2sym[i];
    plt.num = i;
    plt.is_exist = df_plt['is existing'][i];
    plt.capex = df_plt['CAPEX per plant'][i];
    plt.VOM = df_plt['VOM ($/MWh)'][i];
    plt.co2_capture_rate = df_plt['Carbon capture rate'][i];
    plt.heat_rate = df_plt['Heat Rate  (MMBtu/MWh)'][i];
    plt.lifetime = df_plt['Lifetime (year)'][i];
    plt.decom_cost = df_plt['Decom. cost ($) per plant'][i]/10; # distributed over 10 years
    plt.nameplate_cap = df_plt['Nameplate capacity (MW)'][i];
    plt.FOM = df_plt['FOM ($/kW-yr)'][i]*plt.nameplate_cap*1000;
    plt.min_output = df_plt['Minimum stable output (%)'][i];
    plt.ramp_rate = df_plt['Hourly Ramp rate (%)'][i];
    plt.startup_cost = df_plt['Startup Cost (per  plant)'][i];
    plt.startup_fuel = df_plt['Startup Fuel (MMBtu)'][i];
    plt.emission = (1-plt.co2_capture_rate)*Other_input.NG_emission;
    if plt.is_exist==0:
        plt.est_cost_coef = np.zeros(len(df_ebus));
        plt.regional_mult = np.zeros(len(df_ebus));
        for n in range(len(df_ebus)):
            plt.regional_mult[n] = df_reg_mult[df_ebus['State'].iloc[n]].iloc[i-5];
            s1 = (1/(1+Other_input.WACC)**plt.lifetime);
            plt.est_cost_coef[n] = (Other_input.WACC/(1-s1))* plt.regional_mult[n];
    # else:
    #     plt.regional_mult = np.zeros(len(state2zone_id.keys()));
    #     plt.est_cost_coef = np.zeros(len(state2zone_id.keys()));
    
    
    Plants.append(plt);
    

#%% branch
class branch:
    from_node = [];
    to_node= [];
    suscept = [];
    maxFlow = [];
    length = [];
    is_exist = [];
    est_coef = [];
    trans_FOM = [];
Branches = [];

arcs = [[] for x in range(len(Enodes))];
arc_sign = [[] for x in range(len(Enodes))];

for b in range(len(df_br)):
    br = branch();
    br.from_node = int(df_br['from_node'][b]);
    br.to_node = int(df_br['to_node'][b]);
    
    arcs[br.from_node].append(b);
    arcs[br.to_node].append(b);
    if br.from_node>br.to_node:
        arc_sign[br.from_node].append(-1);
        arc_sign[br.to_node].append(1);
    else:
        arc_sign[br.from_node].append(1);
        arc_sign[br.to_node].append(-1);
    
    br.suscept = df_br['susceptance'][b];
    br.maxFlow = df_br['maxFlow'][b];
    br.length = df_br['length'][b];
    br.is_exist = df_br['is_existing'][b];
    s1 = (1/(1+Other_input.WACC)**Other_input.trans_line_lifespan);
    br.est_coef = (Other_input.WACC/(1-s1));
    br.trans_FOM = Other_input.trans_FOM; # $/MW/mile
    Branches.append(br);
    
for i in range(len(arcs)):
    Enodes[i].arcs = np.array(arcs[i]);
    Enodes[i].arc_sign = np.array(arc_sign[i]);

#%% power storage
class eStorage:
    tech = str();
    energy_capex= float();
    power_capex = float();
    eff_ch = float();
    eff_disCh = float();
    eFOM = float();
    pFOM = float();
    lifetime = int();
    self_discharge = float();
    est_coef = float();
    

eStore = list();
str_techs = np.zeros(2); # Li-ion + metal-air with a cost assumption (high, medium, low)
str_techs[0] = 0;
if Setting.Metal_air_storage_cost=='low':
    str_techs[1]=1;
if Setting.Metal_air_storage_cost=='medium':
    str_techs[1]=2;
if Setting.Metal_air_storage_cost=='high':
    str_techs[1]=3;
if Setting.Metal_air_storage_cost=='no-metal-air':
        str_techs=np.zeros(1);

for i in str_techs:
    st = eStorage();
    st.tech = df_str['Storage technology'][i];
    st.energy_capex = df_str['energy capex'][i];
    st.power_capex = df_str['power capex'][i];    
    st.eff_ch = df_str['charging efficiency'][i];
    st.eff_disCh = df_str['discharging efficiency'][i];
    st.eFOM = df_str['energy FOM'][i];
    st.pFOM = df_str['power FOM'][i];
    st.lifetime = int(df_str['lifetime'][i]);
    st.self_discharge = df_str['self-discharge'][i];
    s1 = (1/(1+Other_input.WACC)**st.lifetime);
    st.est_coef = (Other_input.WACC/(1-s1));
    eStore.append(st);
    
    
    
#%% CC-CCS data
class CCS:
    pipe_capex = float();
    str_capex = []; #float
    elec_req_pipe = [];
    elec_req_pump = [];
    comp_dis = [];
    str_loc = np.zeros(2);  #1x2 array
    str_dis = []; #1xlen(nodes) array

CC_CCS = CCS();
CC_CCS.str_loc = [df_ccs['str_loc_lat'][0],df_ccs['str_loc_lon'][0]];
CC_CCS.pipe_capex = df_ccs['lev_inv_pipe'][0];
CC_CCS.str_capex = df_ccs['lev_inv_str'][0];
CC_CCS.elec_req_pipe = df_ccs['E_pipe'][0];
CC_CCS.elec_req_pump = df_ccs['E_pump'][0];
CC_CCS.comp_dis = df_ccs['comp_dis'][0];

CC_CCS.node2str_dis = np.zeros(len(df_ebus));
CC_CCS.node2str_dis=np.zeros(len(Enodes));
for i in range(len(df_ebus)):
    # to avoid using a new package to calculate distance
    CC_CCS.node2str_dis[i]= 45.1*np.linalg.norm(Enodes[i].lat_lon-Other_input.Pennsylvania_carbon_storage_location)
   
    # CC_CCS.node2str_dis[i] = distance(CC_CCS.str_loc,lat_lon).miles;

#%% Gas system: 
    
class gnode:
    num = int();
    fips = int();
    state = str();
    lat_lon = np.array([]);
    outside_region = bool();
    demand = np.array([]);
    injU = float();
    L_exp = np.array([]);
    L_imp = np.array([]);
    adjE = np.array([]);
    adjS = np.array([]);

Gnodes = list();

# ng_dem=0;
for i in range(len(df_gnode)):
    gas = gnode();
    gas.num = i;
    gas.fips = int(df_gnode['FIPS'][i]) if np.isnan(df_gnode['FIPS'][i])==False else [];
    gas.state = df_gnode['State'];
    gas.lat_lon = np.array([df_gnode['Lat'].iloc[i],df_gnode['Lon'].iloc[i]]);
    gas.outside_region = df_gnode['outside_region?'][i];
    gas.injU = df_gnode['inj_capacity (MMBtu/day)'][i];
    gas.adjS =np.array([int(df_gnode['SVL'][i])]);
    
    # s1 = np.array(df_ng_Lexp.iloc[i,:]);
    # s1 = s1[np.logical_not(np.isnan(s1))];
    gas.L_exp = [];
    
    # s1 = np.array(df_ng_Limp.iloc[i,:]);
    # s1 = s1[np.logical_not(np.isnan(s1))];
    gas.L_imp = [];
    
    s1 = np.array(df_ng_adjE.iloc[i,:]);
    s1 = s1[np.logical_not(np.isnan(s1))];
    gas.adjE = s1.astype(int) if len(s1)>0 else [];
    gas.demand = np.array(df_ng_dem[str(i)]);
    # ng_dem += np.sum(gas.demand);
    Gnodes.append(gas);

for i in range(len(df_g2g_br)):
    Gnodes[int(df_g2g_br['from_node'].iloc[i])].L_exp.append(i);
    Gnodes[int(df_g2g_br['to_node'].iloc[i])].L_imp.append(i);


class pipe:
    from_node = int();
    to_node = int();
    is_exist = int();
    length = float();
    Cap = float();
    inv_coef =float();
    FOM=float();
    decom = float();
    

PipeLines = list();
for i in range(len(df_g2g_br)):
    pp = pipe();
    pp.from_node = df_g2g_br['from_node'][i];
    pp.to_node = df_g2g_br['to_node'][i];
    pp.is_exist = df_g2g_br['is_existing'][i];
    pp.length = df_g2g_br['length (mile)'][i];
    pp.Cap = df_g2g_br['Capacity (MMBtu)'][i];
    pp.FOM = Other_input.pipline_FOM; # $/mile
    pp.decom = Other_input.pipeline_decom; # $/mile
    s1 = (1/(1+Other_input.WACC)**Other_input.pipeline_lifetime);
    pp.inv_coef = (Other_input.WACC/(1-s1));
    PipeLines.append(pp);

class exist_SVL:
    num = [];
    str_cap = [];
    vap_cap = [];
    liq_cap = [];

Exist_SVL = list();
for i in range(len(df_exis_svl)):
    eSvl = exist_SVL();
    eSvl.num = i;
    eSvl.str_cap = df_exis_svl['Storage-cap'][i];
    eSvl.vap_cap = df_exis_svl['Vap-cap'][i];
    eSvl.liq_cap = df_exis_svl['Liq-cap'][i];
    
    Exist_SVL.append(eSvl);


class SVL:
    capex = [];
    FOM = [];
    eff_ch = [];
    eff_disCh = [];
    BOG = [];
    inv_coef = [];
    
SVLs = list();
for i in range(len(df_svl_params)):
    svl = SVL();
    svl.capex = df_svl_params['capex'][i];
    svl.FOM = df_svl_params['FOM'][i];
    svl.eff_ch = df_svl_params['eff_ch'][i];
    svl.eff_disCh = df_svl_params['eff_disCh'][i];
    svl.BOG = df_svl_params['BOG'][i];
    s1 = (1/(1+Other_input.WACC)**Other_input.SVL_lifetime);
    svl.inv_coef = (Other_input.WACC/(1-s1));

    SVLs.append(svl);

#%% Quebec Hydro
# hydro import/export data
df_qc_dem = pd.read_csv(f'Hydro-QC/QC_demand.csv');
df_qc_inflow = pd.read_csv(f'Hydro-QC/QC-hydro-inflow-rates.csv');
df_qc_other_params = pd.read_csv(f'Hydro-QC/Other-hydro-params.csv');
class Quebec:
    demand = [];
    init_en_cap = []; 
    max_en_cap = [];
    max_pow_cap = [];
    NE_nodes = [];
    line_max_flow = [];
    ramp_rate = [];
    min_output_rate = [];
    inflow_rate = [];
    FOM_cost = [];
    NE_node2line = [];

QCdat = Quebec();
QCdat.demand = df_qc_dem['Demand (MW)-90%fromHydro'].to_numpy();
QCdat.inflow_rate = df_qc_inflow['Reservoir_hydro_existing_QC'].to_numpy();
QCdat.init_en_cap = df_qc_other_params['init_energy_cap (MWh)'][0];
QCdat.max_en_cap = df_qc_other_params['max_energy_cap (MWh)'][0];
QCdat.max_pow_cap = df_qc_other_params['max_power_cap (MW)'][0];
QCdat.min_output_rate =  df_qc_other_params['min_output_rate'][0];
QCdat.ramp_rate =  df_qc_other_params['ramp_rate'][0];
QCdat.NE_nodes = df_qc_other_params['nodes_connected'].to_numpy();
QCdat.line_max_flow = df_qc_other_params['max_flow_cap(MW)'].to_numpy();
QCdat.FOM_cost = df_qc_other_params['FOM ($/MW/yr)'][0];
QCdat.NE_node2line = {5:0,8:1};



#%% delete unnecessary data from the workspace
del df_br,df_ebus,df_eDem,df_ePlt,df_exis_svl,df_g2g_br,df_gnode;
del df_ng_adjE,df_ng_dem,;
del df_plt,df_reg_mult,df_solar,df_str,df_svl_params,df_wind,df_wind_offshore;
del s1,s2,svl,pp,Pow_dir,Gas_dir,plt,i,st,en;

del br,gas,enode;
# del sym2plant,plant2sym,zone_id2state,eSvl
