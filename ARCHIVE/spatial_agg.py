import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster


def import_data():
    lines_df = pd.read_csv('DATA/Dev/Transmission_Lines.csv')
    nodes_df = pd.read_csv('DATA/Dev/Power_Nodes.csv')
    wind_df = pd.read_csv(
        'DATA/Dev/Availability_Factors/AvailabilityFactors_Wind_Onshore_2020.csv')
    solar_df = pd.read_csv(
        'DATA/Dev/Availability_Factors/AvailabilityFactors_Solar_2020.csv')

    return lines_df, nodes_df, wind_df, solar_df


def plot_network(lines_df, nodes_df):
    world_map = folium.Map(
        location=(43, -70), zoom_start=7, tiles="cartodbpositron")

    for i, row in nodes_df.iterrows():
        lat, long = row['Lat'], -row['Lon']
        folium.CircleMarker(location=(lat, long), radius=4,
                            color='blue', popup=i).add_to(world_map)

    for _, row in lines_df.iterrows():
        fr_node = nodes_df.loc[nodes_df['node_num'] == row['from_node']]
        to_node = nodes_df.loc[nodes_df['node_num'] == row['to_node']]
        fr_node_coords = (fr_node['Lat'].values[0], -fr_node['Lon'].values[0])
        to_node_coords = (to_node['Lat'].values[0], -to_node['Lon'].values[0])
        folium.PolyLine(
            locations=[fr_node_coords, to_node_coords], color='blue').add_to(world_map)

    legend_html = '''
    <div style="position: fixed;
                bottom: 50px; right: 50px; width: 210px; height: 60px;
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color:white; opacity: 0.8;">
        <b>Legend</b><br>
        <span style="display:inline-block; width:12px; height:12px; border: 3px solid red; border-radius:50%; margin-right:5px;"></span> Power Node<br>
        <span style="display:inline-block; width:20px; height:3px; background-color:red; margin-right:5px;"></span> Power Line<br>
    </div>
    '''
    world_map.get_root().html.add_child(folium.Element(legend_html))

    return world_map


def create_capacity_factor_array(nodes_df, TimeSeriesList, TimeHorizon):
    N = nodes_df.shape[0]
    T = TimeHorizon
    G = len(TimeSeriesList)
    CF = np.zeros((N, G, T))

    for g in range(G):
        for n in range(N):
            for t in range(T):
                CF[n, g, t] = TimeSeriesList[g].iloc[t, n]

    print(f"Number of nodes: {N}")
    print(f"Number of time series features per node: {G}")
    print(f"Time horizon: {T}")

    return CF

# def corr_coeff(CF):
#     N, G, T = CF.shape

#     corr_coeff = np.zeros((N, N, G, G))
#     for g in range(G):
#         for h in range(G):
#             if h != g:
#                 for n in range(N):
#                     for m in range(N):
#                         if m != n:
#                             CF_n_g = CF[n, g, :]
#                             CF_m_h = CF[m, h, :]
#                             mean_n_g = np.mean(CF_n_g)
#                             mean_m_h = np.mean(CF_m_h)
#                             numerator = np.sum((CF_n_g - mean_n_g) * (CF_m_h - mean_m_h))
#                             denominator = np.sqrt(np.sum((CF_n_g - mean_n_g)**2) * np.sum((CF_m_h - mean_m_h)**2))
#                             corr_coeff[n, m, g, h] = numerator / denominator
#                             corr_coeff[n, m, g, h] = np.corrcoef(CF[n, g, :], CF[m, h, :])[0,1]

#     return corr_coeff


def corr_coeff(CF):
    N, G, T = CF.shape

    corr_coeff = np.zeros((N, N, G, G))
    for g in range(G):
        for h in range(G):
            if h != g:
                CF_g = CF[:, g, :]
                CF_h = CF[:, h, :]
                corr_matrix = np.corrcoef(CF_g, CF_h, rowvar=True)
                corr_coeff[:, :, g, h] = corr_matrix[:N, N:]

    return corr_coeff


def aggregate_nodes(CF, nodes_df, N_prime, lambdas=[1, 1, 1]):
    N, G, T = CF.shape

    # Initialize the model
    model = gp.Model("node_aggregation")
    model.Params.NonConvex = 2

    # Define variables
    alpha = model.addVars(N, N_prime, vtype=GRB.BINARY, name="alpha")
    C_agg = model.addVars(
        N_prime, G, T, vtype=GRB.CONTINUOUS, name="C_agg", lb=0)
    coords_agg = model.addVars(
        N_prime, 2, vtype=GRB.CONTINUOUS, name="coords_agg", lb=0)

    # Objective components
    NRMSE_av = sum(sum(alpha[n, n_prime] * (1 / G) * sum((1/T * sum((CF[n, g, t] - C_agg[n_prime, g, t])**2 for t in range(
        T)))**0.5 / (CF[n, g, :].max() - CF[n, g, :].min()) for g in range(G)) for n_prime in range(N_prime)) for n in range(N))

    corr_error = sum(sum(sum(sum(alpha[n, n_prime] * alpha[m, m_prime] / (G*(G-1)) * sum(sum(gp.abs_(corr_coeff(CF)[n, m, g, h] - corr_coeff(C_agg)[n_prime, m_prime, g, h])
                     for h in range(G) if h != g) for g in range(G)) for m_prime in range(N_prime)) for n_prime in range(N_prime)) for m in range(N) if m != n) for n in range(N))

    spatial_distance_error = sum(sum(alpha[n, n_prime] * ((nodes_df.iloc[n]['Lat'] - coords_agg[n_prime, 0])**2 + (
        nodes_df.iloc[n]['Lon'] - coords_agg[n_prime, 1])**2)**0.5 for n_prime in range(N_prime)) for n in range(N))

    # Set objective
    model.setObjective(lambdas[0] * NRMSE_av + lambdas[3] * corr_error +
                       lambdas[2] * spatial_distance_error, GRB.MINIMIZE)

    # Add constraints
    model.addConstrs((gp.quicksum(alpha[n, n_prime] for n_prime in range(
        N_prime)) == 1 for n in range(N)), "assignment")
    model.addConstrs((C_agg[n_prime, :, :] * gp.quicksum(alpha[n, n_prime] for n in range(N)) == gp.quicksum(alpha[n, n_prime] * CF[n, :, :] for n in range(
        N)) for n_prime in range(N_prime)), "capacity_factor")

    # Optimize the model
    model.optimize()

    return CF.x, alpha.x, coords_agg.x


# def aggregate_nodes(CF, nodes_df, N_prime, lambdas=[1, 1, 1]):
#     N, G, T = CF.shape

#     # Initialize the model
#     model = gp.Model("node_aggregation")

#     # Define variables
#     alpha = model.addVars(N, N_prime, vtype=GRB.BINARY, name="alpha")
#     C_agg = model.addVars(
#         N_prime, G, T, vtype=GRB.CONTINUOUS, name="C_agg", lb=0)
#     coords_agg = model.addVars(
#         N_prime, 2, vtype=GRB.CONTINUOUS, name="coords_agg", lb=0)

#     # Objective components
#     NRMSE_av = sum(sum(alpha[n, n_prime] * (1 / G) * sum(np.sqrt(1/T * sum((CF[n, g, t] - C_agg[n_prime, g, t])**2 for t in range(
#         T))) / (CF[n, g, :].max() - CF[n, g, :].min()) for g in range(G)) for n_prime in range(N_prime)) for n in range(N))

#     corr_error = sum(sum(sum(sum(alpha[n, n_prime] * alpha[m, m_prime] / (G*(G-1)) * sum(sum(np.abs(corr_coeff(CF)[n, m, g, h] - corr_coeff(C_agg)[n_prime, m_prime, g, h])
#                      for h in range(G) if h != g) for g in range(G)) for m_prime in range(N_prime)) for n_prime in range(N_prime)) for m in range(N) if m != n) for n in range(N))

#     spatial_distance_error = sum(sum(alpha[n, n_prime] * (nodes_df.iloc[n]['Lat'] - coords_agg[n_prime, 0])**2 + (
#         nodes_df.iloc[n]['Lon'] - coords_agg[n_prime, 1])**2 for n_prime in range(N_prime)) for n in range(N))

#     #    gp.quicksum(
#     #         (gp.quicksum((solar_df.iloc[t, n] - C_agg[n_prime, 0, t])**2 for t in range(T)) / T)**0.5 /
#     #         (solar_df.iloc[:, n].max() - solar_df.iloc[:, n].min())
#     #         for n_prime in range(N_prime)
#     #         ) for n in range(N))

#     #         correlation_error = gp.quicksum(gp.quicksum(
#     #     alpha[n, n_prime] * alpha[m, m_prime] * (1 / (len(G) * (len(G) - 1))) *
#     #     gp.quicksum(
#     #         abs(np.corrcoef(solar_df.iloc[:, n], wind_df.iloc[:, m])[
#     #             0, 1] - np.corrcoef(C_agg[n_prime, 0, :], C_agg[m_prime, 1, :])[0, 1])
#     #         for h in range(len(G)) if h != g
#     #     )
#     #     for g in range(len(G))
#     # )
#     #     for n in range(N) for m in range(N) if m != n for n_prime in range(N_prime) for m_prime in range(N_prime)
#     # )

#     #     spatial_distance_error = gp.quicksum(gp.quicksum(
#     #     alpha[n, n_prime] *
#     #     (nodes_df.iloc[n]['Lat'] - coords_agg[n_prime, 0])**2 +
#     #     (nodes_df.iloc[n]['Lon'] - coords_agg[n_prime, 1])**2
#     # ) for n in range(N) for n_prime in range(N_prime)
#     # )

#     # Set objective
#     model.setObjective(lambdas[0] * NRMSE_av + lambdas[3] * corr_error +
#                        lambdas[2] * spatial_distance_error, GRB.MINIMIZE)

#     # Add constraints
#     model.addConstrs((gp.quicksum(alpha[n, n_prime] for n_prime in range(
#         N_prime)) == 1 for n in range(N)), "assignment")
#     model.addConstrs((C_agg[n_prime, :, :] == gp.quicksum(alpha[n, n_prime] * CF[n, :, :] for n in range(
#         N)) / gp.quicksum(alpha[n, n_prime] for n in range(N)) for n_prime in range(N_prime)), "capacity_factor")

#     #     model.addConstrs((gp.quicksum(alpha[n, n_prime] for n_prime in range(
#     #     N_prime)) == 1 for n in range(N)), "assignment")
#     #    model.addConstrs((C_agg[n_prime, g, t] == gp.quicksum(alpha[n, n_prime] * solar_df.iloc[t, n] for n in range(N)) / gp.quicksum(
#     #         alpha[n, n_prime] for n in range(N)) for n_prime in range(N_prime) for g in range(len(G)) for t in range(T)), "capacity_factor")
#     # model.addConstrs((coords_agg[n_prime, 0] == gp.quicksum(alpha[n, n_prime] * nodes_df.iloc[n]['Lat'] for n in range(N)) / gp.quicksum(
#     #     alpha[n, n_prime] for n in range(N)) for n_prime in range(N_prime)), "lat_agg")
#     # model.addConstrs((coords_agg[n_prime, 1] == gp.quicksum(alpha[n, n_prime] * nodes_df.iloc[n]['Lon'] for n in range(N)) / gp.quicksum(
#     #     alpha[n, n_prime] for n in range(N)) for n_prime in range(N_prime)), "lon_agg")

#     # Optimize the model
#     model.optimize()

#     # Extract results

#     # alpha_result = np.zeros((N, N_prime))
#     # for n in range(N):
#     #     for n_prime in range(N_prime):
#     #         if alpha[n, n_prime].x > 0.5:
#     #             alpha_result[n, n_prime] = 1

#     # solar_df_agg = pd.DataFrame(np.zeros((T, N_prime)))
#     # wind_df_agg = pd.DataFrame(np.zeros((T, N_prime)))
#     # for n_prime in range(N_prime):
#     #     for t in range(T):
#     #         solar_df_agg.iloc[t, n_prime] = C_agg[n_prime, 0, t].x
#     #         wind_df_agg.iloc[t, n_prime] = C_agg[n_prime, 1, t].x

#     return CF.x, alpha.x, coords_agg.x
