import numpy as np
from typing import List, Tuple
from .geometry_beta import PanelGeometryBeta
from .geometry_beta import compute_all_edge_influence_coefficients_beta
from .geometry_beta import compute_all_edge_potential_influence_coefficients_beta


def assemble_system_matrix(velocity_geometry: PanelGeometryBeta, source_geometry: PanelGeometryBeta, beta):
    """
    组装单个源几何体对单个速度几何体的系统矩阵贡献。
    
    Args:
        velocity_geometry: 速度控制点所在的面板几何体
        source_geometry: 源项所在的面板几何体
        
    Returns:
        Tuple containing:
        - A_block: 系数矩阵块，形状为(n_equations_vel, n_equations_src)
        - t_sources: 切向源影响系数矩阵
        - n_sources: 法向源影响系数矩阵
        - t_vortices: 切向涡影响系数矩阵
        - n_vortices: 法向涡影响系数矩阵
    """
    assert abs(velocity_geometry.beta - source_geometry.beta) < 1e-5

    n_edges = velocity_geometry.n_panels 
    n_panels_src = source_geometry.n_panels
    n_equations_vel = velocity_geometry.n_equations
    n_equations_src = source_geometry.n_equations
    t_sources, n_sources = compute_all_edge_influence_coefficients_beta(
        velocity_geometry, source_geometry,beta
    )
    
    

    # 初始化系数矩阵块
    A_block = np.zeros((n_equations_vel, n_equations_src))
    
    # source panels 贡献（法向速度边界条件）
    A_block[:n_edges, :n_panels_src] = n_sources
    # vortex panles 贡献（如果源几何体产生升力）
    if source_geometry.generate_lift:
        # vortex panles 贡献
        raise NotImplementedError("Only generate_lift=False permitted!")
    # Kutta条件（如果速度几何体产生升力）
    if velocity_geometry.generate_lift:
        # 第一个和最后一个panel 的切向速度和为0，（切向是顺时针方向），即出速度相同
        raise NotImplementedError("Only generate_lift=False permitted!")

    return A_block, t_sources, n_sources

def assemble_rhs_vector(velocity_geometry: PanelGeometryBeta, angle_of_attack: float, freestream_velocity: float, beta:float):
    """
    组装右侧向量（边界条件）。
    
    Args:
        velocity_geometry: 速度控制点所在的面板几何体
        angle_of_attack: 攻角（弧度）, 暂时强制为0
        freestream_velocity: 自由来流速度大小
        
    Returns:
        b_vector: 右侧向量，形状为(n_equations,)
    """
    assert abs(angle_of_attack) < 1e-5

    n_edges = velocity_geometry.n_panels
    n_equations = velocity_geometry.n_equations
    panel_cosines = velocity_geometry.panel_cosines
    panel_sines = velocity_geometry.panel_sines


    b_vector = np.zeros(n_equations)
    
    # 法向速度边界条件（物面不可穿透条件）
    b_vector[0:n_edges] = -(panel_cosines*np.sin(angle_of_attack) - panel_sines*np.cos(angle_of_attack)) * freestream_velocity

    # Kutta条件（如果几何体产生升力）
    if velocity_geometry.generate_lift:
        raise NotImplementedError("Only generate_lift=False permitted!")
    return b_vector



def solve_panel_method(geometries: List[PanelGeometryBeta], angle_of_attack: float, freestream_velocity: float, Mach: float):
    """
    求解多几何体面元法系统。
    
    Args:
        geometries: 面板几何体列表
        angle_of_attack: 攻角（弧度）
        freestream_velocity: 自由来流速度大小
        
    Returns:
        Dictionary containing:
        - solution: 完整的系统解向量
        - source_strengths: 各几何体的源强度分布
        - vortex_strengths: 各几何体的涡强度
        - tangential_velocities: 各几何体的切向速度分布
        - normal_velocities: 各几何体的法向速度分布
        - pressure_coefficients: 各几何体的压力系数分布
    """
    assert abs(angle_of_attack) < 1e-5
    assert 0 <= Mach < 1
    beta = np.sqrt(1-Mach**2)

    n_geometries = len(geometries)
    n_equations = [geometries[i].n_equations for i in range(n_geometries)]
    equation_offsets = np.concatenate(([0], np.cumsum(n_equations)))
    total_equations = equation_offsets[-1]

    # 初始化全局系统矩阵和右侧向量
    global_matrix = np.zeros((total_equations, total_equations))
    global_rhs = np.zeros(total_equations)
    
    t_sources = [[None] * n_geometries for _ in range(n_geometries)]
    n_sources = [[None] * n_geometries for _ in range(n_geometries)]

    for i, vel_geo in enumerate(geometries):
        row_start = equation_offsets[i]
        row_end = equation_offsets[i + 1]
        
        global_rhs[row_start:row_end] = assemble_rhs_vector(
            vel_geo, angle_of_attack, freestream_velocity, beta
        )

        for j, src_geo in enumerate(geometries):
            col_start = equation_offsets[j]
            col_end = equation_offsets[j + 1]
            
            global_matrix[row_start:row_end, col_start:col_end], t_sources[i][j], n_sources[i][j] = \
                    assemble_system_matrix(vel_geo, src_geo, beta)
            
            

    # 求解线性系统
    solution = np.linalg.solve(global_matrix, global_rhs)
    
    # 提取源强度和涡强度
    strengths = [np.zeros(geometry.n_panels+1) for geometry in geometries]
    for i, geo in enumerate(geometries):
        col_start = equation_offsets[i]
        n_panels = geo.n_panels
        strengths[i][0:n_panels] = solution[col_start:col_start + geo.n_panels]
        strengths[i][n_panels] = solution[col_start + geo.n_panels] if geo.generate_lift else 0.0
    

    t_velocity = [np.zeros(geometry.n_panels) for geometry in geometries]
    n_velocity  = [np.zeros(geometry.n_panels) for geometry in geometries]
    pressure_coefficients = [np.zeros(geometry.n_panels) for geometry in geometries]
    # compute velocities
    for i, vel_geo in enumerate(geometries):
        
        n_panels = vel_geo.n_panels
        panel_sines,panel_cosines = vel_geo.panel_sines,vel_geo.panel_cosines

        t_velocity[i] = (np.cos(angle_of_attack)*panel_cosines + np.sin(angle_of_attack)*panel_sines) * freestream_velocity
        n_velocity[i] = (-np.cos(angle_of_attack)*panel_sines  + np.sin(angle_of_attack)*panel_cosines) * freestream_velocity
        
        # 源项和涡项贡献
        for j in range(n_geometries):
            
            t_velocity[i] += t_sources[i][j]@strengths[j][0:-1]
            n_velocity[i] += n_sources[i][j]@strengths[j][0:-1]

        pressure_coefficients[i] = 1.0 - (t_velocity[i]/freestream_velocity)**2
    return solution, t_velocity, n_velocity, pressure_coefficients
   


def compute_source_potential(geometries_beta: List[PanelGeometryBeta], strengths: List[np.ndarray], beta:float):
    """
    计算源 sigma(x)在曲线 geometries 上的积分。
    这个函数只是用于后处理和测试，在 panel method 里并不会使用
    
    Args:
        geometries: 面板几何体列表
        strengths: 面板中心源强度
        
    Returns:
        - phi(x): 势函数      phi(x) = \int sigma(x) k(x,y) dy, k(x,y)=ln(|x-y|)/2pi
    """
    
    n_geometries = len(geometries_beta)
    phi_sources = [[None] * n_geometries for _ in range(n_geometries)]
    
    for i, vel_geo in enumerate(geometries_beta):
        for j, src_geo in enumerate(geometries_beta):
            phi_sources[i][j] = compute_all_edge_potential_influence_coefficients_beta(vel_geo, src_geo, beta)

    # 根据源强度求积分
    potential = [np.zeros(geometry.n_panels) for geometry in geometries_beta]
    # compute velocities
    for i, vel_geo in enumerate(geometries_beta):  

        # 源项和涡项贡献
        for j in range(n_geometries):
            potential[i]  += phi_sources[i][j]@strengths[j] 
            
    return potential 
   



