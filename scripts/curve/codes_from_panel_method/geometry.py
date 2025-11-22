import numpy as np
from typing import List, Tuple


class PanelGeometry:
    """
    表示用面板离散化的几何体，用于面元法计算。
    
    面板是由一系列坐标点定义的，每个面板连接两个相邻点。
    对于需要产生升力的物体（如机翼），面板应按照特定方向排列以正确应用Kutta条件。
    
    Attributes:
        n_panels (int): 面板数量
        vertices (ndarray): 几何体顶点坐标数组，形状为(n_panels+1, 2)
        generate_lift (bool): 是否产生升力
        n_equations (int): 方程或未知量的个数
        panel_midpoints (ndarray): 每个面板的中点坐标，形状为(n, 2)
        panel_lengths (ndarray): 每个面板的长度，形状为(n,)
        panel_sines (ndarray): 每个面板与x轴夹角的正弦值，形状为(n,)
        panel_cosines (ndarray): 每个面板与x轴夹角的余弦值，形状为(n,)
    """
    def __init__(self, vertices: np.ndarray, generate_lift: bool):
        """
        初始化PanelGeometry
        
        Args:
            vertices (ndarray): 定义物体边界的坐标点数组，形状为(n+1, 2)
                - 对于需要产生升力的物体，应从尾部开始顺时针排列
                - 最后一个点应与第一个点可以相同（形成封闭曲线），也可以不同
        """
        self.generate_lift = generate_lift

        self.n_panels = vertices.shape[0] - 1
        
        self.n_equations = self.n_panels + (1 if generate_lift else 0)

        self.vertices = vertices

        self._compute_panel_properties()

    def _compute_panel_properties(self) -> None:

        vertices = self.vertices

        # 计算面板中点坐标
        self.panel_midpoints = np.column_stack((
            (vertices[:-1, 0] + vertices[1:, 0]) / 2,
            (vertices[:-1, 1] + vertices[1:, 1]) / 2
        ))
        
        # 计算面板长度
        dx = vertices[1:, 0] - vertices[:-1, 0]
        dy = vertices[1:, 1] - vertices[:-1, 1]
        self.panel_lengths = np.sqrt(dx**2 + dy**2)
        
        # 面板方向（切向量分量）
        self.panel_cosines = dx / self.panel_lengths
        self.panel_sines = dy / self.panel_lengths

        
        





def compute_influence_coefficients(point: np.ndarray, cosine: float, sine: float, geometry: PanelGeometry):
    """
    计算给定点对几何体所有面板的影响系数。
    
    Args:
        point: 计算影响的目标点坐标，形状为(2,)
        cosine: 目标点方向的余弦值
        sine: 目标点方向的正弦值
        geometry: 面板几何体对象
        
    Returns:
        Tuple containing:
        - u_sources: 源项在x方向的速度影响系数
        - v_sources: 源项在y方向的速度影响系数  
        - u_vortices: 涡项在x方向的速度影响系数
        - v_vortices: 涡项在y方向的速度影响系数

    注意：point 不能是面板顶点
         当 point 与版面共线时， 在求解v_source 时， 需要使用目标点方向， 当目标点方向与面板方向一致时，点在面板正侧，反之点在面板反侧，结果将不同
    """
        
    x0, y0 = point
    n_panels = geometry.n_panels
    n_points = n_panels + 1
    vertices = geometry.vertices
    panel_lengths = geometry.panel_lengths
    panel_cosines = geometry.panel_cosines
    panel_sines = geometry.panel_sines

    # 计算点到所有顶点的距离
    r_lengths = np.sqrt((x0 - vertices[:,0])**2+(y0 - vertices[:,1])**2)
    
    if np.min(r_lengths) <= 1.0e-10:
        raise ValueError(f"错误: point 和面板顶点重合 (距离={np.min(r_lengths)})， 网格有交叉")
    
    # 转换到面板局部坐标系
    x0_stars = panel_cosines*(x0 - vertices[:-1,0]) + panel_sines*(y0 - vertices[:-1,1])
    y0_stars = -panel_sines*(x0 - vertices[:-1,0]) + panel_cosines*(y0 - vertices[:-1,1])
    
    # 计算几何参数
    c1 = (x0_stars - panel_lengths)/r_lengths[1:]
    s1 = y0_stars/r_lengths[1:]
    c2 = x0_stars/r_lengths[0:-1]
    s2 = y0_stars/r_lengths[0:-1]
    sine_betas = s1*c2 - s2*c1
    cosine_betas = c1*c2 + s1*s2
 
   
    u_source_stars = -np.log(r_lengths[1:]/r_lengths[0:-1])/(2*np.pi)
    v_source_stars = np.arctan2(sine_betas, cosine_betas)/(2*np.pi)


    # 当 y0_stars = 0 时，点与面板共线我，们有
    # 1）点在面板里面，sine_beta = 0， cosine_beta = -1, beta = +pi 或 -pi 
    # 2）点在面板外面，sine_beta = 0， cosine_beta = 1， beta = 0
    # 计算点的切线方向与面板的切线方向的内积
    tangential_inner_product  = cosine * panel_cosines + sine * panel_sines
    # 如果内积 > 0， 顶点自己所在面板， 那么 beta = +pi
    # 识别点在面板内情况（v_source_stars 接近 ±0.5）
    collinear_mask = np.isclose(np.abs(v_source_stars), 0.5, atol=1e-10, rtol=1e-10)
    collinear_indices = np.where(collinear_mask)[0]
    # 验证点在面板内的面板的数量（最多1个：面板自己）
    if len(collinear_indices) > 1:
        raise ValueError(f"发现 {len(collinear_indices)} 个包含点的面板，但最多应为1个")
    if len(collinear_indices) == 1:
        # 根据切向内积符号调整v_source_stars值
        if not np.isclose(np.abs(tangential_inner_product[collinear_indices[0]]), 1.0, atol=1e-10, rtol=1e-10):
            raise ValueError(f"共线点的切向内积 {tangential_inner_product[collinear_indices[0]]} 绝对值不接近1.0")    
        v_source_stars[collinear_indices[0]] = 0.5
        
    
    
    u_vortex_stars =  v_source_stars
    v_vortex_stars = -u_source_stars 


    u_sources = u_source_stars * panel_cosines - v_source_stars * panel_sines
    v_sources = u_source_stars * panel_sines + v_source_stars * panel_cosines
    u_vortices = u_vortex_stars * panel_cosines - v_vortex_stars * panel_sines
    v_vortices = u_vortex_stars * panel_sines + v_vortex_stars * panel_cosines
        
    return u_sources, v_sources, u_vortices, v_vortices



def compute_edge_influence_coefficients(point: np.ndarray, cosine: float, sine: float, geometry: PanelGeometry):
    """
    计算给定点在特定方向上的影响系数。
    
    Args:
        point: 计算影响的目标点坐标
        cosine: 目标方向的余弦值
        sine: 目标方向的正弦值
        geometry: 面板几何体对象
        
    Returns:
        Tuple containing:
        - t_sources: 源项在切向的速度影响系数
        - n_sources: 源项在法向的速度影响系数
        - t_vortices: 涡项在切向的速度影响系数  
        - n_vortices: 涡项在法向的速度影响系数
    """
    u_sources, v_sources, u_vortices, v_vortices = compute_influence_coefficients(point, cosine, sine, geometry)
    # T代表切向,N代表法向,s代表源,v代表涡
    t_sources, n_sources = cosine*u_sources + sine*v_sources, -sine*u_sources + cosine*v_sources
    t_vortices, n_vortices = cosine*u_vortices + sine*v_vortices, -sine*u_vortices + cosine*v_vortices

    return t_sources, n_sources, t_vortices, n_vortices



def compute_all_edge_influence_coefficients(velocity_geometry, source_geometry):
    """
    计算所有源面板对所有速度控制点的影响系数矩阵。
    
    Args:
        velocity_geometry: 速度控制点所在的面板几何体
        source_geometry: 源项所在的面板几何体
        
    Returns:
        Tuple containing influence coefficient matrices:
        - t_sources: 切向源影响系数矩阵，形状为(n_edges, n_panels)
        - n_sources: 法向源影响系数矩阵，形状为(n_edges, n_panels)
        - t_vortices: 切向涡影响系数矩阵，形状为(n_edges, n_panels)
        - n_vortices: 法向涡影响系数矩阵，形状为(n_edges, n_panels)
    """
    n_edges = velocity_geometry.n_panels
    n_panels = source_geometry.n_panels
    
    

    t_sources, n_sources, t_vortices, n_vortices = np.zeros((n_edges, n_panels)), np.zeros((n_edges, n_panels)), np.zeros((n_edges, n_panels)), np.zeros((n_edges, n_panels))

    for i in range(n_edges):
        point = velocity_geometry.panel_midpoints[i]
        cosine = velocity_geometry.panel_cosines[i]
        sine = velocity_geometry.panel_sines[i]
        t_sources[i,:], n_sources[i,:], t_vortices[i,:], n_vortices[i,:] = compute_edge_influence_coefficients(point, cosine, sine, source_geometry)
    
    return t_sources, n_sources, t_vortices, n_vortices








def compute_potential_influence_coefficients(point: np.ndarray, cosine: float, sine: float, geometry: PanelGeometry):
    """
    计算给定点对几何体所有面板的势函数的影响系数。
    这个函数只是用于后处理和测试，在 panel method 里并不会使用
    
    Args:
        point: 计算影响的目标点坐标，形状为(2,)
        cosine: 目标点方向的余弦值
        sine: 目标点方向的正弦值
        geometry: 面板几何体对象
        
    Returns:
        - phi_sources: 源项在x方向的速度影响系数 k(x,y) = ln(|x-y|)/2pi

    """
        
    x0, y0 = point
    vertices = geometry.vertices
    panel_lengths = geometry.panel_lengths
    panel_cosines = geometry.panel_cosines
    panel_sines = geometry.panel_sines

    # 计算点到所有顶点的距离
    r_lengths = np.sqrt((x0 - vertices[:,0])**2+(y0 - vertices[:,1])**2)
    
    if np.min(r_lengths) <= 1.0e-10:
        raise ValueError(f"错误: point 和面板顶点重合 (距离={np.min(r_lengths)})， 网格有交叉")
    
    # 转换到面板局部坐标系
    x0_stars = panel_cosines*(x0 - vertices[:-1,0]) + panel_sines*(y0 - vertices[:-1,1])
    y0_stars = -panel_sines*(x0 - vertices[:-1,0]) + panel_cosines*(y0 - vertices[:-1,1])
    
    # 当 y0_stars = 0 时，点与面板共线我，们有
    collinear_mask = np.isclose(np.abs(y0_stars), 0.0, atol=1e-10, rtol=1e-10)
   
    phi_sources = ( (panel_lengths  - x0_stars)*np.log(r_lengths[1:])  + x0_stars*np.log(r_lengths[0:-1]) - panel_lengths) /(2*np.pi)
    phi_sources[~collinear_mask] += ((y0_stars * (np.arctan((panel_lengths  - x0_stars) / y0_stars) + np.arctan(x0_stars / y0_stars)) ) /(2*np.pi))[~collinear_mask]
    return phi_sources 


def compute_all_edge_potential_influence_coefficients(velocity_geometry, source_geometry):
    """
    计算所有源面板对所有速度控制点的影响系数矩阵。
    这个函数只是用于后处理和测试，在 panel method 里并不会使用
    
    Args:
        velocity_geometry: 速度控制点所在的面板几何体
        source_geometry: 源项所在的面板几何体
        
    Returns:
        - phi_sources: 切向源影响系数矩阵，形状为(n_edges, n_panels)
    """
    n_edges = velocity_geometry.n_panels
    n_panels = source_geometry.n_panels

    phi_sources = np.zeros((n_edges, n_panels))

    for i in range(n_edges):
        point = velocity_geometry.panel_midpoints[i]
        cosine = velocity_geometry.panel_cosines[i]
        sine = velocity_geometry.panel_sines[i]
        phi_sources[i,:] = compute_potential_influence_coefficients(point, cosine, sine, source_geometry)
    
    return phi_sources





