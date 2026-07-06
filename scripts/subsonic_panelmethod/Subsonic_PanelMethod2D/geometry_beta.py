import numpy as np
from .geometry import PanelGeometry, compute_influence_coefficients, compute_potential_influence_coefficients

class PanelGeometryBeta(PanelGeometry):
    """
    扩展的几何体类，继承自 PanelGeometry，包含原类的全部属性
    
    Attributes:
        geom_original (PanelGeometry): 第一个附加的几何体属性，即本身的形状
        geom_beta (PanelGeometry): 第二个附加的几何体属性，即放缩后的形状
        对应P^beta={(x,beta·y)|(x,y)∈P}
        （以下为继承自 PanelGeometry 的属性）
        n_panels, vertices, generate_lift, n_equations,
        panel_midpoints, panel_lengths, panel_sines, panel_cosines
    """
    def __init__(self, vertices: np.ndarray, generate_lift: bool, beta: float):
        """
        初始化 PanelGeometryBeta
        
        Args:
            vertices (ndarray): 定义物体边界的坐标点数组，形状为(n+1, 2)
            generate_lift (bool): 是否产生升力（当前仅允许 False）
            beta (float): 准确来讲是\sqrt{1-M_\infty^2}
        """
        super().__init__(vertices, generate_lift)  # 调用父类初始化，继承所有原有属性
        self.beta = beta

        if generate_lift:
            raise NotImplementedError("Only generate_lift=False permitted!")
        
        assert np.linalg.norm(vertices[-1,:]-vertices[0,:]) < 1e-5

        self.geom_original = PanelGeometry(vertices=vertices, generate_lift=generate_lift)

        vertices_beta = vertices.copy()
        vertices_beta[:,1] = beta * vertices_beta[:,1]
        self.geom_beta = PanelGeometry(vertices=vertices_beta, generate_lift=generate_lift)

def compute_influence_coefficients_beta(point: np.ndarray, cosine_beta: float, sine_beta: float, geometry_beta: PanelGeometryBeta, beta):
    """
    计算给定点对几何体所有面板的影响系数。
    
    Args:
        point: 计算影响的目标点坐标，形状为(2,)
        cosine_beta: 目标点做beta放缩后方向的余弦值
        sine_beta: 目标点做beta放缩后方向的正弦值
        geometry_beta: Beta-面板几何体对象
        
    Returns:
        Tuple containing:
        - u_sources: beta-源项在x方向的速度影响系数
        - v_sources: beta-源项在y方向的速度影响系数

    注意：point 不能是面板顶点
         当 point 与版面共线时， 在求解v_source 时， 需要使用目标点方向(需要是经过了beta压缩的)， 当目标点方向与面板方向一致时，点在面板正侧，反之点在面板反侧，结果将不同
    """
    point_beta = point.copy()
    point_beta[1] = point_beta[1] * beta

    # l_ratio.shape = (num_panel,), 相当于 ds / ds^\beta
    l_ratio = geometry_beta.geom_original.panel_lengths / geometry_beta.geom_beta.panel_lengths

    us_beta, vs_beta, _, _ = compute_influence_coefficients(point=point_beta, cosine=cosine_beta, sine=sine_beta, geometry=geometry_beta.geom_beta)

    u_source = us_beta * l_ratio
    v_source = vs_beta * l_ratio * beta

    return u_source, v_source

def compute_edge_influence_coefficients_beta(point: np.ndarray, cosine: float, sine: float, cosine_beta: float, sine_beta: float, geometry_beta: PanelGeometryBeta, beta):
    """
    计算给定点在特定方向上的影响系数。
    
    Args:
        point: 计算影响的目标点坐标
        cosine: 目标方向的余弦值
        sine: 目标方向的正弦值
        cosine_beta: 压缩之后的目标方向余弦值
        sine_beta: 压缩之后的目标方向正弦值
        geometry_beta: beta-面板几何体对象
        
    Returns:
        Tuple containing:
        - t_sources: 源项在切向的速度影响系数
        - n_sources: 源项在法向的速度影响系数
    """
    u_sources, v_sources = compute_influence_coefficients_beta(point, cosine_beta, sine_beta, geometry_beta, beta)
    # T代表切向,N代表法向,s代表源,v代表涡
    t_sources, n_sources = cosine*u_sources + sine*v_sources, -sine*u_sources + cosine*v_sources

    return t_sources, n_sources

def compute_all_edge_influence_coefficients_beta(velocity_geometry: PanelGeometryBeta, source_geometry: PanelGeometryBeta, beta:float):
    """
    计算所有源面板对所有速度控制点的影响系数矩阵。
    
    Args:
        velocity_geometry: 速度控制点所在的面板几何体
        source_geometry: 源项所在的面板几何体
        
    Returns:
        Tuple containing influence coefficient matrices:
        - t_sources: 切向源影响系数矩阵，形状为(n_edges, n_panels)
        - n_sources: 法向源影响系数矩阵，形状为(n_edges, n_panels)
    """
    n_edges = velocity_geometry.n_panels
    n_panels = source_geometry.n_panels
    
    

    t_sources, n_sources = np.zeros((n_edges, n_panels)), np.zeros((n_edges, n_panels))

    for i in range(n_edges):
        point = velocity_geometry.panel_midpoints[i]
        cosine = velocity_geometry.panel_cosines[i]
        sine = velocity_geometry.panel_sines[i]
        cosine_beta = velocity_geometry.geom_beta.panel_cosines[i]
        sine_beta = velocity_geometry.geom_beta.panel_sines[i]
        t_sources[i,:], n_sources[i,:] = compute_edge_influence_coefficients_beta(point, cosine, sine, cosine_beta, sine_beta, source_geometry, beta)
    
    return t_sources, n_sources

def compute_potential_influence_coefficients_beta(point: np.ndarray, cosine_beta: float, sine_beta: float, geometry_beta: PanelGeometryBeta, beta:float):
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
        
    point_beta = point.copy()
    point_beta[1] = beta * point_beta[1]

    l_ratio = geometry_beta.geom_original.panel_lengths / geometry_beta.geom_original.panel_lengths

    phi_s = compute_potential_influence_coefficients(point_beta, cosine_beta, sine_beta, geometry_beta.geom_beta)
    phi_sources = phi_s * l_ratio
    
    return phi_sources 


def compute_all_edge_potential_influence_coefficients_beta(velocity_geometry: PanelGeometryBeta, source_geometry: PanelGeometryBeta, beta:float):
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
        cosine_beta = velocity_geometry.geom_beta.panel_cosines[i]
        sine_beta = velocity_geometry.geom_beta.panel_sines[i]
        phi_sources[i,:] = compute_potential_influence_coefficients_beta(point, cosine_beta, sine_beta, source_geometry, beta)
    
    return phi_sources

