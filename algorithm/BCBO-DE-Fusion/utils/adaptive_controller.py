#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自适应参数控制器

提供DE算法的自适应参数控制,包括F(缩放因子)和CR(交叉概率)的动态调整。
"""


class AdaptiveFController:
    """
    自适应F控制器(HDE策略)

    使用线性衰减策略动态调整差分进化的缩放因子F。
    """

    def __init__(self, F_max=0.9, F_min=0.4, alpha=0.1):
        """
        初始化F控制器

        参数:
            F_max: 最大缩放因子(初始值)
            F_min: 最小缩放因子(终止值)
            alpha: 自适应系数(保留参数,当前版本未使用)
        """
        self.F_max = F_max
        self.F_min = F_min
        self.alpha = alpha

    def get_F(self, iteration: int, total_iterations: int) -> float:
        """
        HDE线性衰减公式

        参数:
            iteration: 当前迭代次数(从0开始)
            total_iterations: 总迭代次数

        返回:
            F: 当前迭代的缩放因子

        公式:
            F(t) = (F_max - F_min) * (1 - t/T) + F_min

        原理:
            - 初期(t=0): F = F_max (较大的F,增强全局探索)
            - 后期(t=T): F = F_min (较小的F,增强局部开发)
            - 中期: F线性衰减

        示例:
            >>> controller = AdaptiveFController(F_max=0.9, F_min=0.4)
            >>> F_0 = controller.get_F(0, 100)      # 初始F
            >>> F_50 = controller.get_F(50, 100)    # 中期F
            >>> F_99 = controller.get_F(99, 100)    # 终止F
        """
        if total_iterations == 0:
            return self.F_max

        t = iteration / total_iterations
        F = (self.F_max - self.F_min) * (1 - t) + self.F_min
        return F

    def get_adaptive_F_by_fitness(self, current_fitness: float, best_fitness: float) -> float:
        """
        基于适应度的自适应F(可选方法)

        参数:
            current_fitness: 当前个体适应度
            best_fitness: 当前最优适应度

        返回:
            F: 自适应的缩放因子

        策略:
            - 适应度差: F较大(加强探索)
            - 适应度好: F较小(加强开发)
        """
        if best_fitness == 0:
            return self.F_max

        fitness_ratio = current_fitness / best_fitness
        F = self.F_min + (self.F_max - self.F_min) * (1 - fitness_ratio)
        return max(self.F_min, min(self.F_max, F))


class AdaptiveCRController:
    """
    自适应CR控制器

    根据种群多样性动态调整交叉概率。
    """

    def __init__(self, CR_min=0.5, CR_max=0.9):
        """
        初始化CR控制器

        参数:
            CR_min: 最小交叉概率
            CR_max: 最大交叉概率
        """
        self.CR_min = CR_min
        self.CR_max = CR_max

    def get_CR(self, diversity: float) -> float:
        """
        基于多样性的自适应CR

        参数:
            diversity: 当前种群多样性(0.0-1.0)

        返回:
            CR: 当前的交叉概率

        策略:
            - 低多样性(diversity → 0): CR → CR_max (高交叉率,加强探索)
            - 高多样性(diversity → 1): CR → CR_min (低交叉率,保留多样性)

        公式:
            CR = CR_max - (CR_max - CR_min) * diversity

        原理:
            当种群多样性降低时,需要更强的交叉来引入新的基因组合;
            当种群多样性较高时,降低交叉率以保护现有的多样性。

        示例:
            >>> controller = AdaptiveCRController(CR_min=0.5, CR_max=0.9)
            >>> CR_low = controller.get_CR(0.2)    # 低多样性
            >>> CR_high = controller.get_CR(0.8)   # 高多样性
        """
        # 线性映射: diversity越低,CR越高
        CR = self.CR_max - (self.CR_max - self.CR_min) * diversity
        return CR

    def get_CR_by_iteration(self, iteration: int, total_iterations: int) -> float:
        """
        基于迭代次数的自适应CR(可选方法)

        参数:
            iteration: 当前迭代次数
            total_iterations: 总迭代次数

        返回:
            CR: 自适应的交叉概率

        策略:
            - 初期: 高CR(加强探索)
            - 后期: 低CR(加强开发)
        """
        if total_iterations == 0:
            return self.CR_max

        t = iteration / total_iterations
        CR = self.CR_max - (self.CR_max - self.CR_min) * t
        return CR


# 测试代码
if __name__ == '__main__':
    print("自适应参数控制器测试")
    print("=" * 60)

    # 测试F控制器
    print("\n【测试1: AdaptiveFController】")
    f_controller = AdaptiveFController(F_max=0.9, F_min=0.4)

    print("线性衰减策略:")
    total_iters = 100
    test_iters = [0, 25, 50, 75, 99]
    for it in test_iters:
        F = f_controller.get_F(it, total_iters)
        print(f"  迭代 {it:3d}/{total_iters}: F = {F:.4f}")

    # 测试CR控制器
    print("\n【测试2: AdaptiveCRController】")
    cr_controller = AdaptiveCRController(CR_min=0.5, CR_max=0.9)

    print("基于多样性的自适应CR:")
    test_diversities = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for div in test_diversities:
        CR = cr_controller.get_CR(div)
        print(f"  多样性 {div:.1f}: CR = {CR:.4f}")

    # 测试CR的迭代策略
    print("\n基于迭代次数的自适应CR:")
    for it in test_iters:
        CR = cr_controller.get_CR_by_iteration(it, total_iters)
        print(f"  迭代 {it:3d}/{total_iters}: CR = {CR:.4f}")

    # 可视化测试(需要matplotlib)
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        print("\n【绘制参数变化曲线】")
        iterations = np.arange(0, 100)

        # F的变化
        F_values = [f_controller.get_F(it, 100) for it in iterations]

        # CR的变化(基于假设的多样性变化)
        # 假设多样性从0.8逐渐降低到0.2
        diversities = 0.8 - 0.6 * (iterations / 100)
        CR_values = [cr_controller.get_CR(div) for div in diversities]

        # 绘图
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

        # F变化
        ax1.plot(iterations, F_values, 'b-', linewidth=2)
        ax1.set_ylabel('F (缩放因子)')
        ax1.set_title('HDE自适应参数变化')
        ax1.grid(True)

        # CR变化
        ax2.plot(iterations, CR_values, 'r-', linewidth=2)
        ax2.set_ylabel('CR (交叉概率)')
        ax2.grid(True)

        # 多样性变化
        ax3.plot(iterations, diversities, 'g-', linewidth=2)
        ax3.set_xlabel('迭代次数')
        ax3.set_ylabel('种群多样性')
        ax3.grid(True)

        plt.tight_layout()
        plt.savefig('adaptive_parameters.png', dpi=150)
        print("  图表已保存为 'adaptive_parameters.png'")

    except ImportError:
        print("\n(跳过可视化测试,需要安装matplotlib)")

    print("\n" + "=" * 60)
    print("✅ 测试完成!")
