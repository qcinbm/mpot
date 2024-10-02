import os
from pathlib import Path
import time
import matplotlib.pyplot as plt
import torch
from einops._torch_specific import allow_ops_in_compiled_graph  # yêu cầu einops>=0.6.1

from mpot.ot.problem import Epsilon
from mpot.ot.sinkhorn import Sinkhorn
from mpot.planner import MPOT
from mpot.costs import CostGPHolonomic, CostField, CostComposite
from mpot.envs.occupancy import EnvOccupancy2D
from mpot.utils.trajectory import interpolate_trajectory

from torch_robotics.robots.robot_point_mass import RobotPointMass
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import get_torch_device
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer

# Cho phép các thao tác trong đồ thị biên dịch của einops
allow_ops_in_compiled_graph()

if __name__ == "__main__":
    # Thiết lập seed ngẫu nhiên dựa trên thời gian hiện tại để đảm bảo tính ngẫu nhiên và khả năng tái tạo
    seed = int(time.time())
    fix_random_seed(seed)

    # Xác định thiết bị tính toán (CPU hoặc GPU) và các tham số mặc định cho tensor
    device = get_torch_device()
    tensor_args = {'device': device, 'dtype': torch.float32}

    # ---------------------------- Môi trường, Robot, Nhiệm vụ Lập kế hoạch ---------------------------------
    # Giới hạn chuyển động của robot trong không gian làm việc 2D
    q_limits = torch.tensor([[-10, -10], [10, 10]], **tensor_args)
    
    # Khởi tạo môi trường 2D với bản đồ chiếm đóng
    env = EnvOccupancy2D(
        precompute_sdf_obj_fixed=False,
        tensor_args=tensor_args
    )

    # Khởi tạo robot điểm khối lượng với giới hạn chuyển động đã định nghĩa
    robot = RobotPointMass(
        q_limits=q_limits,  # Giới hạn các khớp
        tensor_args=tensor_args
    )

    # Thiết lập nhiệm vụ lập kế hoạch với môi trường và robot đã khởi tạo
    task = PlanningTask(
        env=env,
        robot=robot,
        ws_limits=q_limits,  # Giới hạn không gian làm việc
        obstacle_cutoff_margin=0.05,  # Lề cắt cho các chướng ngại vật
        tensor_args=tensor_args
    )

    # -------------------------------- Tham số ---------------------------------
    # Lưu ý: các tham số này được điều chỉnh phù hợp với môi trường này
    step_radius = 0.15  # Bán kính bước
    probe_radius = 0.15  # Bán kính thăm dò (phải >= step_radius)

    # Lưu ý: thay đổi polytope có thể yêu cầu điều chỉnh lại các tham số
    polytope = 'cube'  # Các lựa chọn: 'simplex', 'orthoplex', 'cube'

    epsilon = 0.01  # Tham số epsilon cho tối ưu hóa
    ent_epsilon = Epsilon(1e-2)  # Tham số entropy epsilon
    num_probe = 5  # Số lượng điểm thăm dò cho mỗi đỉnh của polytope
    num_particles_per_goal = 33  # Số lượng kế hoạch cho mỗi mục tiêu
    pos_limits = [-10, 10]  # Giới hạn vị trí
    vel_limits = [-10, 10]  # Giới hạn vận tốc
    w_coll = 5e-3  # Trọng số cho chi phí va chạm
    w_smooth = 1e-7  # Trọng số cho chi phí GP: error = w_smooth * || Phi x(t) - x(t+1) ||^2
    sigma_gp = 0.1  # Tham số sigma cho chi phí GP: Q_c = sigma_gp^2 * I
    sigma_gp_init = 1.6  # Tham số sigma ban đầu cho biến thiên GP: Q0_c = sigma_gp_init^2 * I
    max_inner_iters = 100  # Số lần lặp tối đa cho thuật toán Sinkhorn-Knopp
    max_outer_iters = 100  # Số lần lặp tối đa cho MPOT

    # Trạng thái khởi đầu của robot: [vị trí X, vị trí Y, vận tốc X, vận tốc Y]
    start_state = torch.tensor([-9, -9, 0., 0.], **tensor_args)

    # Lưu ý: thay đổi các trạng thái mục tiêu ở đây (tất cả mục tiêu có vận tốc bằng 0)
    multi_goal_states = torch.tensor([
        [0, 9, 0., 0.],
        [9, 9, 0., 0.],
        [9, 0, 0., 0.]
    ], **tensor_args)

    traj_len = 64  # Độ dài của chuyển động (số bước thời gian)
    dt = 0.1  # Bước thời gian giữa các bước chuyển động

    #--------------------------------- Hàm Chi Phí ---------------------------------
    
    # Chi phí do va chạm với chướng ngại vật
    cost_coll = CostField(
        robot, traj_len,
        field=env.occupancy_map,
        sigma_coll=1.0,
        tensor_args=tensor_args
    )
    
    # Chi phí mượt mà dựa trên Gaussian Process
    cost_gp = CostGPHolonomic(
        robot, traj_len, dt, sigma_gp, [0, 1],
        weight=w_smooth, tensor_args=tensor_args
    )
    
    # Danh sách các hàm chi phí
    cost_func_list = [cost_coll, cost_gp]
    # Trọng số tương ứng cho các hàm chi phí
    weights_cost_l = [w_coll, w_smooth]
    
    # Kết hợp các hàm chi phí thành một hàm chi phí tổng hợp
    cost = CostComposite(
        robot, traj_len, cost_func_list,
        weights_cost_l=weights_cost_l,
        tensor_args=tensor_args
    )

    #--------------------------------- Khởi Tạo MPOT ---------------------------------
    
    # Khởi tạo thuật toán Sinkhorn để giải quyết tối ưu vận tải
    linear_ot_solver = Sinkhorn(
        threshold=1e-6,
        inner_iterations=1,
        max_iterations=max_inner_iters,
    )
    
    # Các tham số cho thuật toán Sampling and Search (SS) trong MPOT
    ss_params = dict(
        epsilon=epsilon,
        ent_epsilon=ent_epsilon,
        step_radius=step_radius,
        probe_radius=probe_radius,
        num_probe=num_probe,
        min_iterations=5,
        max_iterations=max_outer_iters,
        threshold=2e-3,
        store_history=True,  # Lưu lại lịch sử các bước lặp để trực quan hóa sau này
        tensor_args=tensor_args,
    )

    # Các tham số cần thiết để khởi tạo trình lập kế hoạch MPOT
    mpot_params = dict(
        objective_fn=cost,  # Hàm chi phí tổng hợp
        linear_ot_solver=linear_ot_solver,  # Thuật toán tối ưu vận tải
        ss_params=ss_params,  # Tham số cho thuật toán SS
        dim=2,  # Chiều không gian (2D)
        traj_len=traj_len,  # Độ dài chuyển động
        num_particles_per_goal=num_particles_per_goal,  # Số lượng hạt cho mỗi mục tiêu
        dt=dt,  # Bước thời gian
        start_state=start_state,  # Trạng thái khởi đầu
        multi_goal_states=multi_goal_states,  # Các trạng thái mục tiêu
        pos_limits=pos_limits,  # Giới hạn vị trí
        vel_limits=vel_limits,  # Giới hạn vận tốc
        polytope=polytope,  # Hình đa diện được sử dụng trong thuật toán
        fixed_goal=True,  # Mục tiêu cố định
        sigma_start_init=0.001,  # Tham số sigma ban đầu cho trạng thái khởi đầu
        sigma_goal_init=0.001,  # Tham số sigma ban đầu cho trạng thái mục tiêu
        sigma_gp_init=sigma_gp_init,  # Tham số sigma ban đầu cho GP
        seed=seed,  # Seed ngẫu nhiên
        tensor_args=tensor_args,
    )
    
    # Khởi tạo đối tượng trình lập kế hoạch MPOT với các tham số đã thiết lập
    planner = MPOT(**mpot_params)

    #--------------------------------- Tối Ưu Hóa ---------------------------------
    
    # Sử dụng TimerCUDA để đo thời gian tối ưu hóa trên GPU (nếu có)
    with TimerCUDA() as t:
        trajs, optim_state, opt_iters = planner.optimize()
    
    # Nội suy chuyển động để tăng số lượng điểm, giúp đánh giá va chạm chính xác hơn
    int_trajs = interpolate_trajectory(trajs, num_interpolation=3)
    
    # Kiểm tra va chạm với các chướng ngại vật trong môi trường
    colls = env.occupancy_map.get_collisions(int_trajs[..., :2]).any(dim=1)
    
    # Lấy số lần lặp của thuật toán Sinkhorn trong quá trình tối ưu hóa
    sinkhorn_iters = optim_state.linear_convergence[:opt_iters]
    
    # In ra thông tin về quá trình tối ưu hóa
    print(f'Optimization finished at {opt_iters}! Parallelization Quality (GOOD [%]): {(1 - colls.float().mean()) * 100:.2f}')
    print(f'Time(s) optim: {t.elapsed} sec')
    print(f'Average Sinkhorn Iterations: {sinkhorn_iters.mean():.2f}, min: {sinkhorn_iters.min():.2f}, max: {sinkhorn_iters.max():.2f}')

    # -------------------------------- Trực Quan Hóa ---------------------------------
    # Khởi tạo công cụ trực quan hóa với nhiệm vụ và trình lập kế hoạch đã thiết lập
    planner_visualizer = PlanningVisualizer(
        task=task,
        planner=planner
    )

    # Lấy lịch sử chuyển động từ trạng thái tối ưu hóa
    traj_history = optim_state.X_history[:opt_iters]
    traj_history = traj_history.view(opt_iters, -1, traj_len, 4)  # Định hình lại tensor theo số vòng lặp tối ưu hóa
    
    # Lấy tên tệp cơ bản (không có phần mở rộng) để sử dụng trong tên tệp video
    base_file_name = Path(os.path.basename(__file__)).stem
    
    # Lấy vị trí của robot từ lịch sử chuyển động
    pos_trajs_iters = robot.get_position(traj_history)

    # Tạo video minh họa quá trình tối ưu hóa trong không gian joint của robot
    planner_visualizer.animate_opt_iters_joint_space_state(
        trajs=traj_history,
        pos_start_state=start_state,
        vel_start_state=torch.zeros_like(start_state),
        video_filepath=f'{base_file_name}-joint-space-opt-iters.mp4',
        n_frames=max((2, opt_iters // 5)),  # Số khung hình trong video
        anim_time=5  # Thời gian thực hiện animation (giây)
    )

    # Tạo video minh họa quá trình tối ưu hóa của robot trong không gian làm việc
    planner_visualizer.animate_opt_iters_robots(
        trajs=pos_trajs_iters, start_state=start_state,
        video_filepath=f'{base_file_name}-traj-opt-iters.mp4',
        n_frames=max((2, opt_iters // 5)),  # Số khung hình trong video
        anim_time=5  # Thời gian thực hiện animation (giây)
    )

    # Hiển thị các đồ thị đã được vẽ bởi Matplotlib (nếu có)
    plt.show()