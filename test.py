from libero.libero import benchmark, get_libero_path, set_libero_default_path
import os
import h5py
import imageio

from libero.libero.envs import OffScreenRenderEnv

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

bddl_files_default_path = get_libero_path("bddl_files")

benchmark_dict = benchmark.get_benchmark_dict()
print(benchmark_dict)

task_name = "spatial"
benchmark_instance = benchmark_dict[f"libero_{task_name}"]()
num_tasks = 2 #benchmark_instance.get_num_tasks()
num_demos = 1

for task_id in range(num_tasks):
    task = benchmark_instance.get_task(task_id)
    initial_states = benchmark_instance.get_task_init_states(task_id)

    example_demo_file = os.path.join("/network/projects/real-g-grp/libero", benchmark_instance.get_task_demonstration(task_id))

    for demo_id in range(num_demos):

        with h5py.File(example_demo_file, "r") as f:
            actions = f[f"data/demo_{demo_id}/actions"][()]
            states = f[f"data/demo_{demo_id}/states"][()]

        env_args = {
            "bddl_file_name": os.path.join(bddl_files_default_path, task.problem_folder, task.bddl_file),
            "camera_heights": 512,
            "camera_widths": 512,
            "scene_properties": {
                "floor_style": "light-gray",
                "wall_style": "dark-green",
                "table_style": "blue-plastic",
            }

        }

        env = OffScreenRenderEnv(**env_args)

        env.reset()
        obs = env.set_init_state(states[0])

        for _ in range(10):
            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
            continue
        
        demo_images = [obs["agentview_image"]]
        for action in actions:
           obs, reward, done, info = env.step(action)
           demo_images.append(obs["agentview_image"])
           if done:
               print("Done demo, task:", demo_id, task_id)
               break
        

        video_writer = imageio.get_writer(f"output_{task_name}_task{task_id}_demo{demo_id}_blue_dg.mp4", fps=60)
        for image in demo_images:
            video_writer.append_data(image[::-1])
        video_writer.close()



    env.close()