from gsloc.inference.test import TestConfig, Test
from pathlib import Path
from gsloc.models import opr_graph_extention as network 
import torch
from torchvision.transforms import functional as F
from mmpr.models import MegaLoc
from gsloc.datasets import ThreeRScan, SberRobotics

from torchvision import transforms as T
from gsloc.utils.visual import plot_metrics_from_parquet, plot_metrics_from_experiment_dir
from gsloc.models import FoLBase
from gsloc.models import SelaVPRplusplusBaseRerank


import random
import numpy as np

def make_deterministic(seed=0):
    """Make results deterministic. If seed == -1, do not make deterministic.
    Running the script in a deterministic way might slow it down.
    """
    if seed == -1:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
make_deterministic(0)

similarity_kwargs_list = [
    {
        "mode": "room",
        "trans_tol_m": 3,
        "rot_tol_deg": 180
    },
    {
        "mode": "pose",
        "trans_tol_m": 3,
        "rot_tol_deg": 180
    },
    {
        "mode": "pose",
        "trans_tol_m": 2,
        "rot_tol_deg": 90
    },
]

seq_filter_kwargs_list = [
    {
        "seq_similarity_filter_mode": "none",
        "seq_similarity_trans_tol_m": 0.5,
        "seq_similarity_rot_tol_deg": 15
    },
    {
        "seq_similarity_filter_mode": "pose",
        "seq_similarity_trans_tol_m": 0.5,
        "seq_similarity_rot_tol_deg": 15
    },
    {
        "seq_similarity_filter_mode": "pose",
        "seq_similarity_trans_tol_m": 1,
        "seq_similarity_rot_tol_deg": 30
    },
]

image_transform_fn = T.Compose([
    T.ToTensor(),
    T.Resize([322, 322], antialias=True),
    T.Lambda(lambda x: F.rotate(x, angle=-90)),  # 90° clockwise
    T.Normalize(
        mean=[0.44420420130352495, 0.41322746532289134, 0.3678658064565412], 
        std=[0.24352604373543688, 0.24045797651069503, 0.24250136992133814]
    ),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


fol_base = FoLBase()  # веса из weights/FoL_base.pth
fol_base = fol_base.to(device)
fol_base.eval()


def run_test(
    tests_path, 
    dataset_path, 
    dataset_name,
    date, graph_type, 
    graphmodel_type, 
    image_model_type, 
    rerank_k, 
    per_frame_k, 
    filter_type, 
    similarity_type, 
    graph_dir, 
    edge_normalizer_path, 
    scene_list_path, 
    query_list_path,
    room_json_path,
    seq_filter_kwargs,
    similarity_kwargs,
    graph_model, 
    image_model,
    time_test,
    model_self_rerank_flag,
    rerank_descriptor_save_flag,
    ):

    model_name = f"{graphmodel_type}x{image_model_type}" if image_model_type != "None" and graphmodel_type != "None" \
        else (graphmodel_type + "_pure") if graphmodel_type != "None" else image_model_type
    today_dataset_test_path = tests_path / date / dataset_name
    test_path = today_dataset_test_path / graph_type / model_name if graph_type != "None" else today_dataset_test_path / model_name
    index_path = today_dataset_test_path / "cache" / "indexes" / graph_type / (graphmodel_type + "graph") if graph_type != "None" else today_dataset_test_path / "cache" / "indexes" / image_model_type
    query_cache_path = today_dataset_test_path / "cache" / "query_cache" / graph_type / (graphmodel_type + "graph") if graph_type != "None" else today_dataset_test_path / "cache" / "query_cache" / image_model_type
    rerank_index_path = today_dataset_test_path / "cache" / "indexes" / image_model_type if graph_type != "None" else "None"
    rerank_query_cache_path = today_dataset_test_path / "cache" / "query_cache" / image_model_type if graph_type != "None" else "None"
    frames_path = test_path / ("rerank_k_" + str(rerank_k) + "_per_frame_k_" + str(per_frame_k)) / "frames.npz"
    bench_report_path = test_path / ("rerank_k_" + str(rerank_k) + "_per_frame_k_" + str(per_frame_k)) / filter_type / similarity_type
    modality = []
    if image_model is not None:
        modality.append("image")
    if graph_model is not None:
        modality.append("graph")

    if model_self_rerank_flag:
        # query_cache_path = query_cache_path if rerank_descriptor_save_flag else None
        rerank_index_path = str(today_dataset_test_path / "cache" / "indexes" / image_model_type) + "_rerank" if rerank_descriptor_save_flag else None
        rerank_query_cache_path = str(today_dataset_test_path / "cache" / "query_cache" / image_model_type) + "_rerank" if rerank_descriptor_save_flag else None

    dataset_class = ThreeRScan if dataset_name == "3RScan" else SberRobotics

    cfg = TestConfig(
        dataset_path=dataset_path,
        test_path=test_path,
        index_path=index_path,
        rerank_index_path=rerank_index_path,
        query_cache_path=query_cache_path,
        rerank_query_cache_path=rerank_query_cache_path,
        bench_report_path=bench_report_path,
        graph_path=graph_dir,   
        dataset_class=dataset_class,
        modality=tuple(modality),
        filter_kwargs={"similarity_filter_mode": "none"},
        seq_filter_kwargs=seq_filter_kwargs,
        scene_list_path=scene_list_path,
        query_list_path=query_list_path,
        room_json_path=room_json_path,
        edge_normalizer_path=edge_normalizer_path,
        image_transform_fn=image_transform_fn,
        graph_feat_dim=4,
        graph_edge_attr_dim=10,
        graph_rotate=True,
        device=device,
        batch_size=16,
        time_test=time_test,
        num_workers=4,
        model=graph_model if graph_model is not None else image_model,
        rerank_model=image_model if graph_model is not None else None,
        model_self_rerank_flag=model_self_rerank_flag,
        rerank_descriptor_save_flag=rerank_descriptor_save_flag,
        rerank_k=rerank_k,
        per_frame_k_used=per_frame_k,
        final_k=25,
        seq_lengths=[1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 35],
        recall_at_k=[1, 5, 10, 25],
        similarity_kwargs=similarity_kwargs,
        std_mode="global",
        scene_df_field="scene",
        pose_df_field="pose",
        frames_path=frames_path
    )

    test = Test(cfg)
    test.run()

tests_path = Path("/home/kartashov_ga/projects/GSLoc/data/tests")
dataset_path = Path("/mnt/external_usb_hdd/6YL/Datasets/SberRobotics")
date = "26-06-01"
dataset_name = "Sber" #"3RScan_BIG"
graph_type = "None"
graphmodel_type = "None"
graph_model = None
image_model_type = "FoL_base"
image_model = fol_base

rerank_k_list = [50]
rerank_k = rerank_k_list[0]

per_frame_k_list = [10, 25, 50, 100]
per_frame_k = per_frame_k_list[1]

filter_names = ["base_seq_report", "nearfilter_seq_report", "farfilter_seq_report"]
filter_type = filter_names[0]
seq_filter_kwargs = seq_filter_kwargs_list[0]

similarity_names = ["room-sim", "pose-far-sim", "pose-near-sim"]
similarity_type = similarity_names[1]

graph_dir = "maps/SceneGraphs_pt"
edge_normalizer_path="/mnt/external_usb_hdd/6YL/Datasets/3RScan/weights/graphs/gatv1/edge_normalizer (1).pt"
scene_list_path = "/mnt/external_usb_hdd/6YL/Datasets/SberRobotics/database_maps.txt"# "/mnt/external_usb_hdd/6YL/Datasets/3RScan/files/BIG_test_scans_db.txt"
query_list_path = "/mnt/external_usb_hdd/6YL/Datasets/SberRobotics/queries_maps.txt"
room_json_path = None

time_test = True
model_self_rerank_flag = True
rerank_descriptor_save_flag = True

#for j in range(len(seq_filter_kwargs_list)):
for j in range(len(rerank_k_list)):
    for i in range(1, len(similarity_kwargs_list)):
        run_test(
            tests_path=tests_path, 
            dataset_path=dataset_path, 
            dataset_name=dataset_name,
            date=date, 
            graph_type=graph_type, 
            graphmodel_type=graphmodel_type, 
            image_model_type=image_model_type, 
            rerank_k=rerank_k_list[j], 
            per_frame_k=per_frame_k, 
            filter_type=filter_type, 
            similarity_type=similarity_names[i], 
            graph_dir=graph_dir, 
            edge_normalizer_path=edge_normalizer_path, 
            scene_list_path=scene_list_path, 
            query_list_path=query_list_path,
            room_json_path=room_json_path, 
            seq_filter_kwargs=seq_filter_kwargs, 
            similarity_kwargs=similarity_kwargs_list[i],
            graph_model=graph_model,
            image_model=image_model,   
            time_test=time_test,
            model_self_rerank_flag=model_self_rerank_flag,
            rerank_descriptor_save_flag=rerank_descriptor_save_flag
            )