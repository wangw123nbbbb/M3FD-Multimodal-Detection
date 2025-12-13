import os
import warnings
import numpy as np
from prettytable import PrettyTable

warnings.filterwarnings('ignore')
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info


def get_weight_size(path):
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'


if __name__ == '__main__':
    model_path = './runs/1/train/weights/best.pt'
    model = YOLO(model_path)
    result = model.val(data='./M3FD-rgbt.yaml',
                       split='val',
                       imgsz=640,
                       batch=4,
                       use_simotm="RGBT",
                       channels=4,
                       project='runs/1',
                       name='test',
                       )

    if model.task == 'detect':
        model_names = list(result.names.values())
        preprocess_time_per_image = result.speed['preprocess']
        inference_time_per_image = result.speed['inference']
        postprocess_time_per_image = result.speed['postprocess']
        all_time_per_image = preprocess_time_per_image + inference_time_per_image + postprocess_time_per_image

        n_l, n_p, n_g, flops = model_info(model.model)

        print('-' * 20 + '数据以以下结果为准' + '-' * 20)
        print('-' * 20 + '数据以以下结果为准' + '-' * 20)
        print('-' * 20 + '数据以以下结果为准' + '-' * 20)
        print('-' * 20 + '数据以以下结果为准' + '-' * 20)
        print('-' * 20 + '数据以以下结果为准' + '-' * 20)

        model_info_table = PrettyTable()
        model_info_table.title = "Model Info"
        model_info_table.field_names = ["GFLOPs", "Parameters", "前处理时间/一张图", "推理时间/一张图",
                                        "后处理时间/一张图", "FPS(前处理+模型推理+后处理)", "FPS(推理)",
                                        "Model File Size"]
        model_info_table.add_row([f'{flops:.1f}', f'{n_p:,}',
                                  f'{preprocess_time_per_image / 1000:.6f}s', f'{inference_time_per_image / 1000:.6f}s',
                                  f'{postprocess_time_per_image / 1000:.6f}s', f'{1000 / all_time_per_image:.2f}',
                                  f'{1000 / inference_time_per_image:.2f}', f'{get_weight_size(model_path)}MB'])
        print(model_info_table)

        model_metrics_table = PrettyTable()
        model_metrics_table.title = "Model metrics"
        model_metrics_table.field_names = ["Class Name", "Precision", "Recall", "F1-Score", "mAP50", "mAP75",
                                           "mAP50-95"]
        for idx, cls_name in enumerate(model_names):
            model_metrics_table.add_row([
                cls_name,
                f"{result.box.p[idx]:.4f}",
                f"{result.box.r[idx]:.4f}",
                f"{result.box.f1[idx]:.4f}",
                f"{result.box.ap50[idx]:.4f}",
                f"{result.box.all_ap[idx, 5]:.4f}",
                f"{result.box.ap[idx]:.4f}"
            ])
        model_metrics_table.add_row([
            "all(平均数据)",
            f"{result.results_dict['metrics/precision(B)']:.4f}",
            f"{result.results_dict['metrics/recall(B)']:.4f}",
            f"{np.mean(result.box.f1):.4f}",
            f"{result.results_dict['metrics/mAP50(B)']:.4f}",
            f"{np.mean(result.box.all_ap[:, 5]):.4f}",
            f"{result.results_dict['metrics/mAP50-95(B)']:.4f}"
        ])
        print(model_metrics_table)

        with open(result.save_dir / 'data.txt', 'w+') as f:
            f.write(str(model_info_table))
            f.write('\n')
            f.write(str(model_metrics_table))

        print('-' * 20, f'结果已保存至{result.save_dir}/data.txt...', '-' * 20)
        print('-' * 20, f'结果已保存至{result.save_dir}/data.txt...', '-' * 20)
        print('-' * 20, f'结果已保存至{result.save_dir}/data.txt...', '-' * 20)
        print('-' * 20, f'结果已保存至{result.save_dir}/data.txt...', '-' * 20)
        print('-' * 20, f'结果已保存至{result.save_dir}/data.txt...', '-' * 20)
