#!/usr/bin/env python3
"""SMILEtrack对比试验 - 适配YOLOv5/v8/v11三种版本权重
用于对比不同YOLO版本在SMILEtrack跟踪器上的性能差异
"""

import argparse
import time
import json
from pathlib import Path
import sys
from datetime import datetime

import cv2
import torch
import numpy as np
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt

# 导入SMILEtrack
sys.path.append('D:/SMILEtrack/SMILEtrack_Official-main')
from tracker.mc_SMILEtrack import SMILEtrack


class SMILEtrackComparison:
    """SMILEtrack对比试验类，支持YOLOv5/v8/v11三种版本"""

    def __init__(self, args, model_version='yolov8'):
        self.args = args
        self.model_version = model_version
        self.frame_rate = 30

        # 根据模型版本设置不同的参数
        self._setup_model_specific_params()

        # 创建跟踪器
        self.tracker = SMILEtrack(args, frame_rate=self.frame_rate)

        # 统计信息
        self.stats = {
            'frame_count': 0,
            'total_unique_vehicles': 0,  # 累计出现过的不同车辆ID总数
            'unique_ids': set(),  # 记录所有出现过的唯一ID
            'id_switches': 0,  # ID跳变次数
            'vehicle_detections': 0,
            'processing_times': [],
            'track_counts_per_frame': [],
            'current_ids': set(),
            'previous_ids': set(),
            'id_track_lengths': {},  # 每个ID的跟踪长度
            'id_switch_events': []  # 记录ID跳变事件
        }

        # 用于IoU匹配的存储
        self.prev_track_positions = {}  # 存储上一帧的轨迹位置

        print(f"初始化 {model_version.upper()} 版本对比试验")

    def _setup_model_specific_params(self):
        """根据模型版本设置特定参数"""
        # 使用相对路径访问权重文件
        weights_dir = Path(__file__).parent.parent / 'weights'

        if self.model_version == 'yolov5':
            # YOLOv5特定参数
            self.img_size = 640
            self.conf_thres = 0.25
            self.iou_thres = 0.45
            self.weights_path = str(weights_dir / 'yolov5l.pt')

        elif self.model_version == 'yolov8':
            # YOLOv8特定参数
            self.img_size = 1280
            self.conf_thres = 0.25
            self.iou_thres = 0.45
            self.weights_path = str(weights_dir / 'yolov8l.pt')

        elif self.model_version == 'yolov11':
            # YOLOv11特定参数
            self.img_size = 1280
            self.conf_thres = 0.25
            self.iou_thres = 0.45
            self.weights_path = str(weights_dir / 'yolo11l.pt')

        else:
            raise ValueError(f"不支持的模型版本: {self.model_version}")

    def load_model(self):
        """加载YOLO模型"""
        print(f"加载 {self.model_version.upper()} 模型: {self.weights_path}")

        try:
            if self.model_version == 'yolov5':
                # YOLOv5需要特殊处理
                model = torch.hub.load('ultralytics/yolov5', 'custom',
                                       path=self.weights_path,
                                       force_reload=False)
                model.conf = self.conf_thres
                model.iou = self.iou_thres
                model.classes = None  # 所有类别
            else:
                # YOLOv8和YOLOv11使用ultralytics
                model = YOLO(self.weights_path)

            print(f"✅ {self.model_version.upper()} 模型加载成功")
            return model

        except Exception as e:
            print(f"❌ 加载 {self.model_version.upper()} 模型失败: {e}")
            return None

    def detect_vehicles(self, model, image):
        """使用模型检测车辆"""
        start_time = time.time()

        if self.model_version == 'yolov5':
            # YOLOv5检测
            results = model(image, size=self.img_size)

            detections = []
            if results.xyxy[0] is not None:
                for det in results.xyxy[0].cpu().numpy():
                    x1, y1, x2, y2, conf, cls = det[:6]

                    # 车辆类别: car(2), motorcycle(3), bus(5), truck(7)
                    if int(cls) in [2, 3, 5, 7]:
                        detections.append([x1, y1, x2, y2, conf, int(cls), 0.0])
                        self.stats['vehicle_detections'] += 1

        else:
            # YOLOv8/YOLOv11检测
            results = model(
                image,
                imgsz=self.img_size,
                conf=self.conf_thres,
                iou=self.iou_thres,
                verbose=False
            )

            detections = []
            for r in results:
                if r.boxes is not None:
                    boxes = r.boxes
                    for box in boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = xyxy
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())

                        # 车辆类别
                        if cls in [2, 3, 5, 7]:
                            detections.append([x1, y1, x2, y2, conf, cls, 0.0])
                            self.stats['vehicle_detections'] += 1

        processing_time = time.time() - start_time
        self.stats['processing_times'].append(processing_time)

        return np.array(detections) if detections else np.array([])

    def update_tracker(self, detections, image):
        """更新跟踪器并统计ID跳变 - 基于IoU匹配的跳变检测"""
        # 记录当前ID
        self.stats['previous_ids'] = self.stats['current_ids'].copy()

        # 将检测结果转换为DeepSORT需要的格式（这里适配SMILEtrack）
        # SMILEtrack的update方法期望: detections, image

        # 更新跟踪器
        if len(detections) > 0:
            # 确保detections是numpy数组格式
            if not isinstance(detections, np.ndarray):
                detections = np.array(detections)
            online_targets = self.tracker.update(detections, image)
        else:
            # 没有检测时，只更新跟踪器（预测）
            online_targets = self.tracker.update(np.empty((0, 7)), image)

        # 获取当前激活的跟踪目标ID
        self.stats['current_ids'] = set()

        # 解析跟踪结果（SMILEtrack返回的online_targets格式）
        for track in online_targets:
            if hasattr(track, 'is_activated') and track.is_activated:
                track_id = track.track_id
                self.stats['current_ids'].add(track_id)

                # 记录每个ID的跟踪长度
                if track_id in self.stats['id_track_lengths']:
                    self.stats['id_track_lengths'][track_id] += 1
                else:
                    self.stats['id_track_lengths'][track_id] = 1
                    # 新出现的车辆
                    self.stats['total_unique_vehicles'] += 1
                    print(f"  [新车辆] ID:{track_id} 在第{self.stats['frame_count']}帧出现")
            elif hasattr(track, 'track_id'):
                # 兼容不同的track对象格式
                track_id = track.track_id
                self.stats['current_ids'].add(track_id)

                if track_id in self.stats['id_track_lengths']:
                    self.stats['id_track_lengths'][track_id] += 1
                else:
                    self.stats['id_track_lengths'][track_id] = 1
                    self.stats['total_unique_vehicles'] += 1
                    print(f"  [新车辆] ID:{track_id} 在第{self.stats['frame_count']}帧出现")

        # 更新累计唯一车辆数
        self.stats['unique_ids'].update(self.stats['current_ids'])

        # ID跳变检测逻辑 - 基于IoU匹配
        if self.stats['frame_count'] > 1 and len(self.stats['previous_ids']) > 0:
            # 计算ID集合的变化
            lost_ids = self.stats['previous_ids'] - self.stats['current_ids']
            new_ids = self.stats['current_ids'] - self.stats['previous_ids']

            # 基于IoU匹配的真实跳变检测
            if len(lost_ids) > 0 and len(new_ids) > 0:
                if hasattr(self, 'prev_track_positions') and self.prev_track_positions:
                    real_switches = 0
                    matched_switches = []
                    new_ids_copy = new_ids.copy()  # 复制一份用于修改

                    # 获取当前帧所有track的位置信息
                    current_track_positions = {}
                    for track in online_targets:
                        track_id = track.track_id if hasattr(track, 'track_id') else None
                        if track_id is not None:
                            if hasattr(track, 'tlwh'):
                                # SMILEtrack的STrack对象有tlwh属性
                                tlwh = track.tlwh
                                bbox = [tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]]
                                current_track_positions[track_id] = bbox
                            elif hasattr(track, 'bbox'):
                                bbox = track.bbox
                                current_track_positions[track_id] = bbox

                    # 遍历丢失的ID
                    for lost_id in lost_ids:
                        if lost_id in self.prev_track_positions:
                            lost_bbox = self.prev_track_positions[lost_id]

                            # 在当前帧的新ID中寻找最佳匹配
                            best_iou = 0
                            best_new_id = None

                            for new_id in new_ids_copy:
                                if new_id in current_track_positions:
                                    new_bbox = current_track_positions[new_id]
                                    iou = self._calculate_iou(lost_bbox, new_bbox)
                                    if iou > best_iou and iou > 0.3:  # IoU阈值0.3
                                        best_iou = iou
                                        best_new_id = new_id

                            # 如果找到高IoU匹配，认为是真实的ID跳变
                            if best_new_id is not None:
                                real_switches += 1
                                matched_switches.append((lost_id, best_new_id))
                                new_ids_copy.discard(best_new_id)  # 从新ID中移除已匹配的

                    # 更新跳变统计
                    self.stats['id_switches'] += real_switches

                    if real_switches > 0:
                        self.stats['id_switch_events'].append({
                            'frame': self.stats['frame_count'],
                            'lost_ids': [lost for lost, _ in matched_switches],
                            'new_ids': [new for _, new in matched_switches],
                            'switch_count': real_switches
                        })
                        print(f"  [ID跳变] 帧{self.stats['frame_count']}: "
                              f"检测到{real_switches}次真实跳变 (总丢失{len(lost_ids)}个, 新增{len(new_ids)}个)")
                else:
                    # 如果还没有prev_track_positions，使用原来的简单逻辑
                    if len(lost_ids) > 0 and len(new_ids) > 0:
                        switch_count = min(len(lost_ids), len(new_ids))
                        self.stats['id_switches'] += switch_count

                        if switch_count > 0:
                            self.stats['id_switch_events'].append({
                                'frame': self.stats['frame_count'],
                                'lost_ids': list(lost_ids),
                                'new_ids': list(new_ids),
                                'switch_count': switch_count
                            })
                            print(f"  [ID跳变] 帧{self.stats['frame_count']}: "
                                  f"丢失{len(lost_ids)}个ID({list(lost_ids)[:3]}...), "
                                  f"新增{len(new_ids)}个ID({list(new_ids)[:3]}...), "
                                  f"跳变数:{switch_count}")

        # 存储当前帧的轨迹位置，用于下一帧的IoU匹配
        self.prev_track_positions = {}
        for track in online_targets:
            track_id = track.track_id if hasattr(track, 'track_id') else None
            if track_id is not None:
                if hasattr(track, 'tlwh'):
                    # SMILEtrack的STrack对象有tlwh属性
                    tlwh = track.tlwh
                    bbox = [tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]]
                    self.prev_track_positions[track_id] = bbox
                elif hasattr(track, 'bbox'):
                    self.prev_track_positions[track_id] = track.bbox

        # 更新统计
        self.stats['track_counts_per_frame'].append(len(online_targets))

        # 返回当前激活的跟踪目标（用于绘制）
        return online_targets

    def _calculate_iou(self, bbox1, bbox2):
        """计算两个边界框的IoU"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # 计算交集
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # 计算并集
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def draw_results(self, image, online_targets, show_stats=True):
        """绘制跟踪结果 - 兼容SMILEtrack格式"""
        # 绘制跟踪框
        for t in online_targets:
            # 兼容不同的track对象格式
            if hasattr(t, 'tlwh'):
                tlwh = t.tlwh
                tid = t.track_id
            elif hasattr(t, 'bbox'):
                # 如果只有bbox
                bbox = t.bbox
                tlwh = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                tid = t.track_id
            else:
                continue

            if tlwh[2] * tlwh[3] > self.args.min_box_area:
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h

                # 根据跟踪稳定性选择颜色
                track_len = self.stats['id_track_lengths'].get(tid, 0)
                if track_len > 10:
                    color = (0, 255, 0)  # 稳定跟踪 - 绿色
                elif track_len > 5:
                    color = (0, 165, 255)  # 中等跟踪 - 橙色
                else:
                    color = (0, 0, 255)  # 新跟踪 - 红色

                # 绘制边界框
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                # 绘制跟踪轨迹
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                cv2.circle(image, (center_x, center_y), 3, color, -1)

                # 显示ID和跟踪长度
                label = f"ID:{tid}({track_len})"
                cv2.putText(image, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 显示统计信息
        if show_stats:
            # 计算ID跳变率（每100次检测的跳变数）
            switch_rate_per_100 = (self.stats['id_switches'] / max(1, self.stats['total_unique_vehicles'])) * 100

            info_lines = [
                f"Model: {self.model_version.upper()}",
                f"Frame: {self.stats['frame_count']}",
                f"Current Tracks: {len(online_targets)}",
                f"Total Vehicles: {self.stats['total_unique_vehicles']}",
                f"ID Switches: {self.stats['id_switches']}",
                f"Switch Rate: {switch_rate_per_100:.2f}%",
                f"FPS: {1 / np.mean(self.stats['processing_times'][-10:]):.1f}" if len(
                    self.stats['processing_times']) >= 10 else "FPS: calculating..."
            ]

            y_offset = 30
            for i, line in enumerate(info_lines):
                cv2.putText(image, line, (10, y_offset + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return image

    def get_final_statistics(self):
        """获取最终统计信息"""
        # 计算ID跳变率（每100次检测）
        switch_rate_per_100 = (self.stats['id_switches'] / max(1, self.stats['total_unique_vehicles'])) * 100

        # 计算平均跟踪长度
        avg_track_length = np.mean(list(self.stats['id_track_lengths'].values())) if self.stats[
            'id_track_lengths'] else 0

        avg_processing_time = np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0
        avg_fps = 1 / avg_processing_time if avg_processing_time > 0 else 0

        return {
            'model_version': self.model_version,
            'total_frames': self.stats['frame_count'],
            'total_unique_vehicles': self.stats['total_unique_vehicles'],
            'id_switches': self.stats['id_switches'],
            'switch_rate_percent': switch_rate_per_100,
            'vehicle_detections': self.stats['vehicle_detections'],
            'avg_processing_time_ms': avg_processing_time * 1000,
            'avg_fps': avg_fps,
            'max_tracks_per_frame': max(self.stats['track_counts_per_frame']) if self.stats[
                'track_counts_per_frame'] else 0,
            'avg_tracks_per_frame': np.mean(self.stats['track_counts_per_frame']) if self.stats[
                'track_counts_per_frame'] else 0,
            'avg_track_length_frames': avg_track_length,
            'total_switch_events': len(self.stats['id_switch_events'])
        }


def run_comparison_experiment(args, model_versions=['yolov5', 'yolov8', 'yolov11']):
    """运行对比试验"""
    print("=" * 70)
    print("SMILEtrack 对比试验 - YOLOv5/v8/v11 性能对比")
    print("=" * 70)

    # 检查视频文件
    if not Path(args.source).exists():
        print(f"❌ 错误: 视频文件不存在 {args.source}")
        return

    # 获取视频信息
    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"📹 视频信息:")
    print(f"   文件: {args.source}")
    print(f"   尺寸: {width}x{height}")
    print(f"   帧率: {fps:.1f} FPS")
    print(f"   总帧数: {total_frames}")
    print(f"   测试模型: {', '.join(model_versions)}")
    print()

    # 在创建输出目录之前添加时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"experiment_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 存储所有结果
    all_results = []
    experiment_start_time = time.time()

    # 对每个模型版本运行试验
    for model_version in model_versions:
        print(f"\n{'=' * 40}")
        print(f"开始 {model_version.upper()} 版本试验")
        print(f"{'=' * 40}")

        # 创建实验实例
        experiment = SMILEtrackComparison(args, model_version)

        # 加载模型
        model = experiment.load_model()
        if model is None:
            print(f"跳过 {model_version.upper()} 版本")
            continue

        # 准备输出视频
        if args.save_videos:
            video_output_path = output_dir / f"{model_version}_output.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(video_output_path), fourcc, fps, (width, height))

        # 打开视频
        cap = cv2.VideoCapture(args.source)
        frame_count = 0
        start_time = time.time()
        last_report_time = start_time

        # 处理视频帧
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            experiment.stats['frame_count'] = frame_count

            # 限制处理帧数（如果指定）
            if args.max_frames and frame_count > args.max_frames:
                break

            # 检测车辆
            detections = experiment.detect_vehicles(model, frame)

            # 更新跟踪器
            online_targets = experiment.update_tracker(detections, frame)

            # 绘制结果
            result_frame = frame.copy()
            result_frame = experiment.draw_results(result_frame, online_targets, show_stats=True)

            # 保存视频
            if args.save_videos:
                video_writer.write(result_frame)

            # 显示进度（每50帧或每5秒）
            current_time = time.time()
            if frame_count % 50 == 0 or (current_time - last_report_time) >= 5:
                elapsed = current_time - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"  {model_version}: 帧 {frame_count}/{min(total_frames, args.max_frames or total_frames)}, "
                      f"累计车辆: {experiment.stats['total_unique_vehicles']}, "
                      f"当前跟踪: {len(online_targets)}, "
                      f"ID跳变: {experiment.stats['id_switches']}, "
                      f"FPS: {current_fps:.1f}")
                last_report_time = current_time

        # 清理
        cap.release()
        if args.save_videos:
            video_writer.release()

        # 获取最终统计
        final_stats = experiment.get_final_statistics()
        all_results.append(final_stats)

        # 打印结果
        print(f"\n✅ {model_version.upper()} 版本完成!")
        print(f"   处理帧数: {final_stats['total_frames']}")
        print(f"   累计出现车辆数: {final_stats['total_unique_vehicles']}")
        print(f"   ID跳变次数: {final_stats['id_switches']}")
        print(f"   ID跳变率: {final_stats['switch_rate_percent']:.4f}% (每100次检测)")
        print(f"   平均跟踪长度: {final_stats['avg_track_length_frames']:.1f}帧")
        print(f"   平均FPS: {final_stats['avg_fps']:.1f}")
        print(f"   车辆检测数: {final_stats['vehicle_detections']}")
        print(f"   ID跳变事件数: {final_stats['total_switch_events']}")

    # 实验总时间
    total_experiment_time = time.time() - experiment_start_time

    # 生成对比报告
    generate_comparison_report(all_results, output_dir, total_experiment_time)

    print(f"\n{'=' * 70}")
    print("🎉 对比试验完成!")
    print(f"   总耗时: {total_experiment_time:.1f}秒")
    print(f"   结果保存到: {output_dir}")
    print(f"{'=' * 70}")


def generate_comparison_report(results, output_dir, total_time):
    """生成对比试验报告"""
    print(f"\n📊 生成对比报告...")

    # 转换为DataFrame
    df = pd.DataFrame(results)

    # 保存为CSV
    csv_path = output_dir / "comparison_results.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"   CSV报告: {csv_path}")

    # 保存为JSON
    json_path = output_dir / "comparison_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_duration_seconds': total_time,
            'results': results
        }, f, indent=2, ensure_ascii=False)
    print(f"   JSON报告: {json_path}")

    # 生成可视化图表
    if len(results) > 1:
        generate_visualizations(df, output_dir)

    # 打印对比表格
    print(f"\n{'=' * 100}")
    print("对比试验结果汇总:")
    print(f"{'=' * 100}")
    print(f"{'模型版本':<12} {'累计车辆':<10} {'ID跳变':<10} {'跳变率(%)':<12} {'平均跟踪(帧)':<12} {'平均FPS':<10}")
    print(f"{'-' * 100}")

    for result in results:
        model = result['model_version'].upper()
        vehicles = result['total_unique_vehicles']
        switches = result['id_switches']
        switch_rate = f"{result['switch_rate_percent']:.2f}"
        track_len = f"{result['avg_track_length_frames']:.1f}"
        fps = f"{result['avg_fps']:.1f}"

        print(f"{model:<12} {vehicles:<10} {switches:<10} {switch_rate:<12} {track_len:<12} {fps:<10}")

    print(f"{'=' * 100}")

    # 找出最佳模型
    if results:
        # 按ID跳变率排序（越低越好）
        best_by_stability = min(results, key=lambda x: x['switch_rate_percent'])
        # 按FPS排序（越高越好）
        best_by_speed = max(results, key=lambda x: x['avg_fps'])
        # 按累计车辆数排序（越高越好，说明检测跟踪能力更强）
        best_by_detection = max(results, key=lambda x: x['total_unique_vehicles'])
        # 按平均跟踪长度排序（越长越好）
        best_by_track_len = max(results, key=lambda x: x['avg_track_length_frames'])

        print(f"\n🏆 性能排名:")
        print(f"   最稳定（ID跳变率最低）: {best_by_stability['model_version'].upper()} "
              f"({best_by_stability['switch_rate_percent']:.2f}%)")
        print(f"   最快（FPS最高）: {best_by_speed['model_version'].upper()} "
              f"({best_by_speed['avg_fps']:.1f} FPS)")
        print(f"   检测跟踪能力最强（累计车辆最多）: {best_by_detection['model_version'].upper()} "
              f"({best_by_detection['total_unique_vehicles']} 辆)")
        print(f"   跟踪最稳定（平均跟踪长度最长）: {best_by_track_len['model_version'].upper()} "
              f"({best_by_track_len['avg_track_length_frames']:.1f} 帧)")


def generate_visualizations(df, output_dir):
    """生成可视化图表"""
    try:
        # 设置中文字体（如果需要）
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('SMILEtrack 对比试验结果', fontsize=16, fontweight='bold')

        # 1. ID跳变率对比
        ax1 = axes[0, 0]
        models = df['model_version'].str.upper()
        switch_rates = df['switch_rate_percent']
        bars1 = ax1.bar(models, switch_rates, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax1.set_title('ID跳变率对比 (%)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('ID跳变率 (%)')
        ax1.grid(True, alpha=0.3)

        # 在柱子上添加数值
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{height:.2f}%', ha='center', va='bottom', fontsize=10)

        # 2. 平均FPS对比
        ax2 = axes[0, 1]
        fps_values = df['avg_fps']
        bars2 = ax2.bar(models, fps_values, color=['#95E1D3', '#F38181', '#FCE38A'])
        ax2.set_title('处理速度对比 (FPS)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('平均FPS')
        ax2.grid(True, alpha=0.3)

        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                     f'{height:.1f}', ha='center', va='bottom', fontsize=10)

        # 3. 累计车辆数对比
        ax3 = axes[1, 0]
        vehicle_counts = df['total_unique_vehicles']
        bars3 = ax3.bar(models, vehicle_counts, color=['#A8E6CF', '#DCEDC1', '#FFD3B6'])
        ax3.set_title('累计出现车辆数对比', fontsize=14, fontweight='bold')
        ax3.set_ylabel('车辆数')
        ax3.grid(True, alpha=0.3)

        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                     f'{int(height)}', ha='center', va='bottom', fontsize=10)

        # 4. 平均跟踪长度对比
        ax4 = axes[1, 1]
        track_lengths = df['avg_track_length_frames']
        bars4 = ax4.bar(models, track_lengths, color=['#FFAAA5', '#FF8B94', '#A8E6CF'])
        ax4.set_title('平均跟踪长度对比 (帧)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('跟踪长度 (帧)')
        ax4.grid(True, alpha=0.3)

        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height + 1,
                     f'{height:.1f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        # 保存图表
        chart_path = output_dir / "comparison_charts.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   可视化图表: {chart_path}")

    except Exception as e:
        print(f"   生成图表时出错: {e}")


def main():
    parser = argparse.ArgumentParser(description='SMILEtrack对比试验 - YOLOv5/v8/v11性能对比')

    # 输入输出参数
    parser.add_argument('--source', type=str, default='test2.mp4',
                        help='视频文件路径 (默认: test2.mp4)')
    parser.add_argument('--output-dir', type=str, default='comparison_results',
                        help='输出目录 (默认: comparison_results)')

    # 试验控制参数
    parser.add_argument('--models', nargs='+', default=['yolov5', 'yolov8', 'yolov11'],
                        choices=['yolov5', 'yolov8', 'yolov11'],
                        help='要测试的模型版本 (默认: 全部)')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='最大处理帧数 (None表示处理所有帧)')
    parser.add_argument('--save-videos', action='store_true',
                        help='保存每个模型的输出视频')

    # SMILEtrack参数
    parser.add_argument('--match-thresh', type=float, default=0.7,
                        help='匹配阈值 (默认: 0.7)')
    parser.add_argument('--proximity-thresh', type=float, default=0.5,
                        help='邻近阈值 (默认: 0.5)')
    parser.add_argument('--appearance-thresh', type=float, default=0.25,
                        help='外观相似度阈值 (默认: 0.25)')
    parser.add_argument('--track-buffer', type=int, default=30,
                        help='轨迹缓冲区 (默认: 30)')
    parser.add_argument('--new-track-thresh', type=float, default=0.6,
                        help='新轨迹阈值 (默认: 0.6)')
    parser.add_argument('--track-high-thresh', type=float, default=0.3,
                        help='高置信度检测阈值 (默认: 0.3)')
    parser.add_argument('--track-low-thresh', type=float, default=0.1,
                        help='低置信度检测阈值 (默认: 0.1)')
    parser.add_argument('--min-box-area', type=float, default=10,
                        help='最小框面积 (默认: 10)')

    # 必要参数
    parser.add_argument('--name', type=str, default='comparison_experiment')
    parser.add_argument('--ablation', action='store_true')
    parser.add_argument('--cmc-method', default='sparseOptFlow')
    parser.add_argument('--with-reid', action='store_true', help='使用ReID')
    parser.add_argument('--mot20', action='store_true', help='MOT20模式')
    parser.add_argument('--fuse-score', action='store_true', help='融合分数')

    opt = parser.parse_args()

    # 运行对比试验
    run_comparison_experiment(opt, opt.models)


if __name__ == "__main__":
    main()
