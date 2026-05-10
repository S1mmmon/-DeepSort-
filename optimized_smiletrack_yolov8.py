#!/usr/bin/env python3
"""优化版SMILEtrack跟踪器 - 支持YOLOv8
可以自定义负载参数，解决除画后ID跳变问题
"""

import argparse
import time
from pathlib import Path
import sys

import cv2
import torch
import numpy as np
from ultralytics import YOLO


# 导入SMILEtrack
sys.path.append('.')
from tracker.mc_SMILEtrack import SMILEtrack

class OptimizedSMILEtrack:
    """优化版SMILEtrack，可以调整关键参数解决ID跳变问题
    """
    
    def __init__(self, args, frame_rate=30):
        self.args = args
        self.tracker = SMILEtrack(args, frame_rate=frame_rate)
        
        # 跟踪参数优化
        self.optimization_params = {
            # 1. 匹配键值优化
            'match_thresh': args.match_thresh,  # 匹配键值，超低超加久规格
            'proximity_thresh': args.proximity_thresh,  # 邻近键值，用于IOU过滤
            'appearance_thresh': args.appearance_thresh,  # 大复相似度键值
            
            # 2. 跟踪路径管理优化
            'track_buffer': args.track_buffer,  # 跟踪路径範围，超大超加久不容易丢失
            'new_track_thresh': args.new_track_thresh,  # 新跟踪键值，超高超加久难创建新ID
            
            # 3. 检测键值优化
            'track_high_thresh': args.track_high_thresh,  # 高置信友度检测键值
            'track_low_thresh': args.track_low_thresh,  # 低置信友度检测键值
            
            # 4. 单元段滤消参数
            'min_box_area': args.min_box_area,  # 最小栏区面积，过滤小目标
        }
        
        print("=" * 60)
        print("优化版SMILEtrack参数配置")
        print("=" * 60)
        for key, value in self.optimization_params.items():
            print(f"  {key:20}: {value}")
        print()
        
        # 跟踪统计
        self.track_history = {}  # 跟踪历史记录
        self.id_switches = 0  # ID跳变次数
        self.total_tracks = 0  # 总跟踪数
        
    def update(self, detections, img):
        """更新跟踪器，并记录ID跳变"""
        
        # 记录当前帧的跟踪状态
        current_tracks = {}
        if hasattr(self.tracker, 'tracked_stracks'):
            for track in self.tracker.tracked_stracks:
                if track.is_activated:
                    tlwh = track.tlwh
                    center_x = tlwh[0] + tlwh[2] / 2
                    center_y = tlwh[1] + tlwh[3] / 2
                    current_tracks[track.track_id] = {
                        'center': (center_x, center_y),
                        'bbox': tlwh,
                        'score': track.score,
                        'frame': self.tracker.frame_id
                    }
        
        # 更新跟踪器
        online_targets = self.tracker.update(detections, img)
        
        # 检测ID跳变
        self._detect_id_switches(current_tracks)
        
        return online_targets
    
    def _detect_id_switches(self, previous_tracks):
        """检测ID跳变"""
        if not hasattr(self.tracker, 'tracked_stracks'):
            return
        
        current_ids = set()
        for track in self.tracker.tracked_stracks:
            if track.is_activated:
                current_ids.add(track.track_id)
        
        # 统计新ID和丢失的ID
        if hasattr(self, 'previous_ids'):
            new_ids = current_ids - self.previous_ids
            lost_ids = self.previous_ids - current_ids
            
            if new_ids and lost_ids:
                print(f"⚠️  可能ID跳变: 丢失 {len(lost_ids)} 个ID, 新增 {len(new_ids)} 个ID")
                self.id_switches += min(len(new_ids), len(lost_ids))
        
        self.previous_ids = current_ids
        self.total_tracks = max(self.total_tracks, len(current_ids))
    
    def get_statistics(self):
        """获取跟踪统计信息"""
        return {
            'total_tracks': self.total_tracks,
            'id_switches': self.id_switches,
            'switch_rate': self.id_switches / max(1, self.total_tracks),
            'current_tracks': len(self.previous_ids) if hasattr(self, 'previous_ids') else 0
        }


def main():
    parser = argparse.ArgumentParser(description='优化版SMILEtrack跟踪器')
    
    # 输入输出参数
    parser.add_argument('--source', type=str, default='test2.mp4', help='视频文件')
    parser.add_argument('--weights', type=str, default='weights/yolov8n.pt', help='YOLOv8权重')
    parser.add_argument('--output', type=str, default='output_optimized.mp4', help='输出视频')
    parser.add_argument('--view-img', action='store_true', help='显示结果')
    parser.add_argument('--save-vid', action='store_true', help='保存视频')
    
    # YOLOv8检测参数
    parser.add_argument('--img-size', type=int, default=1280, help='推理尺寸')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU阈值')
    parser.add_argument('--device', default='', help='设备')
    parser.add_argument('--classes', nargs='+', type=int, help='过滤类别')
    
    # ========== 优化参数区 ==========
    # 1. 匹配阈值优化（解决ID跳变的关键）
    parser.add_argument('--match-thresh', type=float, default=0.7, 
                       help='匹配阈值（默认0.7，降低到0.5-0.6可减少ID跳变但可能丢失跟踪）')
    
    parser.add_argument('--proximity-thresh', type=float, default=0.5,
                       help='邻近阈值（默认0.5，降低到0.3-0.4可减少误匹配）')
    
    parser.add_argument('--appearance-thresh', type=float, default=0.25,
                       help='外观相似度阈值（默认0.25，降低到0.1-0.2可增强ReID效果）')
    
    # 2. 轨迹管理优化
    parser.add_argument('--track-buffer', type=int, default=30,
                       help='轨迹缓冲区（默认30，增加到50-100可减少遮挡后丢失）')
    
    parser.add_argument('--new-track-thresh', type=float, default=0.6,
                       help='新轨迹阈值（默认0.4，增加到0.5-0.6可减少新ID创建）')
    
    # 3. 检测阈值优化
    parser.add_argument('--track-high-thresh', type=float, default=0.3,
                       help='高置信度检测阈值（默认0.3）')
    
    parser.add_argument('--track-low-thresh', type=float, default=0.1,
                       help='低置信度检测阈值（默认0.1）')
    
    # 4. 其他参数
    parser.add_argument('--min-box-area', type=float, default=10,
                       help='最小框面积（默认10，增加到50-100可过滤小目标）')
    
    parser.add_argument('--aspect-ratio-thresh', type=float, default=1.6,
                       help='宽高比阈值（默认1.6）')
    
    # 必要参数
    parser.add_argument('--name', type=str, default='optimized_track')
    parser.add_argument('--ablation', action='store_true')
    parser.add_argument('--cmc-method', default='sparseOptFlow')
    parser.add_argument('--with-reid', action='store_true', help='使用ReID')
    parser.add_argument('--mot20', action='store_true', help='MOT20模式')
    parser.add_argument('--fuse-score', action='store_true', help='融合分数')
    
    opt = parser.parse_args()
    
    # 重命名参数以匹配SMILEtrack期望
    opt.proximity_thresh = opt.proximity_thresh
    opt.appearance_thresh = opt.appearance_thresh
    
    print("=" * 60)
    print("优化版SMILEtrack - 解决遮挡后ID跳变问题")
    print("=" * 60)
    
    # 检查文件
    if not Path(opt.source).exists():
        print(f"错误: 视频文件不存在 {opt.source}")
        return
    
    if not Path(opt.weights).exists():
        print(f"错误: 权重文件不存在 {opt.weights}")
        print("请下载YOLOv8权重: yolov8n.pt")
        return
    
    # 获取视频信息
    cap = cv2.VideoCapture(opt.source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # 加载模型
    print(f"加载YOLOv8模型: {opt.weights}")
    model = YOLO(opt.weights)
    # YOLOv8类别名称
    names = model.names if hasattr(model, 'names') else [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]
    
    vehicle_classes = [2, 3, 5, 6, 7]  # 车辆类别
    print(f"使用模型: YOLOv8")
    print(f"车辆类别: {[names[i] for i in vehicle_classes if i < len(names)]}")
    
    # 创建优化版跟踪器
    tracker = OptimizedSMILEtrack(opt, frame_rate=fps)
    
    # 准备输出视频
    if opt.save_vid:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(opt.output, fourcc, fps, (width, height))
    
    frame_count = 0
    start_time = time.time()
    vehicle_detections = 0
    
    print("\n开始处理... (按 'q' 退出, 按 'p' 暂停)")
    print("-" * 60)
    
    # 打开视频
    cap = cv2.VideoCapture(opt.source)
    
    while True:
        ret, im0 = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        detections = []
        
        # YOLOv8推理
        results = model(
            im0,
            imgsz=opt.img_size,
            conf=opt.conf_thres,
            iou=opt.iou_thres,
            classes=opt.classes,
            verbose=False
        )
        
        # 处理YOLOv8检测结果
        for r in results:
            if r.boxes is not None:
                boxes = r.boxes
                for box in boxes:
                    # 获取边界框 [x1, y1, x2, y2]
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    
                    # 获取置信度
                    conf = float(box.conf[0].cpu().numpy())
                    
                    # 获取类别
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # 只添加车辆检测
                    if cls in vehicle_classes:
                        detections.append([x1, y1, x2, y2, conf, cls, 0.0])
                        vehicle_detections += 1
        
        detections = np.array(detections) if detections else np.array([])
        
        # 更新优化版跟踪器
        online_targets = tracker.update(detections, im0)
        
        # 绘制优化后的跟踪结果
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            
            if tlwh[2] * tlwh[3] > opt.min_box_area:
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                
                # 根据跟踪稳定性选择颜色
                if hasattr(t, 'tracklet_len') and t.tracklet_len > 10:
                    color = (0, 255, 0)  # 稳定跟踪 - 绿色
                else:
                    color = (0, 165, 255)  # 新跟踪 - 橙色
                
                # 绘制边界框
                cv2.rectangle(im0, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # 绘制跟踪轨迹
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                cv2.circle(im0, (center_x, center_y), 3, color, -1)
                
                # 显示ID和跟踪长度
                track_len = t.tracklet_len if hasattr(t, 'tracklet_len') else 0
                label = f"ID:{tid}({track_len})"
                cv2.putText(im0, label, (int(x1), int(y1)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 显示统计信息
        stats = tracker.get_statistics()
        
        info_lines = [
            f"Frame: {frame_count}",
            f"Tracks: {len(online_targets)}",
            f"ID Switches: {stats['id_switches']}",
            f"Switch Rate: {stats['switch_rate']:.2%}",
            f"Vehicles: {vehicle_detections}"
        ]
        
        y_offset = 30
        for i, line in enumerate(info_lines):
            cv2.putText(im0, line, (10, y_offset + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 显示参数建议
        if stats['switch_rate'] > 0.1:  # ID跳变率超过10%
            suggestion = "建议: 降低match-thresh或增加track-buffer"
            cv2.putText(im0, suggestion, (width-400, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 显示结果
        if opt.view_img:
            cv2.imshow('Optimized SMILEtrack', im0)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                cv2.waitKey(0)
        
        # 保存视频
        if opt.save_vid:
            out.write(im0)
        
        # 进度显示
        if frame_count % 50 == 0:
            elapsed = time.time() - start_time
            fps_actual = frame_count / elapsed
            print(f"进度: {frame_count}帧, FPS: {fps_actual:.1f}, "
                  f"跟踪: {len(online_targets)}, ID跳变: {stats['id_switches']}")
    
    # 清理
    cap.release()
    if opt.view_img:
        cv2.destroyAllWindows()
    if opt.save_vid:
        out.release()
    
    # 最终统计
    elapsed_total = time.time() - start_time
    final_stats = tracker.get_statistics()
    
    print("\n" + "=" * 60)
    print("✅ 处理完成!")
    print("=" * 60)
    print(f"📊 最终统计:")
    print(f"   总帧数: {frame_count}")
    print(f"   总时间: {elapsed_total:.1f}秒")
    print(f"   平均FPS: {frame_count/elapsed_total:.1f}")
    print(f"   最大跟踪数: {final_stats['total_tracks']}")
    print(f"   ID跳变次数: {final_stats['id_switches']}")
    print(f"   ID跳变率: {final_stats['switch_rate']:.2%}")
    print(f"   车辆检测数: {vehicle_detections}")
    
    print("\n🔧 参数优化建议:")
    if final_stats['switch_rate'] > 0.15:
        print("   ❌ ID跳变严重，建议:")
        print("     1. 降低匹配阈值: --match-thresh 0.5")
        print("     2. 增加轨迹缓冲区: --track-buffer 50")
        print("     3. 降低邻近阈值: --proximity-thresh 0.3")
    elif final_stats['switch_rate'] > 0.05:
        print("   ⚠️  有少量ID跳变，建议:")
        print("     1. 微调匹配阈值: --match-thresh 0.6")
        print("     2. 启用ReID: --with-reid")
    else:
        print("   ✅ ID稳定性良好")
    
    print("\n💡 常用优化组合:")
    print("   1. 减少ID跳变: --match-thresh 0.5 --track-buffer 50 --proximity-thresh 0.3")
    print("   2. 平衡模式: --match-thresh 0.6 --track-buffer 40 --proximity-thresh 0.4")
    print("   3. 高精度模式: --with-reid --appearance-thresh 0.15")
    print("   4. 实时模式: --match-thresh 0.8 --track-buffer 20")

if __name__ == "__main__":
    main()
