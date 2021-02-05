pip3 install -r requirements.txt 
python3 setup.py build_ext
export PYTHONPATH=pose_extractor/build/:$PYTHONPATH
python demo.py --model human-pose-estimation-3d.pth --video 0
# python3 demo.py --model ~/dev/models/human-pose-estimation-3d.pth --video 0
# source /opt/intel/openvino_2021/bin/setupvars.sh
# python3 scripts/convert_to_op --checkpoint-path ~/dev/models/human-pose-estimation-3d.pth 
# pip3 install networkx
# pip3 install defusedxml
# pip3 install test-generator==0.1.1
# python3  /opt/intel/openvino_2021/deployment_tools/model_optimizer/mo.py --input_model human-pose-estimation-3d.onnx --input=data --mean_values=data[128.0,128.0,128.0] --scale_values=data[255.0,255.0,255.0] --output=features,heatmaps,pafs
# python demo.py --model human-pose-estimation-3d.xml --device CPU --use-openvino --video 0
# python3 demo.py --model human-pose-estimation-3d.xml --device CPU --use-openvino --video 0
# python3 video_from_frames.py 
# python3 demo.py --model ~/dev/models/human-pose-estimation-3d.pth --video 0
# python3 demo.py --model human-pose-estimation-3d.xml --device CPU --use-openvino --video 0