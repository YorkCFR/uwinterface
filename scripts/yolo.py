#!/usr/bin/env python3
# Purpose: Ros node to detect objects using tensorflow

import os
import os.path
import sys
import cv2
import random
import numpy as np

import rospy

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose, VisionInfo

from ultralytics import YOLO

random.seed(0)

class YOLONode:

	def __init__(self, model, description, threshold, iou, half_fp):
		self.model = YOLO(model)
		self.threshold = threshold
		self.iou = iou
		self.half_fp = half_fp
		
		self.bridge = CvBridge()
		
		self.colors = [
			tuple([random.randint(0, 255) for _ in range(3)]) for cls in self.model.names
		]
		
		# Publish metadata about the loaded model (i.e class names)
		model_metadata = [self.model.names[idx] for idx in self.model.names]
		metadata_location = "/%s/class_list" % os.path.basename(model)[:-3]
		rospy.set_param(metadata_location, model_metadata)
		
		vision_info = VisionInfo()
		vision_info.method = description
		vision_info.database_location = metadata_location
		vision_info.database_version = 0
		
		self.vision_info_pub = rospy.Publisher("vision_info", VisionInfo, queue_size=1, latch=True)
		self.object_pub = rospy.Publisher("object_detections", Detection2DArray, queue_size=1)
		self.image_pub = rospy.Publisher("labelled_image", CompressedImage, queue_size=1)
		
		rospy.Subscriber("image", CompressedImage, self.image_cb, queue_size=1)
		
		self.vision_info_pub.publish(vision_info)
		pass

	def image_cb(self, msg):
		
		if self.object_pub.get_num_connections() == 0 and self.image_pub.get_num_connections() == 0:
			return

		try:
			cv_image = self.bridge.compressed_imgmsg_to_cv2(msg)
		except CvBridgeError as e:
			print(e)

		cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
		
		results = self.model.predict(
			source=cv_image,
			conf=self.threshold,
			iou=self.iou,
			half=self.half_fp,
			verbose=False
		)

		# Publish raw detections
		if self.object_pub.get_num_connections() > 0:
			objArray = Detection2DArray()
			
			for result in results:
				for box_data in result.boxes:
					
					box = box_data.xywh[0]
					
					obj = Detection2D()
					obj.bbox.size_x = float(box[2])
					obj.bbox.size_y = float(box[3])
					obj.bbox.center.x = float(box[0])
					obj.bbox.center.y = float(box[1])
					
					obj_hypothesis = ObjectHypothesisWithPose()
					obj_hypothesis.id = int(box_data.cls)
					obj_hypothesis.score = float(box_data.conf)
					obj.results.append(obj_hypothesis)
					
					objArray.detections.append(obj)
					pass
				pass
			
			self.object_pub.publish(objArray)
			pass

		if self.image_pub.get_num_connections() > 0:
			for result in results:
				for box_data in result.boxes:
					box = [int(v.item()) for v in box_data.xyxy[0]]
					cv2.rectangle(cv_image, tuple(box[:2]), tuple(box[2:]), self.colors[int(box_data.cls)], thickness=4)
					pass
				pass
			
			cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
			try:
				self.image_pub.publish(self.bridge.cv2_to_compressed_imgmsg(cv_image))
			except CvBridgeError as e:
				print(e)
			pass
		pass

def main(args):
	# Initialize the node
	rospy.init_node('yolo_node', anonymous=False)

	arg_defaults = {
		"model" : "scubanetv2.pt",
		"description" : "SCUBANetV2 - YOLO",
		"threshold" : "0.25",
		"iou" : 0.7,
		"half_fp" : False
	}
	args = updateArgs(arg_defaults)

	node = YOLONode(**args)

	try:
		rospy.spin()
	except rospy.ROSInterruptException as e:
		print(e)
	pass

def updateArgs(arg_defaults):
	# Look up parameters starting in the node's private parameter space, but also search outer namespaces.
	args = {}
	for name, val in arg_defaults.items():
		full_name = rospy.search_param(name)
		print("name %s %s" % (name, full_name))
		if full_name is None:
			args[name] = val
		else:
			args[name] = rospy.get_param(full_name, val)
			print("We have args %s value %s" % (val, args[name]))
	return (args)

if __name__ == '__main__':
	main(sys.argv)
