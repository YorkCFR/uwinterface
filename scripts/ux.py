#!/usr/bin/env python3
# Purpose: Ros node to detect objects using tensorflow

import os
import os.path
import sys
import cv2
import random
import numpy as np

import rospy

from std_msgs.msg import String, Header
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose, VisionInfo

from ultralytics import YOLO

random.seed(0)

MARGIN = 30
FONT_FAMILY = cv2.FONT_HERSHEY_SIMPLEX

def get_optimal_font_scale(text, width):
	for scale in reversed(range(0, 15, 1)):
		textSize = cv2.getTextSize(text, fontFace=FONT_FAMILY, fontScale=scale/10, thickness=4)
		new_width = textSize[0][0]
		if new_width <= width:
			return scale/10
		pass
	return 1

class UXNode:

	def __init__(self, font_size, delay):
		self.bridge = CvBridge()
		
		self.font_size = font_size
		self.delay = delay
		
		self.vision_info = None
		self.colors = []
		
		self.command = ""
		self.command_time = rospy.Time.now()
		
		self.uximage_pub = rospy.Publisher("ux_image", CompressedImage, queue_size=1)
		
		rospy.Subscriber("vision_info", VisionInfo, self.vision_info_cb, queue_size=1)
		rospy.Subscriber("image", CompressedImage, self.image_cb, queue_size=1)
		rospy.Subscriber("object_detections", Detection2DArray, self.detection_cb, queue_size=1)
		rospy.Subscriber("command", String, self.command_cb, queue_size=1)
		pass
	
	def command_cb(self, msg):
		self.command = msg.data
		self.command_time = rospy.Time.now()
		pass

	def vision_info_cb(self, msg):
		self.vision_info = msg
		
		self.class_list = rospy.get_param(msg.database_location)
		self.colors = [
			tuple([random.randint(0, 255) for _ in range(3)]) for cls in self.class_list
		]
		pass

	def image_cb(self, msg):
		self.image = msg
		pass
	
	def detection_cb(self, msg):
		
		if self.vision_info is None or self.image is None:
			return
		
		if self.uximage_pub.get_num_connections() == 0:
			return

		try:
			cv_image = self.bridge.compressed_imgmsg_to_cv2(self.image)
		except CvBridgeError as e:
			print(e)
		
		# Search for the dominant gesture amoung all detections
		dominant = None
		for detection in msg.detections:
			cls = detection.results[0].id
			score = detection.results[0].score
			
			if self.class_list[cls] in ["Off Hand", "Idle", "Head"]:
				continue
				
			if dominant is None or score > dominant.results[0].score:
				dominant = detection
				pass
			pass

		if self.uximage_pub.get_num_connections() > 0:
		
			if dominant is None:
				(_, label_height), _ = cv2.getTextSize("Idle", FONT_FAMILY, self.font_size, 2)
				cv2.putText(cv_image, "Idle", (MARGIN, label_height + MARGIN), FONT_FAMILY, self.font_size, 3*[255], thickness=4)
				pass
			else:
				idx = dominant.results[0].id
				cls = self.class_list[idx]
				bbox = [
					int(dominant.bbox.center.x - dominant.bbox.size_x/2),
					int(dominant.bbox.center.y - dominant.bbox.size_y/2),
					int(dominant.bbox.center.x + dominant.bbox.size_x/2),
					int(dominant.bbox.center.y + dominant.bbox.size_y/2),
				]
				cv2.rectangle(cv_image, tuple(bbox[:2]), tuple(bbox[2:]), self.colors[idx], thickness=4)
				(_, label_height), _ = cv2.getTextSize(cls, FONT_FAMILY, self.font_size, 2)
				cv2.putText(cv_image, cls, (MARGIN, label_height + MARGIN), FONT_FAMILY, self.font_size, 3*[255], thickness=4)
				pass
			
			if rospy.Time.now() - self.command_time < rospy.Duration(self.delay):
				sentence = self.command.split(" ")
				color = [255, 0, 255]
				if len(sentence) > 1:
					if sentence[-1] == "Ok":
						color = [0, 0, 255]
						pass
					if sentence[0] == "You":
						color = [0, 255, 0]
						pass
					pass
				scale = get_optimal_font_scale(self.command, 500)
				(label_width, label_height), _ = cv2.getTextSize(self.command, FONT_FAMILY, scale, thickness=3)
				org = (int(cv_image.shape[0]/2 - label_width/2), int(cv_image.shape[1]/2 - label_height - MARGIN))
				cv2.putText(cv_image, self.command, org, FONT_FAMILY, scale, color, thickness=3)
				pass
			
			try:
				self.uximage_pub.publish(self.bridge.cv2_to_compressed_imgmsg(cv_image))
			except CvBridgeError as e:
				print(e)
			pass
		pass

def main(args):
	# Initialize the node
	rospy.init_node('ux_node', anonymous=False)

	arg_defaults = {
		"font_size" : 3,
		"delay" : 5
	}
	args = updateArgs(arg_defaults)

	node = UXNode(**args)

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
