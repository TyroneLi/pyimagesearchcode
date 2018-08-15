from keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os

class TrainingMonitor(BaseLogger):
	def __init__(self, figPath, jsonPath=None, startAt=0):
		# 保存loss图片到指定路径，同时也保存json文件
		super(TrainingMonitor, self).__init__()
		self.figPath = figPath
		self.jsonPath = jsonPath
		# 开始模型开始保存的开始epoch
		self.startAt = startAt

	def on_train_begin(self, logs={}):
		# 初始化保存文件的目录dict
		self.H = {}
		# 判断是否存在文件和该目录
		if self.jsonPath is not None:
			if os.path.exists(self.jsonPath):
				self.H = json.loads(open(self.jsonPath).read())
				# 开始保存的epoch是否提供
				if self.startAt > 0:
					for k in self.H.keys():
						# 循环保存历史记录，从startAt开始
						self.H[k] = self.H[k][:self.startAt]

	def on_epoch_end(self, epoch, logs={}):
		# 不断更新logs和loss accuracy等等
		for (k, v) in logs.items():
			l = self.H.get(k, [])
			l.append(v)
			self.H[k] = l
		# 查看训练参数记录是否应该保存
		# 主要是看jsonPath是否提供
		if self.jsonPath is not None:
			f = open(self.jsonPath, 'w')
			f.write(json.dumps(self.H))
			f.close()
		# 保存loss acc等成图片
		if len(self.H["loss"]) > 1:
			N = np.arange(0, len(self.H["loss"]))
			plt.style.use("ggplot")
			plt.figure()
			plt.plot(N, self.H["loss"], label="train_loss")
			plt.plot(N, self.H["val_loss"], label="val_loss")
			plt.plot(N, self.H["acc"], label="train_acc")
			plt.plot(N, self.H["val_acc"], label="val_acc")
			plt.title("Training Loss and Accuracy [Epoch {}]".format(len(self.H["loss"])))
			plt.xlabel("Epoch #")
			plt.ylabel("Loss/Accuracy")
			plt.legend()
			plt.savefig(self.figPath)
			plt.close()
