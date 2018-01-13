#! /usr/bin/env python
import numpy as np
import scipy.special
import matplotlib.pyplot

class neuralNetwork(object):

	def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes

		self.lr = learningrate

		self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
		self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

		#활성화 함수는 시그모이드
		self.activation_function = lambda x: scipy.special.expit(x)


	# 신경망 학습시키기
	def train(self, inputs_list, targets_list):
		# 입력 리스트를 2차원 행렬로 변환
		T_inputs = np.array(inputs_list).T
		T_targets = np.array(targets_list).T

		# 은닉 계층으로 들어오는 신호를 계산
		hidden_inputs = np.dot(self.wih, T_inputs)
		# 은닉 계층으로 나가는 신호를 계산
		hidden_outputs = self.activation_function(hidden_inputs)

		# 최종 출력 계층으로 들어오는 신호를 계산
		final_inputs = np.dot(self.who, hidden_outputs)
		# 최종 출력 계층에서 나가는 신호를 계산
		final_outputs = self.activation_function(final_inputs)

		# 출력 계층의 오차 (실제 값 - 계산 값)
		output_errors = T_targets - final_outputs
		# 은닉 계층의 오차는 가중치에 의해 나뉜 출력 계층의 오차들을 재조합해 계산
		hidden_errors = np.dot(self.who.T, output_errors)

		# 은닉 계층과 출력 계층 간의 가중치 업데이트
		self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))

		# 입력 계층과 은닉 계층 간의 가중치 업데이트
		self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(T_inputs))
		pass

	# 신경망에 질의 하기
	def query(self, inputs_list):
		# 입력리스트를 2차원 행렬로 변환(역핼렬)
		T_inputs = np.array(inputs_list, ndmin=2).T

		# 은닉 계층으로 들어오는 신호를 계산
		hidden_inputs = np.dot(self.wih, T_inputs)
		# 은닉 계층에서 나가는 신호 계산
		hidden_outputs = self.activation_function(hidden_inputs)
		# 최종 출력 계층으로 들어오는 신호를 계산
		final_inputs = np.dot(self.who, hidden_outputs)
		# 최종 출력 계층에서 나가는 신호를 계산
		final_outputs = self.activation_function(final_inputs)

		return final_outputs


def get_values():
	img_arr = None
	with open("data/mnist/mnist_train_100.csv") as fp:
		data_list = fp.readlines()
		all_values = data_list[0].split(',')
		img_arr = np.asfarray(all_values[1:]).reshape((28,28))
	return img_arr

if __name__ == "__main__":
	inputnodes = 3
	hiddennodes = 3
	outputnodes = 3
	lr = 0.3
	n = neuralNetwork(inputnodes, hiddennodes, outputnodes,lr)

	print(n.query([1.0, 0.5, -1.0]))

	arr = get_values()
	matplotlib.pyplot.imshow(arr, cmap='Greys', interpolation='None')
	matplotlib.pyplot.show()


