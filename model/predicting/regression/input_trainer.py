import numpy as np
import torch 
from model.utils import torch_scale_minmax

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class TrainInput_RegressionNet(torch.nn.Module):
	def __init__(self, model, target_score):
		super(TrainInput_RegressionNet, self).__init__()
		model=model.to(device)
		model.eval()
		for param in model.parameters():
			param.requires_grad = False
		self.target_score = target_score
		self.model = model

	def forward(self, x):
		score = self.model(x)
		return score      
		
def train_input(model, target_score, input_shape, random_size = 1000, epochs = 50, fluct_range = 0.05):
	rd_img = np.random.rand(random_size, input_shape)
	rd_img = torch.from_numpy(np.array(rd_img)).float()
	rd_img, tmp1, tmp2 = torch_scale_minmax(rd_img)

	output_shape = model(rd_img).shape
	target = target_score + np.random.uniform(low = target_score-fluct_range,high = target_score+fluct_range, size = output_shape)
	target = torch.from_numpy(np.array(target)).float()

	rd_img.requires_grad = True
	target.requires_grad = False
	net = TrainInput_RegressionNet(model, target)
	optimizer = torch.optim.Adam([rd_img], lr=0.001, weight_decay=0.05)
	for epoch in range(epochs): 
		output = net(rd_img)
		loss = torch.nn.functional.mse_loss(output, target)
		if (epoch ==0 or epoch % (epochs/5) == (epochs/5)-1):
			print ("        . Epoch %d: Loss = %.6f" % (epoch, loss))
		net.zero_grad()
		loss.backward()
		optimizer.step()
	return torch.mean(rd_img,0)  