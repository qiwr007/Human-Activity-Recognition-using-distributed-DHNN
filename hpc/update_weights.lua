require("torch")
require("nn")

local torchNetwork, torchWeightIndices, learningRate, _ ,_ = (require("xml_interface"))("client.xml", true, "client",0,0)

print("Flushing client-side network")

local temp = os.clock()
--Load the weights and gradients from disk
for index, torchIndex in ipairs(torchWeightIndices) do
	local biasIndex = index * 2
	local weightIndex = biasIndex - 1

	print("Loading client/w" .. weightIndex .. ".txt")
	torchNetwork.modules[torchIndex].weight = torch.load("client/w" .. weightIndex .. ".txt", "ascii")
	print("Loading client/w" .. biasIndex .. ".txt")
	torchNetwork.modules[torchIndex].bias = torch.load("client/w" .. biasIndex .. ".txt", "ascii")

	print("Loading client/g" .. weightIndex .. ".txt")
	torchNetwork.modules[torchIndex].gradWeight = torch.load("client/g" .. weightIndex .. ".txt", "ascii")
	print("Loading client/g" .. biasIndex .. ".txt")
	torchNetwork.modules[torchIndex].gradBias = torch.load("client/g" .. biasIndex .. ".txt", "ascii")

	os.remove("client/g" .. weightIndex .. ".txt")
	os.remove("client/g" .. biasIndex .. ".txt")
end
print((os.clock() - temp)*1000 .. " interruption time")

--Apply the gradients to our weights
torchNetwork:updateParameters(learningRate)

local temp = os.clock()
--Dump the new weights to the disk
for index, torchIndex in ipairs(torchWeightIndices) do
	local biasIndex = index * 2
	local weightIndex = biasIndex - 1
	local weights = torchNetwork.modules[torchIndex].weight
	local biases = torchNetwork.modules[torchIndex].bias

	print("Dumping client/w" .. weightIndex .. ".txt")
	torch.save("client/w" .. weightIndex .. ".txt", weights, "ascii")
	print("Dumping client/w" .. biasIndex .. ".txt")
	torch.save("client/w" .. biasIndex .. ".txt", biases, "ascii")
end
print((os.clock() - temp)*1000 .. " interruption time")
