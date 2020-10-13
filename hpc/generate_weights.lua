require("torch")
require("nn")

local network_creator = require("xml_interface")
local serverStart = 0
local initialPath = "./initial_weights_mlp"
local serverEnd = 0
local torchNetwork, torchWeightIndices, _, _, serverStartLayer, serverEndLayer = network_creator("client.xml", true, initialPath, serverStart, serverEnd)

print("the client network architecture",torchNetwork)
--Dump the new weights to the disk
for index, torchIndex in ipairs(torchWeightIndices) do
	local biasIndex = index * 2
	local weightIndex = biasIndex - 1
	local weights = torchNetwork.modules[torchIndex].weight
	local biases = torchNetwork.modules[torchIndex].bias
	if not paths.filep("client") then
		paths.mkdir("client")
	end
	print("Creating client/w" .. weightIndex .. ".txt")
	torch.save("client/w" .. weightIndex .. ".txt", weights, "ascii")
	print("Creating client/w" .. biasIndex .. ".txt")
	torch.save("client/w" .. biasIndex .. ".txt", biases, "ascii")
end

local clientStart = serverEndLayer
local clientEnd = 0
local torchNetwork, torchWeightIndices, _, _,clinetStartLayer,clientEndLayer = network_creator("server.xml", true, initialPath, clientStart, clientEnd)

print("the server network architecture",torchNetwork)
--Dump the new weights to the disk
for index, torchIndex in ipairs(torchWeightIndices) do
	local biasIndex = index * 2
	local weightIndex = biasIndex - 1
	local weights = torchNetwork.modules[torchIndex].weight
	local biases = torchNetwork.modules[torchIndex].bias
	if not paths.filep("server") then
		paths.mkdir("server")
	end
	print("Creating server/w" .. weightIndex .. ".txt")
	torch.save("server/w" .. weightIndex .. ".txt", weights, "ascii")
	print("Creating server/w" .. biasIndex .. ".txt")
	torch.save("server/w" .. biasIndex .. ".txt", biases, "ascii")
end
