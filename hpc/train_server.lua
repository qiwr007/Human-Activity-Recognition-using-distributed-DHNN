require("torch")
require("nn")

local torchNetwork, torchWeightIndices, learningRate, maxIterations, _ , _ = (require("xml_interface"))("server.xml", true, "server",0,0)

local outputs = torch.load("server/output.txt", "ascii")
local _targets = torch.load("server/targets.txt", "ascii")

local batchSize = outputs:size(1)

function update_network(network, prevOutputs, targets)
    local gradOutputs = {} 


    --for iteration = 1, maxIterations do
       network:zeroGradParameters()
       for t = 1, batchSize do
--	    network:zeroGradParameters()

	    local criterion = nn.ClassNLLCriterion()

            local input = prevOutputs[t]
            local target = targets[t]
            local pred = network:forward(input)

            local gradOutput = nn.utils.addSingletonDimension(network:updateGradInput(input, criterion:backward(network.output, target)))
            --local gradOutput = criterion:backward(network.output, target)

       --     criterion:forward(pred, target)
            --if gradOutputs[t] == nil then
                gradOutputs[t] = gradOutput:clone()
            --else
            --    gradOutputs[t] = gradOutputs[t] + network:updateGradInput(input, criterion:updateGradInput(network.output, target))
            --end
            network:accGradParameters(input, criterion.gradInput, 1)
--	    network:updateParameters(learningRate)
        end
        network:updateParameters(learningRate)
    --end

    return (torch.cat(gradOutputs))
end

print("Training server-side")

torch.save("client/gradOutputs.txt", update_network(torchNetwork, outputs, _targets), "ascii")

local temp = os.clock()

for index, torchIndex in ipairs(torchWeightIndices) do
   local biasIndex = index * 2
   local weightIndex = biasIndex - 1
   local weights = torchNetwork.modules[torchIndex].weight
   local biases = torchNetwork.modules[torchIndex].bias

   print("Dumping server/w" .. weightIndex .. ".txt")
   torch.save("server/w" .. weightIndex .. ".txt", weights, "ascii")
   print("Dumping server/w" .. biasIndex .. ".txt")
   torch.save("server/w" .. biasIndex .. ".txt", biases, "ascii")
end

print((os.clock() - temp)*1000 .. " interruption time")
