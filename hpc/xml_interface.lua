require("torch")
require("nn")

local torchWeightIndices = {} --Some layers don't have weights and biases to train. This array keeps track of the ones that do

function getAddString(layertype)
    local checkString = "return nn." .. layertype .. " ~= nil"
    local checkFunction = loadstring(checkString)

    if checkFunction() then
        return "network:add(nn." .. layertype .. "("
    else
        checkString = "return " .. layertype .. " == nil"
        checkFunction = loadstring(checkString)

        if checkFunction() then
            require(layertype)
        end

        return "network:add(" .. layertype .. "("
    end
end

function create_network(network_file, load_weights, weight_dir,start_layer,end_layer)
    require("xml2lua")
    local xmlHandler = require("xmlhandler/tree")
    local xml = xml2lua.loadFile(network_file)
    local parser = xml2lua.parser(xmlHandler)
    local numTorchLayers = 0

    parser:parse(xml)

    network = nn.Sequential()

    learningRate = tonumber(xmlHandler.root.NeuralNetwork.LearningRate._attr.value) or 0.01
    maxIteration = tonumber(xmlHandler.root.NeuralNetwork.BatchIterations._attr.value) or 1
    maxEpochs = tonumber(xmlHandler.root.NeuralNetwork.Epochs._attr.value) or 5
    batchsize = tonumber(xmlHandler.root.NeuralNetwork.BatchSize._attr.value) or 10

    layers = xmlHandler.root.NeuralNetwork.Layers.Layer

    torchWeightIndices = {}

    --Build a Torch neural network based on the XML file
    local i = 1
    while layers[i] ~= nil do
        local addString = getAddString(layers[i]._attr.layertype)
        --local addString = "network:add(nn." .. layers[i]._attr.layertype .. "("
        local params = layers[i].Param
        local activationFunction = nil
        local hasParameters = false

        numTorchLayers = numTorchLayers + 1
        local weightedLayer = numTorchLayers

        if params ~= nil and params._attr ~= nil then
            addString = addString .. params._attr.value .. ", "
            hasParameters = true
        elseif params ~= nil then
            local j = 1
            while params[j] ~= nil do
                addString = addString .. params[j]._attr.value .. ", "
                hasParameters = true
                j = j + 1
            end
        end

        if layers[i].Func ~= nil then
            activationFunction = "network:add(nn." .. layers[i].Func._attr.name .. "())"
            numTorchLayers = numTorchLayers + 1
        end

        if hasParameters then
            addString = addString:sub(1,-3) .. "))" --Get rid of the trailing ", " and add a "))"
        else
            addString = addString .. "))" --No trailing commas to get rid of
        end

        local addStatement = loadstring(addString)

        addStatement()

        if network.modules[#(network.modules)].weight ~= nil then
            torchWeightIndices[#torchWeightIndices + 1] = weightedLayer
        end

        if activationFunction ~= nil then
            activationFunctionCompiled = loadstring(activationFunction)
            activationFunctionCompiled()
        end

        i = i + 1
    end

    local LayerNum = 0
    if load_weights == true then
       print("Load weights from " .. weight_dir)
       local temp = os.clock()
       --Load the weights and gradients from disk
       for index, torchIndex in ipairs(torchWeightIndices) do
          local biasIndex = (index+start_layer) * 2
          local weightIndex = biasIndex - 1

          network.modules[torchIndex].weight = torch.load(weight_dir .. "/w" .. weightIndex .. ".txt", "ascii")
          network.modules[torchIndex].bias = torch.load(weight_dir .. "/w" .. biasIndex .. ".txt", "ascii")
          LayerNum = index
       end
       end_layer = LayerNum

       print((os.clock() - temp)*1000 .. " interruption time")
    end

    return network,end_layer
end

function network_closure(network_file, load_weights, weight_dir,start_layer,end_layer)
    local network,end_layer = create_network(network_file, load_weights, weight_dir,start_layer,end_layer)
    return network, torchWeightIndices, learningRate, maxIteration,start_layer,end_layer
end

return network_closure
