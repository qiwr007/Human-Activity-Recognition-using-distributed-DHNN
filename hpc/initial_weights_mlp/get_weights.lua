require "torch"
require "nn"
require "math"


-- global variables
learningRate = 0.01
innerIteration = 1
outerIteration = 1
fraction = 0.01
batchsize = 500

function subset(dataset, head, tail)
  local sub = {}
  local index = 0
  for i = head, tail do
    index = index + 1
    sub[index] = dataset[i]
  end
  function sub:size() return index end

  return sub
end

-- here we set up the architecture of the neural network
-- input raw data 561 * 1
--ANN with two hidden layer
function create_network()

  local ann = nn.Sequential(); 

  ann:add(nn.View(561 * 1))
  ann:add(nn.Linear(561, 300))
  ann:add(nn.ReLU())

  ann:add(nn.Linear(300, 100))
  ann:add(nn.ReLU())

  --output layer
  ann:add(nn.Linear(100, 12))
  ann:add(nn.ReLU())
  ann:add(nn.LogSoftMax())

  return ann
end

--[[ANN with no hidden layer
-- input raw data 561 * 1
function create_network()

  local ann = nn.Sequential(); 

  --output layer
  ann:add(nn.View(561 * 1))
  ann:add(nn.Linear(561, 12))
  ann:add(nn.ReLU())
  ann:add(nn.LogSoftMax())

  return ann
end
--]]

--
--[[ANN with one hidden layer
-- input raw data 561 * 1
function create_network()

  local ann = nn.Sequential(); 

  --output layer
  ann:add(nn.View(561 * 1))
  ann:add(nn.Linear(561, 100))
  ann:add(nn.ReLU())

  ann:add(nn.Linear(100, 12))
  ann:add(nn.ReLU())
  ann:add(nn.LogSoftMax())

  return ann
end
--]]

--[[
--ANN with one hidden layer
---- input raw data 561 * 1
function create_network()

  local ann = nn.Sequential();

  ann:add(nn.View(561 * 1))
  ann:add(nn.Linear(561, 300))
  ann:add(nn.LeakyReLU())
  
  ann:add(nn.Linear(300, 100))
  ann:add(nn.LeakyReLU())
  
  ann:add(nn.Linear(100, 12))
  ann:add(nn.LogSoftMax())

  return ann
  end
--]]

--[[
function create_network()

    local ann = nn.Sequential();
  
    ann:add(nn.View(900 * 1))
    ann:add(nn.Linear(900, 400))
    ann:add(nn.LeakyReLU())

    ann:add(nn.Linear(400, 60))
    ann:add(nn.LeakyReLU())

    ann:add(nn.Linear(60, 20))
    ann:add(nn.LeakyReLU())
    
    ann:add(nn.Linear(20, 2))
    ann:add(nn.LogSoftMax())
  
    return ann
    end

    --]]


function main()
  -- create the network
  network = create_network()

  --get the weights
  local weights, grads = network:parameters()
  for i=1,#weights do
    torch.save("w%d.txt"%i, weights[i], "ascii")
  end
end
	

main()