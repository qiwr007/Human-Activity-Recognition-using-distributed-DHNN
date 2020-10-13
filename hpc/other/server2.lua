require "torch"
require "nn"
require "math"

batchsize=100

local newWeights = { torch.Tensor(300, 561):zero(), torch.Tensor(300):zero(), torch.Tensor(100, 300):zero(), torch.Tensor(100):zero(), torch.Tensor(50, 100):zero(), torch.Tensor(50):zero(), torch.Tensor(12, 50):zero(), torch.Tensor(12):zero() }

-- here we set up the architecture of the neural network
function create_network()
    local mlp = nn.Sequential()
    mlp:add(nn.View(561*1))
    mlp:add(nn.Linear(561,300))
    mlp:add(nn.ReLU())
    mlp:add(nn.Linear(300,100))
    mlp:add(nn.ReLU())
    mlp:add(nn.Linear(100,50))
    mlp:add(nn.ReLU())
    mlp:add(nn.Linear(50,12))
    mlp:add(nn.ReLU())
    mlp:add(nn.LogSoftMax())
    return mlp
end


-- main routine
function main()

  local network_server = create_network()
  local weights, grads = network_server:parameters()

  w1 = torch.load("w1.txt", "ascii") --initial weight
  w2 = torch.load("w2.txt", "ascii") --initial weight
  w3 = torch.load("w3.txt", "ascii") --initial weight
  w4 = torch.load("w4.txt", "ascii") --initial weight
  w5 = torch.load("w5.txt", "ascii") --initial weight
  w6 = torch.load("w6.txt", "ascii") --initial weight
  w7 = torch.load("w7.txt", "ascii") --initial weight
  w8 = torch.load("w8.txt", "ascii") --initial weight

  --renew bias
  for t=1, 300 do
     network_server.modules[2].bias[t]=w2[t]
  end

  for t=1, 100 do
     network_server.modules[4].bias[t]=w4[t]
  end

  for t=1, 50 do
     network_server.modules[6].bias[t]=w6[t]
  end

  for t=1, 12 do
     network_server.modules[8].bias[t]=w8[t]
  end
  --renew weight
  for t=1, 300 do
    for r=1, 561 do
       network_server.modules[2].weight[t][r]=w1[t][r]
    end
  end

  for t=1, 100 do
    for r=1, 300 do
       network_server.modules[4].weight[t][r]=w3[t][r]
    end
  end

  for t=1, 50 do
    for r=1, 100 do
       network_server.modules[6].weight[t][r]=w5[t][r]
    end
  end

  for t=1, 12 do
    for r=1, 50 do
       network_server.modules[8].weight[t][r]=w7[t][r]
    end
  end
  print("finish loading")

  g1 = torch.load("g1.txt", "ascii")
  g2 = torch.load("g2.txt", "ascii")
  g3 = torch.load("g3.txt", "ascii")
  g4 = torch.load("g4.txt", "ascii")
  g5 = torch.load("g5.txt", "ascii")
  g6 = torch.load("g6.txt", "ascii")
  g7 = torch.load("g7.txt", "ascii")
  g8 = torch.load("g8.txt", "ascii")

  --renew bias
  for t=1, 300 do
    network_server.modules[2].gradBias[t]=g2[t]
  end

  for t=1, 100 do
    network_server.modules[4].gradBias[t]=g4[t]
  end

  for t=1, 50 do
    network_server.modules[6].gradBias[t]=g6[t]
  end

  for t=1, 12 do
    network_server.modules[8].gradBias[t]=g8[t]
  end
  --renew weight
  for t=1, 300 do
    for r=1, 561 do
      network_server.modules[2].gradWeight[t][r]=g1[t][r]
    end
  end

  for t=1, 100 do
    for r=1, 300 do
      network_server.modules[4].gradWeight[t][r]=g3[t][r]
    end
  end

  for t=1, 50 do
    for r=1, 100 do
      network_server.modules[6].gradWeight[t][r]=g5[t][r]
    end
  end

  for t=1, 12 do
    for r=1, 50 do
      network_server.modules[8].gradWeight[t][r]=g7[t][r]
    end
  end


  network_server:updateParameters(0.01)

  --renew bias
  for t=1, 300 do
    newWeights[2][t] = network_server.modules[2].bias[t]
  end

  for t=1, 100 do
    newWeights[4][t] = network_server.modules[4].bias[t]
  end

  for t=1, 50 do
    newWeights[6][t] = network_server.modules[6].bias[t]
  end

  for t=1, 12 do
    newWeights[8][t] = network_server.modules[8].bias[t]
  end
  --renew weight
  for t=1, 300 do
    for r=1, 561 do
      newWeights[1][t][r] = network_server.modules[2].weight[t][r]
    end
  end

  for t=1, 100 do
    for r=1, 300 do
      newWeights[3][t][r] = network_server.modules[4].weight[t][r]
    end
  end

  for t=1, 50 do
    for r=1, 100 do
      newWeights[5][t][r] = network_server.modules[6].weight[t][r]
    end
  end

  for t=1, 12 do
    for r=1, 50 do
      newWeights[7][t][r] = network_server.modules[8].weight[t][r]
    end
  end

  for idx, w in pairs(newWeights) do
    torch.save("w" .. tostring(idx) .. ".txt", w, "ascii")
  end
  print(weights[6])

  os.remove('g1.txt')
  os.remove('g2.txt')
  os.remove('g3.txt')
  os.remove('g4.txt')
  os.remove('g5.txt')
  os.remove('g6.txt')
  os.remove('g7.txt')
  os.remove('g8.txt')

end

main()








