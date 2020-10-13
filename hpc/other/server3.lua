require "torch"
require "nn"
require "math"
require 'image'
require 'optim'

require 'LSTM'

require 'LRCN'
require 'util.DataLoader'


local utils = require 'util.utils'
local unpack = unpack or table.unpack

local cmd = torch.CmdLine()

-- Dataset options
cmd:option('-numClasses', '5') -- necessary
cmd:option('-scaledHeight', '24') -- uses native height if unprovided
cmd:option('-scaledWidth', '32') -- uses native width if unprovided
cmd:option('-numChannels', 3)

-- Model options
cmd:option('-batchnorm', 1)
cmd:option('-dropout', 0.5)
cmd:option('-seqLength', 8)
cmd:option('-lstmHidden', 256)

-- Optimization options
cmd:option('-numEpochs', 3)
cmd:option('-learningRate', 0.01)



local opt = cmd:parse(arg)

-- Torch cmd parses user input as strings so we need to convert number strings to numbers
for k, v in pairs(opt) do
  if tonumber(v) then
    opt[k] = tonumber(v)
  end
end

-- Set up GPU
opt.dtype = 'torch.FloatTensor'

-- Initialize model and criterion
utils.printTime("Initializing LRCN")
local model = LRCN(opt):type(opt.dtype)

local newWeights = { torch.FloatTensor(64, 3, 7, 7):zero(), torch.FloatTensor(64):zero(), torch.FloatTensor(64):zero(), torch.FloatTensor(64):zero(), torch.FloatTensor(96, 64, 5, 5):zero(), torch.FloatTensor(96):zero(), torch.FloatTensor(96):zero(), torch.FloatTensor(96):zero(), torch.FloatTensor(128, 96, 3, 3):zero(), torch.FloatTensor(128):zero(), torch.FloatTensor(128, 128, 3, 3):zero(), torch.FloatTensor(128):zero(), torch.FloatTensor(196, 128, 3, 3):zero(), torch.FloatTensor(196):zero(), torch.FloatTensor(320, 2352):zero(), torch.FloatTensor(320):zero(), torch.FloatTensor(576, 1024):zero(), torch.FloatTensor(1024):zero(), torch.FloatTensor(5, 256):zero(), torch.FloatTensor(5):zero() }

-- main routine
function main()

  local config = {learningRate = opt.learningRate}
  local weights, grads = model:parameters()
  print(weights)
  local mWeights, mGrads = model:getParameters()
  w1 = torch.load("w1.txt", "ascii") --initial weight
  w2 = torch.load("w2.txt", "ascii") --initial weight
  w3 = torch.load("w3.txt", "ascii") --initial weight
  w4 = torch.load("w4.txt", "ascii") --initial weight
  w5 = torch.load("w5.txt", "ascii") --initial weight
  w6 = torch.load("w6.txt", "ascii") --initial weight
  w7 = torch.load("w7.txt", "ascii") --initial weight
  w8 = torch.load("w8.txt", "ascii") --initial weight
  w9 = torch.load("w9.txt", "ascii") --initial weight
  w10 = torch.load("w10.txt", "ascii") --initial weight
  w11 = torch.load("w11.txt", "ascii") --initial weight
  w12 = torch.load("w12.txt", "ascii") --initial weight
  w13 = torch.load("w13.txt", "ascii") --initial weight
  w14 = torch.load("w14.txt", "ascii") --initial weight
  w15 = torch.load("w15.txt", "ascii") --initial weight
  w16 = torch.load("w16.txt", "ascii") --initial weight
  w17 = torch.load("w17.txt", "ascii") --initial weight
  w18 = torch.load("w18.txt", "ascii") --initial weight
  w19 = torch.load("w19.txt", "ascii") --initial weight
  w20 = torch.load("w20.txt", "ascii") --initial weight
  --renew bias
  for t=1, 64 do
    model.modules[1].bias[t]=w2[t]
  end

  for t=1, 64 do
    model.modules[2].bias[t]=w4[t]
  end

  for t=1, 96 do
    model.modules[5].bias[t]=w6[t]
  end

  for t=1, 96 do
    model.modules[6].bias[t]=w8[t]
  end

  for t=1, 128 do
    model.modules[9].bias[t]=w10[t]
  end

  for t=1, 128 do
    model.modules[11].bias[t]=w12[t]
  end

  for t=1, 196 do
    model.modules[13].bias[t]=w14[t]
  end

  for t=1, 320 do
    model.modules[17].bias[t]=w16[t]
  end

  for t=1, 1024 do
    model.modules[21].bias[t]=w18[t]
  end

  for t=1, 5 do
    model.modules[24].bias[t]=w20[t]
  end

  --renew weight
  for t=1, 64 do
    for r=1, 3 do
      for e=1, 7 do
        for w=1, 7 do
          model.modules[1].weight[t][r][e][w]=w1[t][r][e][w]
        end
      end
    end
  end

  for t=1, 64 do
    model.modules[2].weight[t]=w3[t]
  end

  for t=1, 96 do
    for r=1, 64 do
      for e=1, 5 do
        for w=1, 5 do
          model.modules[5].weight[t][r][e][w]=w5[t][r][e][w]
        end
      end
    end
  end

  for t=1, 96 do
    model.modules[6].weight[t]=w7[t]
  end

  for t=1, 128 do
    for r=1, 96 do
      for e=1, 3 do
        for w=1, 3 do
          model.modules[9].weight[t][r][e][w]=w9[t][r][e][w]
        end
      end
    end
  end

  for t=1, 128 do
    for r=1, 128 do
      for e=1, 3 do
        for w=1, 3 do
          model.modules[11].weight[t][r][e][w]=w11[t][r][e][w]
        end
      end
    end
  end

  for t=1, 196 do
    for r=1, 128 do
      for e=1, 3 do
        for w=1, 3 do
          model.modules[13].weight[t][r][e][w]=w13[t][r][e][w]
        end
      end
    end
  end

  for t=1, 320 do
    for r=1, 2352 do
      model.modules[17].weight[t][r]=w15[t][r]
    end
  end

  for t=1, 576 do
    for r=1, 1024 do
      model.modules[21].weight[t][r]=w17[t][r]
    end
  end

  for t=1, 5 do
    for r=1, 256 do
      model.modules[24].weight[t][r]=w19[t][r]
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
  g9 = torch.load("g9.txt", "ascii") 
  g10 = torch.load("g10.txt", "ascii") 
  g11 = torch.load("g11.txt", "ascii") 
  g12 = torch.load("g12.txt", "ascii") 
  g13 = torch.load("g13.txt", "ascii") 
  g14 = torch.load("g14.txt", "ascii") 
  g15 = torch.load("g15.txt", "ascii") 
  g16 = torch.load("g16.txt", "ascii") 
  g17 = torch.load("g17.txt", "ascii")
  g18 = torch.load("g18.txt", "ascii") 
  g19 = torch.load("g19.txt", "ascii") 
  g20 = torch.load("g20.txt", "ascii") 

  --renew gradbias
  for t=1, 64 do
    model.modules[1].gradBias[t]=g2[t]
  end

  for t=1, 64 do
    model.modules[2].gradBias[t]=g4[t]
  end

  for t=1, 96 do
    model.modules[5].gradBias[t]=g6[t]
  end

  for t=1, 96 do
    model.modules[6].gradBias[t]=g8[t]
  end

  for t=1, 128 do
    model.modules[9].gradBias[t]=g10[t]
  end

  for t=1, 128 do
    model.modules[11].gradBias[t]=g12[t]
  end

  for t=1, 196 do
    model.modules[13].gradBias[t]=g14[t]
  end

  for t=1, 320 do
    model.modules[17].gradBias[t]=g16[t]
  end

  for t=1, 1024 do
    model.modules[21].gradBias[t]=g18[t]
  end

  for t=1, 5 do
    model.modules[24].gradBias[t]=g20[t]
  end

  --renew gradweight
  for t=1, 64 do
    for r=1, 3 do
      for e=1, 7 do
        for w=1, 7 do
          model.modules[1].gradWeight[t][r][e][w]=g1[t][r][e][w]
        end
      end
    end
  end

  for t=1, 64 do
    model.modules[2].gradWeight[t]=g3[t]
  end

  for t=1, 96 do
    for r=1, 64 do
      for e=1, 5 do
        for w=1, 5 do
          model.modules[5].gradWeight[t][r][e][w]=g5[t][r][e][w]
        end
      end
    end
  end

  for t=1, 96 do
    model.modules[6].gradWeight[t]=g7[t]
  end

  for t=1, 128 do
    for r=1, 96 do
      for e=1, 3 do
        for w=1, 3 do
          model.modules[9].gradWeight[t][r][e][w]=g9[t][r][e][w]
        end
      end
    end
  end

  for t=1, 128 do
    for r=1, 128 do
      for e=1, 3 do
        for w=1, 3 do
          model.modules[11].gradWeight[t][r][e][w]=g11[t][r][e][w]
        end
      end
    end
  end

  for t=1, 196 do
    for r=1, 128 do
      for e=1, 3 do
        for w=1, 3 do
          model.modules[13].gradWeight[t][r][e][w]=g13[t][r][e][w]
        end
      end
    end
  end

  for t=1, 320 do
    for r=1, 2352 do
      model.modules[17].gradWeight[t][r]=g15[t][r]
    end
  end

  for t=1, 576 do
    for r=1, 1024 do
      model.modules[21].gradWeight[t][r]=g17[t][r]
    end
  end

  for t=1, 5 do
    for r=1, 256 do
      model.modules[24].gradWeight[t][r]=g19[t][r]
    end
  end

  model:updateParameters(0.01)

  --renew bias
  for t=1, 64 do
    newWeights[2][t]=model.modules[1].bias[t]
  end

  for t=1, 64 do
    newWeights[4][t]=model.modules[2].bias[t]
  end

  for t=1, 96 do
    newWeights[6][t]=model.modules[5].bias[t]
  end

  for t=1, 96 do
    newWeights[8][t]=model.modules[6].bias[t]
  end

  for t=1, 128 do
    newWeights[10][t]=model.modules[9].bias[t]
  end

  for t=1, 128 do
    newWeights[12][t]=model.modules[11].bias[t]
  end

  for t=1, 196 do
    newWeights[14][t]=model.modules[13].bias[t]
  end

  for t=1, 320 do
    newWeights[16][t]=model.modules[17].bias[t]
  end

  for t=1, 1024 do
    newWeights[18][t]=model.modules[21].bias[t]
  end

  for t=1, 5 do
    newWeights[20][t]=model.modules[24].bias[t]
  end

  --renew weight
  for t=1, 64 do
    for r=1, 3 do
      for e=1, 7 do
        for w=1, 7 do
          newWeights[1][t][r][e][w]=model.modules[1].weight[t][r][e][w]
        end
      end
    end
  end

  for t=1, 64 do
    model.modules[2].weight[t]=w3[t]
  end

  for t=1, 96 do
    for r=1, 64 do
      for e=1, 5 do
        for w=1, 5 do
          newWeights[5][t][r][e][w]=model.modules[5].weight[t][r][e][w]
        end
      end
    end
  end

  for t=1, 96 do
    newWeights[7][t]=model.modules[6].weight[t]
  end

  for t=1, 128 do
    for r=1, 96 do
      for e=1, 3 do
        for w=1, 3 do
          newWeights[9][t][r][e][w]=model.modules[9].weight[t][r][e][w]
        end
      end
    end
  end

  for t=1, 128 do
    for r=1, 128 do
      for e=1, 3 do
        for w=1, 3 do
          newWeights[11][t][r][e][w]=model.modules[11].weight[t][r][e][w]
        end
      end
    end
  end

  for t=1, 196 do
    for r=1, 128 do
      for e=1, 3 do
        for w=1, 3 do
          newWeights[13][t][r][e][w]=model.modules[13].weight[t][r][e][w]
        end
      end
    end
  end

  for t=1, 320 do
    for r=1, 2352 do
      newWeights[15][t][r]=model.modules[17].weight[t][r]
    end
  end

  for t=1, 576 do
    for r=1, 1024 do
      newWeights[17][t][r]=model.modules[21].weight[t][r]
    end
  end

  for t=1, 5 do
    for r=1, 256 do
      newWeights[19][t][r]=model.modules[24].weight[t][r]
    end
  end

  for idx, w in pairs(newWeights) do
    torch.save("w" .. tostring(idx) .. ".txt", w, "ascii")
  end

--[[
  torch.save("w1.txt", newWeights[1], "ascii") 
  torch.save("w2.txt", newWeights[2], "ascii") 
  torch.save("w3.txt", newWeights[3], "ascii") 
  torch.save("w4.txt", newWeights[4], "ascii") 
  torch.save("w5.txt", newWeights[5], "ascii") 
  torch.save("w6.txt", newWeights[6], "ascii") 
  torch.save("w7.txt", newWeights[7], "ascii") 
  torch.save("w8.txt", newWeights[8], "ascii") 
  torch.save("w9.txt", newWeights[9], "ascii") 
  torch.save("w10.txt", newWeights[10], "ascii")
  torch.save("w11.txt", newWeights[11], "ascii")
  torch.save("w12.txt", newWeights[12], "ascii") 
  torch.save("w13.txt", newWeights[13], "ascii") 
  torch.save("w14.txt", newWeights[14], "ascii") 
  torch.save("w15.txt", newWeights[15], "ascii") 
  torch.save("w16.txt", newWeights[16], "ascii") 
  torch.save("w17.txt", newWeights[17], "ascii") 
  torch.save("w18.txt", newWeights[18], "ascii") 
  torch.save("w19.txt", newWeights[19], "ascii") 
  torch.save("w20.txt", newWeights[20], "ascii") --]]

end


main()


--  1 : FloatTensor - size: 64x3x7x7        -- module1 weight
--  2 : FloatTensor - size: 64              -- module1 bias
--  3 : FloatTensor - size: 64              -- module2 weight
--  4 : FloatTensor - size: 64              -- module2 bias
--  5 : FloatTensor - size: 96x64x5x5       -- module5 weight
--  6 : FloatTensor - size: 96              -- module5 bias
--  7 : FloatTensor - size: 96              -- module6 weight
--  8 : FloatTensor - size: 96              -- module6 bias
--  9 : FloatTensor - size: 128x96x3x3      -- module9 weight
--  10 : FloatTensor - size: 128            -- module9 bias
--  11 : FloatTensor - size: 128x128x3x3    -- module11 weight
--  12 : FloatTensor - size: 128            -- module11 bias
--  13 : FloatTensor - size: 196x128x3x3    -- module13 weight
--  14 : FloatTensor - size: 196            -- module13 bias
--  15 : FloatTensor - size: 320x2352       -- module17 weight
--  16 : FloatTensor - size: 320            -- module17 bias
--  17 : FloatTensor - size: 576x1024       -- module21 weight
--  18 : FloatTensor - size: 1024           -- module21 bias
--  19 : FloatTensor - size: 4x256          -- module24 weight
--  20 : FloatTensor - size: 4              -- module24 bias






