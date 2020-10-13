# Architecture summary
nn.ReLU
nn.SpatialConvolution(3 -> 64, 7x7, 1,1, 3,3)
nn.SpatialBatchNormalization (4D) (64)
nn.ReLU
nn.SpatialMaxPooling(2x2, 2,2)
nn.SpatialConvolution(64 -> 96, 5x5, 1,1, 2,2)
nn.SpatialBatchNormalization (4D) (96)
nn.ReLU
nn.SpatialMaxPooling(2x2, 2,2)
nn.SpatialConvolution(96 -> 128, 3x3, 1,1, 1,1)
nn.ReLU
nn.SpatialConvolution(128 -> 128, 3x3, 1,1, 1,1)
nn.ReLU
nn.SpatialConvolution(128 -> 196, 3x3, 1,1, 1,1)
nn.ReLU
nn.SpatialMaxPooling(2x2, 2,2)
nn.View(2352)
nn.Linear(2352 -> 320)
nn.ReLU
nn.Dropout(0.500000)
nn.View(-1, 8, 320)
LSTM
nn.View(-1, 256)
nn.Dropout(0.500000)
nn.Linear(256 -> 4)
nn.LogSoftMax

  1 : FloatTensor - size: 64x3x7x7        -- module1 weight
  2 : FloatTensor - size: 64              -- module1 bias
  3 : FloatTensor - size: 64              -- module2 weight
  4 : FloatTensor - size: 64              -- module2 bias
  5 : FloatTensor - size: 96x64x5x5       -- module5 weight
  6 : FloatTensor - size: 96              -- module5 bias
  7 : FloatTensor - size: 96              -- module6 weight
  8 : FloatTensor - size: 96              -- module6 bias
  9 : FloatTensor - size: 128x96x3x3      -- module9 weight
  10 : FloatTensor - size: 128            -- module9 bias
  11 : FloatTensor - size: 128x128x3x3    -- module11 weight
  12 : FloatTensor - size: 128            -- module11 bias
  13 : FloatTensor - size: 196x128x3x3    -- module13 weight
  14 : FloatTensor - size: 196            -- module13 bias
  15 : FloatTensor - size: 320x2352       -- module17 weight
  16 : FloatTensor - size: 320            -- module17 bias
  17 : FloatTensor - size: 576x1024       -- module21 weight
  18 : FloatTensor - size: 1024           -- module21 bias
  19 : FloatTensor - size: 4x256          -- module24 weight
  20 : FloatTensor - size: 4              -- module24 bias

