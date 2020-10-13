--- keyPrefix format is like "local3:it47:"
-- Returns a table of new keys inserted into Redis

local function getHowManyFiles()
  local i = 0
  local j = 0

  while true do
    i = i + 1
    local replyFilename = "client/w" .. i .. ".txt"
    local f = io.open(replyFilename, "r")
    if f == nil then
	break
    end
    f:close()
  end

  while true do
    j = j + 1
    local replyFilename = "server/w" .. j .. ".txt"
    local f = io.open(replyFilename, "r")
    if f == nil then
	break
    end
    f:close()
  end

  return (i-1), (j-1)
end

local function readWeightFiles(howManyClientFiles, howManyServerFiles, redisClient, keyPrefix)
  local i = 0
  local j = 0
  local fileKeys = {}
  local howManyFiles = howManyClientFiles + howManyServerFiles

  while i < howManyClientFiles do
    i = i + 1
    local replyFilename = "client/w" .. i .. ".txt"
    local f = io.open(replyFilename, "r")
    print(replyFilename)
    local fileGuts = f:read("*all")
    f:close()
    local key = keyPrefix .. "d" .. tostring(i) .. ":of" .. tostring(howManyFiles)
    print(key)
    redisClient:set(key, fileGuts)
    fileKeys[i] = key
  end

  while j < howManyServerFiles do
    j = j + 1
    local replyFilename = "server/w" .. j .. ".txt"
    local f = io.open(replyFilename, "r")
    print(replyFilename)
    local fileGuts = f:read("*all")
    f:close()
    local key = keyPrefix .. "d" .. tostring(i+j) .. ":of" .. tostring(howManyFiles)
    print(key)
    redisClient:set(key, fileGuts)
    fileKeys[i+j] = key
  end

  return fileKeys
end

local function readGradOutputsFile(redisClient, keyPrefix)
  local fileKeys = {}
  local key = keyPrefix .. "d1:of1"
  local f = io.open("client/gradOutputs.txt", "r")
  local fileGuts = f:read("*all")
  f:close()
  redisClient:set(key, fileGuts)
  fileKeys[1] = key
  return fileKeys
end

local function dumpGradientFiles(rawTable, redisClient)
  for seq, key in pairs(rawTable) do
    print("Dumping file No. " .. tostring(seq))
    local fileGuts = redisClient:get(key)
    local filename = "client/g" .. tostring(seq) .. ".txt"
    local f = assert(io.open(filename, "w"))
    f:write(fileGuts)
    f:close()
  end
end

local function dumpOutputAndTargets(rawTable, redisClient)
  for seq, key in pairs(rawTable) do
    local fileGuts = redisClient:get(key)
    local f = nil
    if seq == 1 then
      f = assert(io.open("server/output.txt", "w"))
    else
      f = assert(io.open("server/targets.txt", "w"))
    end
    f:write(fileGuts)
    f:close()
  end
end

--- Magic function to be run on all incoming data.
-- @param raw the unprocessed string or table of strings
-- @return function making table of new keys for files
local function processInboundString(raw, redisClient, clientState)
  print("Processing inbound files, please hold.")
  if clientState == 1 then
    local temp = os.clock()
    dumpGradientFiles(raw, redisClient)
    print((os.clock() - temp)*1000 .. " interruption time")
    os.execute("th update_weights.lua")
  else
    local temp = os.clock()
    dumpOutputAndTargets(raw, redisClient)
    print((os.clock() - temp)*1000 .. " interruption time")
    os.execute("th train_server.lua")
  end
end

local numberOfParticipants = 10
local maximumNumberOfIterations = 0 -- zero for infinite
local closurePuttingWeightsInRedis = function(redisClient)
  return function(keyPrefix)
    local numClientFiles, numServerFiles = getHowManyFiles()
   -- local temp = os.clock()
    local ret = readWeightFiles(numClientFiles, numServerFiles, redisClient, keyPrefix)
   -- print((os.clock() - temp)*1000 .. " interruption time")
    return ret
--  return readWeightFiles(20, redisClient, keyPrefix)
  end
end
local closurePuttingGradOutputsInRedis = function(redisClient)
  return function(keyPrefix)
    local temp = os.clock()
    local ret = readGradOutputsFile(redisClient, keyPrefix)
    print((os.clock() - temp)*1000 .. " interruption time")
    return ret
  end
end

return function() return
processInboundString,
numberOfParticipants,
maximumNumberOfIterations,
closurePuttingWeightsInRedis,
closurePuttingGradOutputsInRedis
end

