--! ~/torch/install/bin/th
-- running with th defines torch for us

local pis, N, maxIterations, weightReaderClosure, gradOutFileReaderClosure = require("processInboundString")()
pis = pis or function(s) return s end
N = N or 1
maxIterations = maxIterations or 0
local f = io.open("./info/studentName.txt", "r")
local fileGuts = f:read("*all")
local _,_,StudentNumber = fileGuts:find("^student Name:[%s]*(.*)")
local clientStates = {} --State of the clients. 0 means waiting on weight files, 1 means waiting on gradient outputs
for i=1,N do
  clientStates[i] = 0
end

local iterations = {}
for i=1,N do
  iterations[i] = 0
end

--region Redis
local redis = require("redis") -- https://github.com/nrk/redis-lua
local redisHost = "127.0.0.1" -- change me if not running locally
local redisPort = 6379
local redisAuth = "AAAAB3NzaC1yc2EAAAABJQAAAQEAnweC1sKULcZUP58BUHDQgYD0EOIZgerZ3zOznnw6JFqvxeANmILGaBDzKXVCgiJGPrMOkuRenZlbNimBcuTI6ophoFTpMCjirUPjBWsuIAz2vC9yp7ItTDmLaHmZ7XBoFL5rdscQpztYzKrbK11XGgKoRCBluiuJPYF64OTfjkZaiXmuCcoRrcrcoPxrPFqFnwteUJLUpGI71l1gpdqsB1hktNUaaAzcIRwc8NrtTrvFJJFZOskFEfjA5It6cdF1BqL0ezUxrMXu9JT8cAkFcctNnvGX1y6UNub882vvjj5Y8GvmJg7DwHX9PQqA3jC4YHiVtFgPhTcQ"

local redisClient = redis.connect(redisHost, redisPort)
redisClient:auth(redisAuth)
local weightFileReader = weightReaderClosure(redisClient)
local gradOutFileReader = gradOutFileReaderClosure(redisClient)
--endregion

require("lib")
local reply = lib.generateReply(redisClient)

local inspect = require("inspect")

local asyncRxCache = {}
local dead = false

local function finishedRxPosRun(chId, totalNumber)
  local tableOfRxData = asyncRxCache[chId]

  print("asyncRxCache is " .. inspect(asyncRxCache))

  -- dependent on #foo meaning counting foo[1], foo[2], ...
  if #tableOfRxData ~= totalNumber then
     return false
  end

  return true
end

local function updateThenTxNegRun(chId)
  print("keys of asyncRxCache are " .. inspect(lib.dumpKeys(asyncRxCache)))
  pis(asyncRxCache[chId], redisClient, clientStates[chId])
  redisClient:del(asyncRxCache[chId])
  asyncRxCache[chId] = nil
  print("keys of asyncRxCache are " .. inspect(lib.dumpKeys(asyncRxCache)))

  if clientStates[chId] == 1 then
    clientStates[chId] = 0

    -- check if enough iterations
    iterations[chId] = iterations[chId] + 1
    print("Finished iteration #" .. iterations[chId])
    if maxIterations > 0 and iterations[chId] >= maxIterations then
       return false
    end
 
    local keyPrefix = StudentNumber..":".."local" .. tostring(chId) .. ":it" .. tostring(iterations[chId]) .. ":"
    local listOfKeysToSend = weightFileReader(keyPrefix)
    reply(chId, listOfKeysToSend)

  else
    local keyPrefix = StudentNumber..":".."local" .. tostring(chId) .. ":it" .. tostring(iterations[chId]) .. ":"
    local listOfKeysToSend = gradOutFileReader(keyPrefix)
    reply(chId, listOfKeysToSend)

    clientStates[chId] = 1
  end

  return true
end

local function sendInitModel()
  for i=1,N do
      local keyPrefix = StudentNumber..":".."local" .. tostring(i) .. ":it" .. tostring(iterations[i]) .. ":"
      local listOfKeysToSend = weightFileReader(keyPrefix)

      reply(i, listOfKeysToSend)
  end
end

-- updateThenTxNegRun() -- replaces kickoff.(bat|sh)
sendInitModel()

while not dead do
  print("\t\tWaiting for next message!")
  local nextKey = redisClient:blpop("fodder", 0) -- format: see parseKey()
  nextKey = nextKey[2]
  print('Received message: '..nextKey)
  local chId, iter, sequenceNumber, totalNumber = lib.parseKey(nextKey)
  if chId ~= nil then
    if asyncRxCache[chId] == nil then
      asyncRxCache[chId] = {}
    end
    asyncRxCache[chId][sequenceNumber] = nextKey
    if finishedRxPosRun(chId, totalNumber) then
      print(" finished Rx pos run!")
      if not updateThenTxNegRun(chId) then return end
    end
  end
end
