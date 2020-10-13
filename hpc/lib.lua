lib = {}

local function sgn(x)
  if x > 0 then return 1 else return 0 end
end

function lib.dumpKeys(tab)
  local keyset = {}
  local n = 0
  for k, v in pairs(tab) do
    n = n + 1
    keyset[n] = k
  end
  return keyset
end

--- Do not modify this! Fix your ASYNC_ORDER instead.
function lib.asyncOrderValid(ao)
  local have = {}
  assert( next(ao) ~= nil, "ASYNC_ORDER: list was empty")
  local idx = 1
  for i, dxn in pairs(ao) do
    assert( i == idx, "ASYNC_ORDER: please do not use indices, only a list of values { -1, -2, ...}")
    idx = idx + 1
    assert( dxn == math.floor(dxn), "ASYNC_ORDER: directions must be integers")
    assert( dxn ~= 0, "ASYNC_ORDER: zero is not a direction")
    if dxn < 0 then
      have[dxn] = true
    else
      assert( have[-dxn] ~= nil, "ASYNC_ORDER: for each number, negative must go before positive")
      have[-dxn] = nil -- duplicates are allowed, so long as -+-+ order (not --++ et al.)
    end
  end
  assert( next(have) == nil, "ASYNC_ORDER: unmatched directions: negative direction had no positive")
  return true
end

--- http://stackoverflow.com/a/641993
function table.shallow_copy(t)
  local t2 = {}
  for k,v in pairs(t) do
    t2[k] = v
  end
  return t2
end

--- Should look like this where list == { -1, -3, 1, -2, 2, 3 }
-- out = {
-- { 1, 2 },
-- { 3 },
-- { 4 },
-- { 5, 6 }
-- }
function lib.findSignRuns(list)
  local c = sgn(list[1])
  local out = {}
  local this = {}
  for idx, el in pairs(list) do
    if sgn(el) == sgn(c) then
      this[#this + 1] = idx
    else
      out[#out + 1] = table.shallow_copy(this)
      this = { idx }
      c = sgn(el)
    end
  end
  out[#out + 1] = table.shallow_copy(this)
  return out
end

function lib.generateReply(redisClient)
  return function(channelId, message)
    local ch = tostring(channelId)
    local downPipe = "down_" .. ch
    print("Publishing. channelId = " .. ch)
    if type(message) == "string" then
      redisClient:rpush(downPipe, message)
    elseif type(message) == "table" then
      for _, msg in pairs(message) do
        redisClient:rpush(downPipe, msg)
      end
    end
  end
end

function lib.parseKey(keyString)
  local f = io.open("./info/studentName.txt", "r")
  local fileGuts = f:read("*all")
  local _,_,studentName = fileGuts:find("^student Name:[%s]*(.*)")
  local _, _, chId, iter, sequenceNumber, totalNumber =
  keyString:find("^"..studentName..":local(%d+):it(%d+):[ud](%d+):of(%d+)$")
  chId = tonumber(chId)
  iter = tonumber(iter)
  sequenceNumber = tonumber(sequenceNumber)
  totalNumber = tonumber(totalNumber)
  if chId ~= nil and iter ~= nil and sequenceNumber ~= nil and totalNumber ~= nil then
    return chId, iter, sequenceNumber, totalNumber
  end
end

return lib
