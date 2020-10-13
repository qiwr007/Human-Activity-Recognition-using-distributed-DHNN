require("xml2lua")
local handler = require("xmlhandler/tree")
local xml = xml2lua.loadFile("test.xml")
local parser = xml2lua.parser(handler)

parser:parse(xml)

print(handler.root.NeuralNetwork.Layers)
