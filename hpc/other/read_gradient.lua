
require "torch"
require "image"
require "math"


function loadtxt(gradientfile, column)
	local datafile,dataerr=io.open(gradientfile,"r")

	if dataerr then print("Open data file Error")	end

    print(datafile)
	local index = 0
    local dataset={}
	while true do
		local dataline=datafile:read('*l')
        --print (dataline)
		if dataline == nil then break end
		index=index+1
		--print (index)
		data1=dataline:split(' ')
		data=torch.Tensor(column)
		--print(data1)
		for i=1,column do
			data[i]=tonumber(data1[i])
			--print(data[i])
		end
		--print(data)
		dataset[index]={}
		for t=1, column do
		  dataset[index][t]=data[t]
		  --print(dataset[index][t])
		end
        
		
	end
	--print(dataset[4][5])
	function dataset:size() return index end
	return dataset
end

local latest_gradient1 = loadtxt('g1.txt', 32)
local latest_gradient4 = loadtxt('g2.txt', 320)
local latest_gradient8 = loadtxt('g3.txt', 585)
local latest_gradient10 = loadtxt('g4.txt', 300)
local latest_gradientbias1 = loadtxt('g5.txt', 1)
local latest_gradientbias4 = loadtxt('g6.txt', 1)
local latest_gradientbias8 = loadtxt('g7.txt', 1)
local latest_gradientbias10 = loadtxt('g8.txt', 1)
--local latest_weight = loadtxt('weightMatrix.txt')

--return latest_weight1
return latest_gradient1, latest_gradient4, latest_gradient8, latest_gradient10, latest_gradientbias1, latest_gradientbias4, latest_gradientbias8, latest_gradientbias10






