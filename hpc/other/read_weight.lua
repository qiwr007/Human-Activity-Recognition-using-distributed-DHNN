
require "torch"
require "image"
require "math"


function loadtxt(weightfile, column)
	local datafile,dataerr=io.open(weightfile,"r")

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

local latest_weight1 = loadtxt('w1.txt', 32)
local latest_weight4 = loadtxt('w2.txt', 320)
local latest_weight8 = loadtxt('w3.txt', 585)
local latest_weight10 = loadtxt('w4.txt', 300)

local latest_weightbias1 = loadtxt('w5.txt', 1)
local latest_weightbias4 = loadtxt('w6.txt', 1)
local latest_weightbias8 = loadtxt('w7.txt', 1)
local latest_weightbias10 = loadtxt('w8.txt', 1)
--local latest_weight = loadtxt('weightMatrix.txt')

--return latest_weight1
return latest_weight1, latest_weight4, latest_weight8, latest_weight10, latest_weightbias1, latest_weightbias4, latest_weightbias8, latest_weightbias10