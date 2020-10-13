===== Basic Usage =====

The interface format is based on XML. The root node should be the tag "NeuralNetwork".
The only subnode for the NeuralNetwork node should be a node tagged "Layers". If you
are following along, your file should look like

<NeuralNetwork>
	<Layers>
	</Layers>
</NeuralNetwork>

Within the
Layers node is an array of "Layer" nodes that define the various layers in the network.
These layers are parsed top to bottom, so the first layer you define will be the input
layer and the last layer you define will be the output layer. The layer nodes take one
attribute, called "layertype". This attribute defines the type of layer, using terms
from Torchs "nn" library. So if you wanted to make a temporal convolution layer, for
example, you would construct the layer like:

<Layer layertype="TemporalConvolution">
</Layer>

Within a Layer node, there are 1 or more Param nodes and an optional Func node.

The Param nodes define the parameters for initializing the layer. These parameters
correspond directly to the parameters used to initialize the layers in the nn library.
The value of these parameters is stored in an attribute called "value".

The Func node defines the activation function for the layer. The names of the activation
functions are taken from the nn library. To use a certain activation function, add an
attribute to the Func node called "name" and set its value to the name of the activation
function you want to use.

So for a temporal convolution layer, the parameters are the input frame size, the
output frame size, the kernel width of the convolution, and optionally, the step of
the convolution. You might choose the values 1, 10, 32, and 1 for these respectively.
You might also choose to have a rectified linear unit activation function. To create a
layer with these parameters, you would define a Layer node like:

<Layer layertype="TemporalConvolution">
	<Param value="1" />
	<Param value="10" />
	<Param value="32" />
	<Param value="1" />
	<Func name="ReLU" />
</Layer>

A list of the layer types and their parameters is given in Appendix A. A list of the
activation functions available is given in Appendix B. Full lists for both of these can
be found in the nn library documentation: https://nn.readthedocs.io/en/rtd/index.html

A full example of an interface file is given below. It has 3 Linear layers with
various parameters and activation functions:

<NeuralNetwork>
	<Layers>
		<Layer layertype="Linear">
			<Param value="561" />
			<Param value="300" />
			<Func name="ReLU" />
		</Layer>
		<Layer layertype="Linear">
			<Param value="761" />
			<Param value="302" />
			<Func name="LogSoftMax" />
		</Layer>
		<Layer layertype="Linear">
			<Param value="861" />
			<Param value="303" />
			<Func name="Sigmoid" />
		</Layer>
	</Layers>
</NeuralNetwork>

If you want to use a custom neural network layer not defined in the nn library, like
say LSTM, you can define one in a file. If you call the class name the same as the file
name and set the layertype of your layer to the class name, the interface will
automatically load your custom module and include it in the neural network. An LSTM
layer, for example, could be defined as a class called "LSTM" in the file "LSTM.lua",
and using it would be a matter of defining a layer like:

<Layer layertype="LSTM">
</Layer>



===== Appendix A =====
Note: this list is not complete! See the nn library docs for a complete list
https://nn.readthedocs.io/en/rtd/index.html

Linear layer
	Name: Linear
	Parameter 1: input dimension
	Parameter 2: output dimension

Sparse linear layer
	Name: Sparse
	Parameter 1: input dimension
	Parameter 2: output dimension

Max layer
	Name: Max
	Parameter: dimension

Add layer (adds bias to the incoming data)
	Name: Add
	Parameter 1: input dimension
	Parameter 2: scalar

Temporal convolution
	Name: TemporalConvolution
	Parameter 1: input frame size
	Parameter 2: output frame size
	Parameter 3: kernel width
	Parameter 4 (optional): step of convolution. 1 by default

Temporal max-pooling
	Name: TemporalMaxPooling
	Parameter 1: kernel width
	Parameter 2( optional): step of convolution. 1 by default

Spatial convolution
	Name: SpatialConvolution
	Parameter 1: number of expected input planes
	Parameter 2: number of output planes
	Parameter 3: kernel width
	Parameter 4: kernel height
	Parameter 5 (optional): horizontal (width-wise) step. 1 by default
	Parameter 6 (optional): vertical (height-wise) step. 1 by default

Spatial max-pooling
	Name: SpatialMaxPooling
	Parameter 1: kernel width
	Parameter 2: kernel height
	Parameter 3 (optional): horizontal (width-wise) step. 1 by default
	Parameter 4 (optional): vertical (height-wise) step. 1 by default

===== Appendix B =====
Note: this list is not complete! See the nn library docs for a complete list
https://nn.readthedocs.io/en/rtd/index.html

Rectified linear unit: ReLU

Sigmoid: Sigmoid

Logarithmic sigmoid: LogSigmoid

Hyperbolic tangent: Tanh

"Hard" hyperbolic tangent: HardTanh
