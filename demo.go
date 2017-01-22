package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

func NeuralNetwork() *NN {
	rand.Seed(time.Now().UnixNano())
	data := make([][]float64, 3)
	for i := range data {
		data[i] = []float64{2*rand.NormFloat64() - 1}
	}
	return &NN{
		SynapticWeights: data,
	}
}

type NN struct {
	SynapticWeights [][]float64
}

// The neural network thinks.
func (nn *NN) Think(inputs [][]float64) [][]float64 {
	// Pass inputs through our neural network (our single neuron).
	return nn.Sigmoid(dot(inputs, nn.SynapticWeights))

}

// The Sigmoid function, which describes an S shaped curve.
// We pass the weighted sum of the inputs through this function to
// normalise them between 0 and 1.
func (nn *NN) Sigmoid(x [][]float64) [][]float64 {
	y := x[:][:]
	for i := 0; i < len(y); i++ {
		for j := 0; j < len(y[i]); j++ {
			y[i][j] = 1 / (1 + math.Exp(-y[i][j]))
		}
	}
	return y
}

// The derivative of the Sigmoid function.
// This is the gradient of the Sigmoid curve.
// It indicates how confident we are about the existing weight.
func (nn *NN) SigmoidDerivative(x [][]float64) [][]float64 {
	y := x[:][:]
	for i := 0; i < len(y); i++ {
		for j := 0; j < len(y[i]); j++ {
			y[i][j] = y[i][j] * (1 - y[i][j])
		}
	}
	return y
}

//We train the neural network through a process of trial and error.
// Adjusting the synaptic weights each time.
func (nn *NN) Train(trainingSetInputs [][]float64, trainingSetOutputs [][]float64, numberOfTrainingIterations int) {
	for i := 0; i < numberOfTrainingIterations; i++ {
		//Pass the training set through our neural network (a single neuron).
		output := nn.Think(trainingSetInputs)
		// Calculate the error (The difference between the desired output
		// and the predicted output).
		errorLoss := difference(trainingSetOutputs, output)
		// Multiply the error by the input and again by the gradient of the Sigmoid curve.
		// This means less confident weights are adjusted more.
		// This means inputs, which are zero, do not cause changes to the weights.
		/*training_set_inputs.T, error * self.__sigmoid_derivative(output)*/
		adjustment := dot(t(trainingSetInputs), multiple(errorLoss, nn.SigmoidDerivative(output)))

		// Adjust the weights.
		//nn.SynapticWeights += adjustment
		nn.SynapticWeights = addition(nn.SynapticWeights, adjustment)
	}
}

// UTILS for matrix op

// x - matrix, y - vector
func addition(x, y [][]float64) [][]float64 {
	var (
		lx  = len(x)
		lx0 = len(x[0])
		z   = make([][]float64, lx)
	)

	for i := 0; i < lx; i++ {
		zI := make([]float64, lx0)
		xI := x[i]
		yI0 := y[i][0]
		for j := 0; j < lx0; j++ {
			zI[j] = xI[j] + yI0
		}
		z[i] = zI
	}
	return z
}

// x - matrix, y - vector
func difference(x, y [][]float64) [][]float64 {
	var (
		lx  = len(x)
		lx0 = len(x[0])
		z   = make([][]float64, lx)
	)

	for i := 0; i < lx; i++ {
		zI := make([]float64, lx0)
		xI := x[i]
		yI0 := y[i][0]
		for j := 0; j < lx0; j++ {
			zI[j] = xI[j] - yI0
		}
		z[i] = zI
	}
	return z
}

func multiple(x, y [][]float64) [][]float64 {
	var (
		lx  = len(x)
		lx0 = len(x[0])
		z   = make([][]float64, lx)
	)

	for i := 0; i < lx; i++ {
		zI := make([]float64, lx0)
		xI := x[i]
		yI := y[i]
		for j := 0; j < lx0; j++ {
			zI[j] = xI[j] * yI[j]
		}
		z[i] = zI
	}
	return z
}

func t(in [][]float64) [][]float64 {
	outY := len(in)
	outX := len(in[outY-1])
	out := make([][]float64, outX)
	for j := 0; j < outX; j++ {
		out[j] = make([]float64, outY)
		for i := 0; i < outY; i++ {
			out[j][i] = in[i][j]
		}
	}
	return out
}

// Matrix() * Matrix() more info http://mathinsight.org/matrix_vector_multiplication
/*
fmt.Println(dot([][]float64{
		{0, 4, 4, 0},
		{0, 4, 4, 0},
		{0, 4, 4, 0},
	}, t([][]float64{{0, 4, 4, 0}})))
	output: [[32] [32] [32]]
*/
func dot(a, b [][]float64) [][]float64 {
	lca := len(a[0])
	lcb := len(b)
	if lca != lcb {
		panic(`len of rows A must be equal len of cols B(see here http://mathinsight.org/matrix_vector_multiplication)`)
	}
	lr := len(a)
	lc := len(b[0])

	out := make([][]float64, lr)
	for i := 0; i < lr; i++ {
		outI := make([]float64, lc)
		aI := a[i]
		for j := 0; j < lc; j++ {
			sum := 0.0000000000000
			for z := 0; z < lca; z++ {
				sum += aI[z] * b[z][j]
			}
			outI[j] = sum
		}
		out[i] = outI
	}
	return out
}

func main() {
	nn := NeuralNetwork()

	fmt.Printf("Random starting synaptic weights: %v \n", nn.SynapticWeights)
	// The training set. We have 4 examples, each consisting of 3 input values
	// and 1 output value.
	trainingSetInputs := [][]float64{
		{0, 0, 1},
		{1, 1, 1},
		{1, 0, 1},
		{0, 1, 1},
	}
	trainingSetOutputs := t([][]float64{
		{0, 1, 1, 0},
	})
	now := time.Now()
	// Train the neural network using a training set.
	// Do it 10,000 times and make small adjustments each time.
	nn.Train(trainingSetInputs, trainingSetOutputs, 10000)
	fmt.Printf("New synaptic weights after training: %v\n", nn.SynapticWeights)
	fmt.Printf("letancy %s\n", time.Since(now).String())
	// Test the neural network with a new situation.
	now = time.Now()
	result := nn.Think([][]float64{{1, 0, 0}})

	fmt.Printf("Considering new situation [1, 0, 0] -> ?: %v (letancy %s)\n", result, time.Since(now).String())
}
