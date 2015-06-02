using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks2
{
    public class Perceptron
    {
        private int numInput;
        private double[] inputs;
        private double[] weights;
        private double bias;
        private int _output;
        private Random rnd;

        public Perceptron(int numInput)
        {
            this.numInput = numInput;
            this.inputs = new double[numInput];
            this.weights = new double[numInput];
            this.rnd = new Random(0);
            InitializeWeights();
        }

        private void InitializeWeights()
        {
            double lo = -0.01;
            double hi = 0.01;
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = (hi - lo)*rnd.NextDouble() + lo;
            }
            bias = (hi - lo)*rnd.NextDouble() + lo;
        }

        public int ComputeOutput(double[] xValues)
        {
            if (xValues.Length != numInput)
            {
                throw new Exception("Bad xValues in ComputeOutput");
            }
            double sum = 0.0;
            for (int i = 0; i < xValues.Length; i++)
            {
                sum += xValues[i]*weights[i];
            }
            sum += bias;
            int result = Activation(sum);
            _output = result;
            return result;
        }

        private static int Activation(double v)
        {
            if (v > 0.0)
            {
                return 1;
            }
            return -1;
        }

        public double[] Train(double[][] trainData, double alpha, int maxEpochs)
        {
            int epoch = 0;
            double[] xValues = new double[numInput];
            int desired = 0;

            int[] sequence = new int[trainData.Length];
            for (int i = 0; i < sequence.Length; i++)
            {
                sequence[i] = i;
            }

            while (epoch < maxEpochs)
            {
                Shuffle(sequence);
                for (int i = 0; i < trainData.Length; i++)
                {
                    int idx = sequence[i];
                    Array.Copy(trainData[idx], xValues, numInput);
                    desired = (int) trainData[idx][numInput];
                    int computed = ComputeOutput(xValues);
                    Update(computed, desired, alpha); //modify weight and bias value
                }
                epoch++;
            }

            double[] result = new double[numInput+1];
            Array.Copy(weights, result, numInput);
            result[result.Length - 1] = bias;
            return result;
        }

        private void Shuffle(int[] sequence)
        {
            for (int i = 0; i < sequence.Length; i++)
            {
                int r = rnd.Next(i, sequence.Length);
                int tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }
        }

        private void Update(int computed, int desired, double alpha)
        {
            if (computed == desired)
            {
                return;
            }
            int delta = computed - desired;

            for (int i = 0; i < weights.Length; i++)
            {
                if (inputs[i] >= 0.0) // need to reduce weights
                {
                    weights[i] -= (alpha*delta*inputs[i]);
                }
                else if (inputs[i] < 0.0) //Need to reduce weights
                {
                    weights[i] += (alpha * delta * inputs[i]);
                }
            }
            bias -= (alpha*delta);
        }
    }
}
