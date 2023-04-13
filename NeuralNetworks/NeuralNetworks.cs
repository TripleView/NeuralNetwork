namespace NeuralNetworks;

/// <summary>
/// 神经网络
/// </summary>
public class NeuralNetworks
{

    public List<NeuronLayer> HiddenLayers { get; set; }
    /// <summary>
    /// 输出层
    /// </summary>
    public NeuronLayer OutputLayer { get; set; }
    /// <summary>
    /// 学习率
    /// </summary>
    public double LearningRate { get; set; } = 0.1;

    public int InputLayerNeuronNums { set; get; }


    public NeuralNetworks(int inputLayerNeuronNums, int hiddenLayerNeuronNums, int outputLayerNeuronNums, int hiddenLayerNums = 1, double? hiddenLayerBias = null, double? outLayerBias = null)
    {
        this.InputLayerNeuronNums = inputLayerNeuronNums;

        InitHiddenLayers(hiddenLayerNums, hiddenLayerNeuronNums, hiddenLayerBias);

        InitOutputLayer(outputLayerNeuronNums, outLayerBias);
    }

    /// <summary>
    /// 权重不能都是0
    /// </summary>
    /// <param name="weights"></param>
    private void CheckWeight(double[]? weights)
    {
        if (weights != null)
        {
            var allTrue = true;
            for (var i = 0; i < weights.Length; i++)
            {
                var weight = weights[i];
                allTrue &= weight == 0;
            }

            if (allTrue)
            {
                throw new Exception("权重不能全部为0");
            }
        }
    }



    /// <summary>
    /// 初始化隐藏层
    /// </summary>
    /// <param name="hiddenLayerWeights"></param>
    private void InitHiddenLayers(int hiddenLayerNums, int hiddenLayerNeuronNums, double? hiddenLayerBias = null)
    {
        this.HiddenLayers = new List<NeuronLayer>();
        var randon = new Random(1);
        var previewLayerNeuronNums = InputLayerNeuronNums;
        for (int x = 0; x < hiddenLayerNums; x++)
        {
            var hiddenLayer = new NeuronLayer(hiddenLayerNeuronNums, hiddenLayerBias.GetValueOrDefault(0), NeuronLayerType.隐含层);

            for (var i = 0; i < hiddenLayer.Neurons.Count; i++)
            {
                for (int j = 0; j < previewLayerNeuronNums; j++)
                {
                    var value = randon.NextDouble();
                    hiddenLayer.Neurons[i].Weights.Add(value);
                    hiddenLayer.Neurons[i].WeightQdValues.Add(0d);
                    hiddenLayer.Neurons[i].InputQdValues.Add(0d);


                }
            }
            this.HiddenLayers.Add(hiddenLayer);
            previewLayerNeuronNums = hiddenLayer.Neurons.Count;
        }


    }

    /// <summary>
    /// 初始化输出层
    /// </summary>
    /// <param name="outputLayerNeuronNums"></param>
    private void InitOutputLayer(int outputLayerNeuronNums, double? outLayerBias)
    {
        this.OutputLayer = new NeuronLayer(outputLayerNeuronNums, outLayerBias ?? 0);
        var randon = new Random(1);
        var k = 0;
        for (var i = 0; i < this.OutputLayer.Neurons.Count; i++)
        {
            for (int j = 0; j < this.HiddenLayers.Last().Neurons.Count; j++)
            {
                //权重必须在(0,1)之间
                var value = randon.NextDouble();
                this.OutputLayer.Neurons[i].Weights.Add(value);
                this.OutputLayer.Neurons[i].WeightQdValues.Add(0d);
                k++;
            }
        }
    }
    /// <summary>
    /// 正向传播
    /// </summary>
    public List<double> FeedForward(double[] inputs)
    {
        var lastInputs = inputs.ToList();
        for (int i = 0; i < this.HiddenLayers.Count; i++)
        {
            var hiddenLayer = this.HiddenLayers[i];

            var hiddenLayerOutputs = hiddenLayer.FeedForward(lastInputs);
            lastInputs = hiddenLayerOutputs;
        }

        var outputLayerOutput = this.OutputLayer.FeedForward(lastInputs);
        return outputLayerOutput;
    }

    public double CalculateTotalError(double[] inputs, double[] outputs)
    {
        this.FeedForward(inputs);
        var result = 0d;
        for (var i = 0; i < outputs.Length; i++)
        {
            var error = this.OutputLayer.Neurons[i].CalculateError(outputs[i]);
            result += error;
        }

        return result;
    }

    public void TrainHiddenLayer()
    {

    }

    public void Train(double[] inputs, double[] outputs)
    {
        this.FeedForward(inputs);

        var outCount = this.OutputLayer.Neurons.Count;
        for (int i = 0; i < outCount; i++)
        {
            for (int j = 0; j < this.OutputLayer.Neurons[i].Weights.Count; j++)
            {
                this.OutputLayer.Neurons[i].QdError2Weight(outputs[i], j);
                this.OutputLayer.Neurons[i].QdError2Bias(outputs[i]);
            }

        }

        NeuronLayer previewLayer = new NeuronLayer(1, 1);
        for (int x = this.HiddenLayers.Count - 1; x >= 0; x--)
        {
            var hiddenLayer = this.HiddenLayers[x];
            var hiddenNeuronCount = hiddenLayer.Neurons.Count;
            for (int i = 0; i < hiddenNeuronCount; i++)
            {
                if (x == this.HiddenLayers.Count - 1)
                {
                    var all = 0d;
                    for (int j = 0; j < outCount; j++)
                    {
                        all += this.OutputLayer.Neurons[j].QdError2Input(outputs[j], i);
                    }

                    for (int k = 0; k < hiddenLayer.Neurons[i].Weights.Count; k++)
                    {
                        hiddenLayer.Neurons[i].WeightQdValues[k] = all * hiddenLayer.Neurons[i].QdOutToWeight(k);
                        hiddenLayer.Neurons[i].InputQdValues[k] = all * hiddenLayer.Neurons[i].QdOutToInput(k);
                        hiddenLayer.Neurons[i].BiasQdValue = all * hiddenLayer.Neurons[i].Qd_Out2Net();
                    }
                }
                else
                {
                    var all = 0d;
                    var previewNeuronCount = previewLayer.Neurons.Count;
                    for (int j = 0; j < previewNeuronCount; j++)
                    {
                        all += previewLayer.Neurons[j].InputQdValues[i];
                    }

                    for (int k = 0; k < hiddenLayer.Neurons[i].Weights.Count; k++)
                    {
                        hiddenLayer.Neurons[i].WeightQdValues[k] = all * hiddenLayer.Neurons[i].QdOutToWeight(k);
                        hiddenLayer.Neurons[i].InputQdValues[k] = all * hiddenLayer.Neurons[i].QdOutToInput(k);
                        hiddenLayer.Neurons[i].BiasQdValue = all * hiddenLayer.Neurons[i].Qd_Out2Net();
                    }
                }

            }

            previewLayer = hiddenLayer;
        }


        //更新输出层
        for (int i = 0; i < outCount; i++)
        {
            for (int j = 0; j < this.OutputLayer.Neurons[i].Weights.Count; j++)
            {
                this.OutputLayer.Neurons[i].Weights[j] -=
                    this.LearningRate * this.OutputLayer.Neurons[i].WeightQdValues[j];
            }
            this.OutputLayer.Neurons[i].Bias -= this.LearningRate * this.OutputLayer.Neurons[i].BiasQdValue;
        }

        ////更新隐含层
        for (int x = this.HiddenLayers.Count - 1; x >= 0; x--)
        {
            var hiddenLayer = this.HiddenLayers[x];
            var hiddenCount = hiddenLayer.Neurons.Count;
            for (int i = 0; i < hiddenCount; i++)
            {
                for (int j = 0; j < hiddenLayer.Neurons[i].Weights.Count; j++)
                {
                    hiddenLayer.Neurons[i].Weights[j] -= this.LearningRate * hiddenLayer.Neurons[i].WeightQdValues[j];
                }
                hiddenLayer.Neurons[i].Bias -= this.LearningRate * hiddenLayer.Neurons[i].BiasQdValue;
            }

        }




    }
}